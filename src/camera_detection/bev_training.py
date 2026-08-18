
import argparse
import random
from pathlib import Path
from typing import Dict, List
import wandb
import json
import csv
 
import numpy as np
import torch
from PIL import Image
from torch.amp import autocast
from torch.utils.data import DataLoader
 
#training del ramo camera in bev
#carica il backbone allenato nel warmup e allena backbone (lr basso) + head BEV (lr alto). Lifting e pooling sono fissi ma differenziabili.
 
from bev_config import BEVConfig
from bev_dataset import BEVDataset, cone_visible, load_calib, load_cones_3d, world_to_grid, COLOR_TO_CLASS
from bev_model import CameraBEVNet
from losses import WarmupLoss                                    #riciclato da  warmup
from logger import TrainingLogger                                #riciclato da  warmup
from metrics import extract_peaks_from_heatmap, match_detections_to_gt, compute_metrics, compute_color_metrics, compute_ap, confusion_matrix_from_tp, DISTANCE_BINS_BEV  #riciclato da  warmup
from visualization import denormalize_image, color_heatmap       #riciclato da  warmup



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

 
#VALIDATION-------------------------------------------------------------------

class BEVValidationAccumulator:

    def __init__(self, cfg: BEVConfig):
        self.cfg = cfg
        self.reset()

    
    def reset(self):
        self.all_tp, self.all_fp, self.all_fn = [], [], []
        self.all_tp_agnostic = []


    def update(self, heatmap_logits, offset_pred, sample_ids: List[str]):
        cfg = self.cfg
        probs = torch.sigmoid(heatmap_logits)
 
        for b in range(probs.shape[0]):
            detections = extract_peaks_from_heatmap(
                probs[b].cpu(), offset_pred[b].cpu(),
                stride=1, threshold=cfg.detection_threshold_val,
                max_detections=300,
            )
 
            scene_id, frame_stem = sample_ids[b].split("/")
            labels_path = Path(cfg.dataset_root) / "scenes" / scene_id / "labels" / f"{frame_stem}.json"
            calib_path = Path(cfg.dataset_root) / "scenes" / scene_id / "calib.yaml"

            calib = load_calib(calib_path)

            gt = []
            for c in load_cones_3d(labels_path):
                if COLOR_TO_CLASS.get(c["color"]) is None:
                    continue

                #filtro fov
                if not cone_visible(c, calib):
                    continue

                row, col = world_to_grid(c["x"], c["y"], cfg)
                #i coni fuori griglia sono  fuori dal dominio quindi non contano come FN
                if not (0 <= row < cfg.bev_H and 0 <= col < cfg.bev_W):
                    continue
                gt.append({
                    "color": c["color"],
                    "center_px": (col, row),
                    "depth_m": float(c["x"]),
                    "fully_in_image": True,
                })
 
            radius_cells = cfg.match_radius_m / cfg.resolution

            tp, fp, fn = match_detections_to_gt(
                detections, gt,
                match_radius_px=radius_cells,
                color_to_class=COLOR_TO_CLASS,
            )
            self.all_tp.extend(tp)
            self.all_fp.extend(fp)
            self.all_fn.extend(fn)

            #secondo matching solo posizione
            tp_agn, _, _ = match_detections_to_gt(
                detections, gt,
                match_radius_px=radius_cells,
                color_to_class=COLOR_TO_CLASS,
                class_agnostic=True,
            )
            self.all_tp_agnostic.extend(tp_agn)

    def confusion_matrix(self) -> np.ndarray:
        return confusion_matrix_from_tp(self.all_tp_agnostic, self.cfg.num_classes)
 
    def compute(self) -> Dict[str, float]:
        m = compute_metrics(self.all_tp, self.all_fp, self.all_fn, distance_bins=DISTANCE_BINS_BEV, num_classes=self.cfg.num_classes)
        m.update(compute_color_metrics(self.all_tp_agnostic, self.cfg.num_classes))
        m["ap"] = compute_ap(self.all_tp, self.all_fp, num_gt=len(self.all_tp) + len(self.all_fn))
        return m


#VISUALIZATION----------------------------------------------------------------


from pathlib import Path
import matplotlib
matplotlib.use("Agg")   # backend senza display, per server headless
import matplotlib.pyplot as plt
import torch

from bev_dataset import COLOR_TO_CLASS, load_cones_3d, world_to_grid, grid_to_world
from metrics import extract_peaks_from_heatmap

_CLASS_COLORS = {0: "#2b6cb0", 1: "#d4a017", 2: "#dd6b20"}

def _draw_bev_panel(ax, cones, cfg, title, gt_cones=None, fov_params=None):
    for r in range(10, int(cfg.x_max) + 1, 10):          # anelli di distanza
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color="0.85", lw=0.8, zorder=0))
        ax.text(0, r, f"{r}m", color="0.55", fontsize=7, ha="center", va="bottom", zorder=1)
    ax.plot(0, 0, marker="^", color="0.15", markersize=11, zorder=6)   # veicolo
    
    #DISEGNO FOV TELECAMERA
    if fov_params is not None:
        cx, cy = fov_params["cam_x"], fov_params["cam_y"]
        slope = fov_params["slope"]
        
        x_end = cfg.x_max
        #+y sinistra 
        #-y destra
        y_left = cy + (x_end - cx) * slope
        y_right = cy - (x_end - cx) * slope
        
        ax.plot([cy, y_left], [cx, x_end], color="lime", linestyle="--", linewidth=1.5, alpha=0.8, zorder=2)
        ax.plot([cy, y_right], [cx, x_end], color="lime", linestyle="--", linewidth=1.5, alpha=0.8, zorder=2)
        
        #area visibile
        ax.fill_betweenx([cx, x_end], [cy, y_right], [cy, y_left], color="lime", alpha=0.06, zorder=1)

    if gt_cones is not None:                             # gt in trasparenza (pannello pred)
        for c in gt_cones:
            ax.scatter(c["y"], c["x"], s=110, facecolors="none", edgecolors=_CLASS_COLORS.get(c["cls"], "#808080"), linewidths=1.3, alpha=0.45, zorder=3)
                       
    for c in cones:
        ax.scatter(c["y"], c["x"], s=42, color=_CLASS_COLORS.get(c["cls"], "#808080"), edgecolors="k", linewidths=0.4, zorder=4)
                   
    ax.set_xlim(cfg.y_max, cfg.y_min)    # +y (sinistra del veicolo) a sinistra
    ax.set_ylim(cfg.x_min, cfg.x_max)    # veicolo in basso, lontano in alto
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("y laterale (m)")
    ax.set_ylabel("x avanti (m)")
    ax.grid(True, color="0.93", lw=0.5)


def save_bev_visualizations(pred_heatmap_logits, pred_offset, sample_ids,
                            cfg, output_dir, epoch, max_to_save=8):
    subdir = f"epoch_{epoch:03d}" if isinstance(epoch, int) else str(epoch)
    output_dir = Path(output_dir) / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    probs = torch.sigmoid(pred_heatmap_logits)

    for i in range(min(max_to_save, probs.shape[0])):
        scene_id, frame_stem = sample_ids[i].split("/")

        labels_path = Path(cfg.dataset_root) / "scenes" / scene_id / "labels" / f"{frame_stem}.json"
        calib_path = Path(cfg.dataset_root) / "scenes" / scene_id / "calib.yaml"
        
        #carichiamo la calibrazione per estrarre i parametri del fov
        calib = load_calib(calib_path)
        
        #pendenza: tan(theta) = cx / fx
        fov_slope = calib["K"][2] / calib["K"][0] 
        #posizione della telecamera nel frame bev
        cam_x = float(calib["T"][0, 3])
        cam_y = float(calib["T"][1, 3])
        
        fov_params = {
            "cam_x": cam_x,
            "cam_y": cam_y,
            "slope": float(fov_slope)
        }

        #carichiamo tutti i coni
        gt_cones = []
        for c in load_cones_3d(labels_path):
            cls = COLOR_TO_CLASS.get(c["color"])
            row, col = world_to_grid(c["x"], c["y"], cfg)
            if cls is not None and 0 <= row < cfg.bev_H and 0 <= col < cfg.bev_W:
                gt_cones.append({"x": c["x"], "y": c["y"], "cls": cls})

        #pred: picchi estratti (stesse detection della validation) -> metri
        dets = extract_peaks_from_heatmap(
            probs[i].cpu(), pred_offset[i].cpu(), stride=1, threshold=cfg.detection_threshold
        )
        pred_cones = []
        for d in dets:
            x, y = grid_to_world(d["y"], d["x"], cfg)   # d["y"]=riga, d["x"]=colonna
            pred_cones.append({"x": x, "y": y, "cls": d["class_id"]})

        fig, axs = plt.subplots(1, 2, figsize=(11, 6.5))
    
        _draw_bev_panel(axs[0], gt_cones, cfg, "BEV GT (Tutti i coni)", fov_params=fov_params)
        _draw_bev_panel(axs[1], pred_cones, cfg, "BEV pred (GT in trasparenza)", gt_cones=gt_cones, fov_params=fov_params)
        
        fig.suptitle(f"{sample_ids[i]}  -  epoch {epoch:03d}", fontsize=11)
        fig.tight_layout()
        fig.savefig(output_dir / f"{sample_ids[i].replace('/', '_')}.png",
                    dpi=110, bbox_inches="tight")
        plt.close(fig)
        

_CLASS_NAMES = {0: "blue", 1: "yellow", 2: "orange_small"}


def save_confusion_matrix(matrix, cfg, output_dir, epoch):
    #matrice di confusione dei colori, righe = GT, colonne = predetto
    

    output_dir = Path(output_dir) / f"epoch_{epoch:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    n = cfg.num_classes
    names = [_CLASS_NAMES.get(c, f"c{c}") for c in range(n)]

    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(1.6 * n + 2.2, 1.6 * n + 1.8))
    im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    ax.set_xlabel("predetto"); ax.set_ylabel("ground truth")

    for i in range(n):
        for j in range(n):
            
            color = "white" if normalized[i, j] > 0.55 else "black"
            ax.text(j, i, f"{normalized[i, j]:.3f}\n({matrix[i, j]})",
                    ha="center", va="center", fontsize=9, color=color)

    accuracy = np.trace(matrix) / max(matrix.sum(), 1)
    ax.set_title(f"Confusione colore - epoch {epoch:03d}\n"
                 f"accuracy {accuracy:.4f} su {matrix.sum()} coni localizzati",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

#SETUP AND DATA LOADER-------------------------------------------------------------


def build_dataloaders(cfg: BEVConfig):

    color_jitter_params = {
        "brightness": cfg.color_jitter_brightness,
        "contrast": cfg.color_jitter_contrast,
        "saturation": cfg.color_jitter_saturation,
        "hue": cfg.color_jitter_hue,
    }

    train_dataset = BEVDataset(cfg, cfg.train_split_file, augment=True, color_jitter_params=color_jitter_params)
    val_dataset = BEVDataset(cfg, cfg.val_split_file, augment=False)
    test_dataset = BEVDataset(cfg, cfg.test_split_file, augment=False)
 
    print(f"Train: {len(train_dataset)} sample, Val: {len(val_dataset)} sample, Test: {len(test_dataset)} sample")
    print(f"Griglia BEV: {cfg.bev_H}x{cfg.bev_W} celle @ {cfg.resolution} m, "
          f"x [{cfg.x_min},{cfg.x_max}]m, y [{cfg.y_min},{cfg.y_max}]m")
 
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True, persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers > 0)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,num_workers=cfg.num_workers, pin_memory=True,)

    return train_loader, val_loader, test_loader
 
 
def build_scheduler(optimizer, cfg: BEVConfig, steps_per_epoch: int):

    #identico al warmup (warmup lineare + cosine annealing)
    total_steps = cfg.num_epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
 
    def lr(step):
        if step < warmup_steps:
            return float(step) / float(max(warmup_steps, 1))
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
 
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr)
 
 
def to_device(batch, device):

    inputs = (batch["image"].to(device, non_blocking=True),
              batch["depth"].to(device, non_blocking=True),
              batch["K"].to(device, non_blocking=True),
              batch["T"].to(device, non_blocking=True))
    
    targets = {k: batch[k].to(device, non_blocking=True)
               for k in ("heatmap", "offset", "offset_mask")}
    
    return inputs, targets


#TRAIN, VALIDATION, TEST---------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, cfg,
                    epoch, global_step, logger):
    model.train()
    epoch_losses = {"loss_total": 0.0, "loss_focal": 0.0, "loss_offset": 0.0}
    num_batches = 0
 
    for batch in loader:
        inputs, targets = to_device(batch, "cuda")
 
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = model(*inputs)
            loss, log_dict = loss_fn(predictions, targets)
 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        scheduler.step()
 
        for k, v in log_dict.items():
            epoch_losses[k] += v
        num_batches += 1
 
        lrs = [pg["lr"] for pg in optimizer.param_groups]
        logger.log_step(epoch, global_step, log_dict, lrs[0], lrs[1] if len(lrs) > 1 else lrs[0])

        if cfg.use_wandb and global_step % cfg.log_every_n_steps == 0:
            wandb.log({
                "train/loss_total": log_dict["loss_total"],
                "train/loss_focal": log_dict["loss_focal"],
                "train/loss_offset": log_dict["loss_offset"],
                "lr/backbone": lrs[0],
                "lr/head": lrs[1] if len(lrs) > 1 else lrs[0],
                "epoch": epoch
            }, step=global_step)


        global_step += 1
 
    for k in epoch_losses:
        epoch_losses[k] /= max(num_batches, 1)
    return global_step, epoch_losses


@torch.no_grad()
def validate(model, loader, loss_fn, val_accumulator, cfg, epoch):
    model.eval()
    val_accumulator.reset()
 
    sum_losses = {"loss_total": 0.0, "loss_focal": 0.0, "loss_offset": 0.0}
    num_batches = 0
 
    for batch_idx, batch in enumerate(loader):
        inputs, targets = to_device(batch, "cuda")
        sample_ids = batch["sample_id"]
 
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = model(*inputs)
            loss, log_dict = loss_fn(predictions, targets)
 
        for k, v in log_dict.items():
            sum_losses[k] += v
        num_batches += 1
 
        val_accumulator.update(predictions["heatmap_logits"].float(),
                               predictions["offset_pred"].float(), sample_ids)
 
        if cfg.save_visualizations and batch_idx == 0:
            save_bev_visualizations(
                predictions["heatmap_logits"].float(), predictions["offset_pred"].float(),
                list(sample_ids), cfg, "../visualizations_bev", epoch,
                max_to_save=cfg.num_visualizations_per_val,
            )
 
    for k in sum_losses:
        sum_losses[k] /= max(num_batches, 1)

    result = {f"val_{k}": v for k, v in sum_losses.items()}
    result.update({f"val_{k}": v for k, v in val_accumulator.compute().items()})

    if cfg.save_visualizations:
        save_confusion_matrix(val_accumulator.confusion_matrix(), cfg, "../visualizations_bev", epoch)

    return result


@torch.no_grad()
def test(model, loader, loss_fn, cfg, checkpoint_path=None, max_visualizations = 500):
    
    #valutazione finale sul test split

    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        print(f"[Test] Carico il best checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        loaded_epoch = state["epoch"]
        print(f"[Test] Checkpoint dell'epoca {state['epoch']}")
    else:
        print("[Test] Nessun checkpoint trovato, uso i pesi finali")

    model.eval()
    accumulator = BEVValidationAccumulator(cfg)
    accumulator.reset()

    sum_losses = {"loss_total": 0.0, "loss_focal": 0.0, "loss_offset": 0.0}
    num_batches = 0

    num_visualized = 0

    for batch in loader:
        inputs, targets = to_device(batch, "cuda")

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = model(*inputs)
            _, log_dict = loss_fn(predictions, targets)

        for k, v in log_dict.items():
            sum_losses[k] += v
        num_batches += 1

        accumulator.update(predictions["heatmap_logits"].float(), predictions["offset_pred"].float(), batch["sample_id"])

        if cfg.save_visualizations and num_visualized < max_visualizations:
            batch_size = predictions["heatmap_logits"].shape[0]
            to_save = min(batch_size, max_visualizations - num_visualized)
            save_bev_visualizations(
                predictions["heatmap_logits"].float(), predictions["offset_pred"].float(),
                list(batch["sample_id"]), cfg, "../visualizations_bev", epoch="test",
                max_to_save=to_save,
            )
            num_visualized += to_save


    for k in sum_losses:
        sum_losses[k] /= max(num_batches, 1)

    metrics = {f"test_{k}": v for k, v in sum_losses.items()}
    metrics.update({f"test_{k}": v for k, v in accumulator.compute().items()})

    if cfg.save_visualizations:
        save_confusion_matrix(accumulator.confusion_matrix(), cfg, "../visualizations_bev", epoch="test")

        print("\n" + "=" * 52)
    print("RISULTATI SUL TEST SPLIT (dati mai visti)")
    print("=" * 52)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("=" * 52)

    #salvataggio su file dedicati
    out_dir = Path(cfg.output_dir)

    serializable = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in metrics.items()}
    serializable["checkpoint"] = str(checkpoint_path)
    serializable["checkpoint_epoch"] = loaded_epoch
    serializable["depth_source"] = cfg.depth_source
    serializable["resolution"] = cfg.resolution
    serializable["lift_subsamples"] = cfg.lift_subsamples
    serializable["detection_threshold_val"] = cfg.detection_threshold_val
    serializable["num_test_frames"] = len(loader.dataset)

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(serializable, f, indent=2)

    with open(out_dir / "test_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in serializable.items():
            writer.writerow([k, v])

    print(f"[Test] Metriche salvate in {out_dir}/test_metrics.json e .csv")

    return metrics



#CHECKPOINT----------------------------------------------------------------------


def save_checkpoint(model, optimizer, scheduler, epoch, cfg, name):
    path = cfg.output_dir / name
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": vars(cfg),
    }, path)
    print(f"[Checkpoint] Saved: {path}")
 
 
def save_camera_branch(model, cfg, name="camera_branch.pth"):
    #deliverable per la testa di fusione (il lifting non ha pesi)
    out_path = cfg.models_dir / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": vars(cfg)}, out_path)
    print(f"[Checkpoint] Camera branch saved: {out_path}")



#MAIN--------------------------------------------------------------------------



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--backbone_checkpoint", type=str, default=None,
                        help="'none' per partire da ImageNet (ablation sul warmup)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--overfit_test", action="store_true")
    parser.add_argument("--test", action="store_true", help="Salta il training: carica best_model.pth e valuta sul test split")
    args = parser.parse_args()
 
    cfg = BEVConfig()

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=vars(cfg) if not isinstance(cfg, dict) else cfg
        )

    if args.dataset_root:
        cfg.dataset_root = Path(args.dataset_root)
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if args.backbone_checkpoint:
        cfg.backbone_checkpoint = (None if args.backbone_checkpoint.lower() == "none"
                                   else Path(args.backbone_checkpoint))
 
    if args.overfit_test:
        print("[Mode] OVERFIT TEST active")
        cfg.num_epochs = 200
        cfg.batch_size = 4
        cfg.val_every_n_epochs = 5
 
    print(f"[Config] Dataset: {cfg.dataset_root}")
    print(f"[Config] Backbone ckpt: {cfg.backbone_checkpoint}")
    print(f"[Config] Limite distanza BEV: {cfg.x_max} m")
    print(f"[Config] Epochs: {cfg.num_epochs}, batch: {cfg.batch_size}")
 
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
 
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
 
    model = CameraBEVNet(cfg, pretrained=True, backbone_checkpoint_path=cfg.backbone_checkpoint).to("cuda")
    print(f"[Model] Totale parametri: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
 
    loss_fn = WarmupLoss(
        focal_weight=cfg.focal_loss_weight,
        offset_weight=cfg.offset_loss_weight,
        focal_alpha=cfg.focal_alpha,
        focal_beta=cfg.focal_beta,
    ).to("cuda")

    if args.test:
        ckpt_path = Path(cfg.output_dir) / "best_model.pth"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint non trovato: {ckpt_path}")
        test(model, test_loader, loss_fn, cfg, checkpoint_path=ckpt_path)
        if cfg.use_wandb:
            wandb.finish()
        return
 
    optimizer = torch.optim.AdamW(model.get_param(cfg.backbone_lr, cfg.head_lr, cfg.weight_decay))
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
 
    logger = TrainingLogger(cfg.output_dir, log_every_n_steps=cfg.log_every_n_steps)
    val_accumulator = BEVValidationAccumulator(cfg)
 
    start_epoch, global_step = 0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cuda", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = start_epoch * len(train_loader)
        print(f"[Resume] Restarting from epoch {start_epoch}")
 
    best_val_ap = 0.0
    for epoch in range(start_epoch, cfg.num_epochs):
        print(f"\n============ Epoch {epoch}/{cfg.num_epochs} ============")
 
        global_step, train_losses = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn,
            cfg, epoch, global_step, logger,
        )
 
        epoch_summary = {f"train_{k}": v for k, v in train_losses.items()}
 
        if (epoch + 1) % cfg.val_every_n_epochs == 0 or epoch == cfg.num_epochs - 1:
            val_metrics = validate(model, val_loader, loss_fn, val_accumulator, cfg, epoch)
            epoch_summary.update(val_metrics)
            logger.log_epoch(epoch, epoch_summary)
            
            #log validation metrics su WandB se attivato
            if cfg.use_wandb:
                wandb.log(epoch_summary, step=global_step)
 
            if val_metrics.get("val_ap", 0.0) > best_val_ap:
                best_val_ap = val_metrics["val_ap"]
                save_checkpoint(model, optimizer, scheduler, epoch, cfg, "best_model.pth")
                save_camera_branch(model, cfg, "camera_branch.pth")
        else:
            logger.log_epoch(epoch, epoch_summary)
 
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, cfg, f"checkpoint_epoch_{epoch:03d}.pth")
 
    save_checkpoint(model, optimizer, scheduler, cfg.num_epochs - 1, cfg, "full_model_final.pth")
    save_camera_branch(model, cfg, "camera_branch_final.pth")

    test_metrics = test(model, test_loader, loss_fn, cfg, checkpoint_path=Path(cfg.output_dir) / "best_model.pth")
    logger.log_epoch(cfg.num_epochs, test_metrics)
 
    logger.close()
    print("\n[Done] Training BEV completato.")
    print(f"Best val AP: {best_val_ap:.4f}")
    print(f"Deliverable per la fusione: {cfg.models_dir}/camera_branch.pth")

    if cfg.use_wandb:
        wandb.finish()
 
 
if __name__ == "__main__":
    main()
