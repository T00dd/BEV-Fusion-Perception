
import argparse
import random
from pathlib import Path
from typing import Dict, List
 
import numpy as np
import torch
from PIL import Image
from torch.amp import autocast
from torch.utils.data import DataLoader
 
#training del ramo camera in bev
#carica il backbone allenato nel warmup e allena backbone (lr basso) + head BEV (lr alto). Lifting e pooling sono fissi ma differenziabili.
 
from bev_config import BEVConfig
from bev_dataset import BEVDataset, load_cones_3d, world_to_grid, COLOR_TO_CLASS
from bev_model import CameraBEVNet
from losses import WarmupLoss                                    #riciclato da  warmup
from logger import TrainingLogger                                #riciclato da  warmup
from metrics import extract_peaks_from_heatmap, match_detections_to_gt, compute_metrics  #riciclato da  warmup
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


    def update(self, heatmap_logits, offset_pred, sample_ids: List[str]):
        cfg = self.cfg
        probs = torch.sigmoid(heatmap_logits)
 
        for b in range(probs.shape[0]):
            detections = extract_peaks_from_heatmap(
                probs[b].cpu(), offset_pred[b].cpu(),
                stride=1, threshold=cfg.detection_threshold,
            )
 
            scene_id, frame_stem = sample_ids[b].split("/")
            labels_path = Path(cfg.dataset_root) / "scenes" / scene_id / "labels" / f"{frame_stem}.json"
 
            gt = []
            for c in load_cones_3d(labels_path):
                if COLOR_TO_CLASS.get(c["color"]) is None:
                    continue
                row, col = world_to_grid(c["x"], c["y"], cfg)
                #i coni fuori griglia sono  strutturalmente fuori dal dominio del modello quindi non contano come FN
                if not (0 <= row < cfg.bev_H and 0 <= col < cfg.bev_W):
                    continue
                gt.append({
                    "color": c["color"],
                    "center_px": (col, row),
                    "depth_m": float(np.hypot(c["x"], c["y"])),
                    "fully_in_image": True,
                })
 
            tp, fp, fn = match_detections_to_gt(
                detections, gt,
                match_radius_px=cfg.match_radius_m / cfg.resolution,  # metri -> celle
            )
            self.all_tp.extend(tp)
            self.all_fp.extend(fp)
            self.all_fn.extend(fn)
 
    def compute(self) -> Dict[str, float]:
        return compute_metrics(self.all_tp, self.all_fp, self.all_fn)


#VISUALIZATION----------------------------------------------------------------


def save_bev_visualizations(images, bev_gt, bev_pred_logits, sample_ids, output_dir, epoch, max_to_save=8, scale=2):
    #[camera | BEV GT | BEV pred] come nel warmup
    output_dir = Path(output_dir) / f"epoch_{epoch:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
 
    for i in range(min(max_to_save, images.shape[0])):
        img = denormalize_image(images[i])
        gt = color_heatmap(bev_gt[i].cpu().numpy())
        pred = color_heatmap(torch.sigmoid(bev_pred_logits[i]).cpu().numpy())
        if scale > 1:
            gt = np.repeat(np.repeat(gt, scale, 0), scale, 1)
            pred = np.repeat(np.repeat(pred, scale, 0), scale, 1)
 
        h = gt.shape[0]
        w = int(round(img.shape[1] * h / img.shape[0]))
        img = np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))
 
        sep = np.full((h, 4, 3), 80, dtype=np.uint8)
        panel = np.concatenate([img, sep, gt, sep, pred], axis=1)
        Image.fromarray(panel).save(output_dir / f"{sample_ids[i].replace('/', '_')}.png")


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
 
    print(f"Train: {len(train_dataset)} sample, Val: {len(val_dataset)} sample")
    print(f"Griglia BEV: {cfg.bev_H}x{cfg.bev_W} celle @ {cfg.resolution} m, "
          f"x [{cfg.x_min},{cfg.x_max}]m, y [{cfg.y_min},{cfg.y_max}]m")
 
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True,
                            persistent_workers=cfg.num_workers > 0)
    return train_loader, val_loader
 
 
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


#TRAIN AND VALIDATION---------------------------------------------------------------------


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
    return result


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
    return result




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
    args = parser.parse_args()
 
    cfg = BEVConfig()
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
 
    train_loader, val_loader = build_dataloaders(cfg)
 
    model = CameraBEVNet(cfg, pretrained=True,
                         backbone_checkpoint=cfg.backbone_checkpoint).to("cuda")
    print(f"[Model] Totale parametri: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
 
    loss_fn = WarmupLoss(
        focal_weight=cfg.focal_loss_weight,
        offset_weight=cfg.offset_loss_weight,
        focal_alpha=cfg.focal_alpha,
        focal_beta=cfg.focal_beta,
    ).to("cuda")
 
    optimizer = torch.optim.AdamW(model.get_param(cfg.backbone_lr, cfg.head_lr, cfg.weight_decay))
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
 
    logger = TrainingLogger(cfg.output_dir, log_every_n_steps=cfg.log_every_n_steps)
    val_accumulator = BEVValidationAccumulator(cfg)
 
    start_epoch, global_step = 0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cuda")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = start_epoch * len(train_loader)
        print(f"[Resume] Restarting from epoch {start_epoch}")
 
    best_val_f1 = 0.0
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
 
            if val_metrics.get("val_f1", 0.0) > best_val_f1:
                best_val_f1 = val_metrics["val_f1"]
                save_checkpoint(model, optimizer, scheduler, epoch, cfg, "best_model.pth")
                save_camera_branch(model, cfg, "camera_branch.pth")
        else:
            logger.log_epoch(epoch, epoch_summary)
 
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, cfg, f"checkpoint_epoch_{epoch:03d}.pth")
 
    save_checkpoint(model, optimizer, scheduler, cfg.num_epochs - 1, cfg, "full_model_final.pth")
    save_camera_branch(model, cfg, "camera_branch_final.pth")
 
    logger.close()
    print("\n[Done] Training BEV completato.")
    print(f"Best val F1: {best_val_f1:.4f}")
    print(f"Deliverable per la fusione: {cfg.models_dir}/camera_branch.pth")
 
 
if __name__ == "__main__":
    main()
