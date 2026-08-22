import argparse
import csv
import time
from pathlib import Path
from typing import Dict

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from camera_detection.logger import TrainingLogger  #riusato invariato

from .bev_metrics import (BEVValidationAccumulator, compare_to_baseline, save_baseline)
from .fusion_config import FusionTrainConfig
from .fusion_dataset import FusionDataset, collate_fusion
from .head import FusionHead, FusionHeadConfig
from .head_loss import FusionLoss, FusionLossConfig
from .targets import TargetConfig, build_targets
from .fusion_visualization import save_bev_visualizations, save_confusion_matrix


def setup_phase(backbone, head, phase: str) -> Dict:
    #phase0: la fusione è congelta non solo inizializzata a zero
    train_fusion = phase in ("phase1", "phase2")
    train_color = phase in ("phase1", "phase2")
    train_encoders = phase == "phase2"

    for enc in (backbone.lidar_encoder, backbone.camera_encoder):
        enc.requires_grad_(train_encoders)
    backbone.fusion.requires_grad_(train_fusion)
    head.freeze_color(not train_color)

    lr_head = {"phase0": 1e-3, "phase1": 2e-4, "phase2": 5e-5}[phase]
    lr_fusion = {"phase0": 0.0, "phase1": 1e-3, "phase2": 1e-4}[phase]
    groups = [{"params": [p for p in head.parameters() if p.requires_grad],"lr": lr_head}]


    if train_fusion:
        #niente weight decay su gate e context: lo tirerebbe verso g = 0.5 disfacendo il bias +2 che fa partire il sistema fidandosi del LiDAR
        no_decay = {id(p) for p in backbone.fusion.no_decay_parameters()}
        fusion = list(backbone.fusion.parameters())
        groups.append({"params": [p for p in fusion if id(p) not in no_decay], "lr": lr_fusion})
        groups.append({"params": [p for p in fusion if id(p) in no_decay], "lr": lr_fusion, "weight_decay": 0.0})

    if train_encoders:
        groups.append({"params": list(backbone.lidar_encoder.parameters()) + list(backbone.camera_encoder.parameters()), "lr": 1e-6})

    return {"groups": groups, "lr_head": lr_head, "lr_fusion": lr_fusion, "color_weight": 0.0 if phase == "phase0" else 0.5}


@torch.no_grad()
def gate_diagnostics(backbone, aux) -> Dict[str, float]:
    out = backbone.diagnostics(aux)
    #gate_std sull'intero tensore mescola varianza spaziale e varianza fra frame: un gate degenerato in "scalare per frame" la supererebbe pur
    #essendo il fallimento da intercettare
    #questa è la deviazione spaziale dentro ciascun frame, poi mediata
    g = aux["gate"]
    out["gate_std_spatial"] = g.flatten(2).std(dim=-1).mean().item()
    return out


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) if torch.is_tensor(vv) else vv
                      for kk, vv in v.items()}
        else:
            out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def train_one_epoch(backbone, head, loader, loss_fn, optimizer, cfg, device, tcfg, logger, phase, epoch, step):
    head.train()
    backbone.fusion.train()

    for batch in loader:
        batch = to_device(batch, device)
        targets = build_targets(batch["gt"]["cones"], batch["lidar"]["batch_size"], backbone.grid, tcfg)

        with autocast(device_type=device.type, dtype=torch.bfloat16):
            features, aux = backbone(batch, return_aux=True)
            loss, log = loss_fn(head(features), targets)

        #le diagnostiche si riducono a scalari prima del backward e aux viene rilasciato: aux["delta"] è (B, 256, 250, 250) cioè 256 MB a B=8 in bf16
        log.update(gate_diagnostics(backbone, aux))
        aux = None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g["params"]], cfg.grad_clip)
        optimizer.step()

        log["grad_norm"] = float(grad_norm)
        logger.log_step(epoch, step, log,lr_backbone=phase["lr_fusion"], lr_head=phase["lr_head"])
        step += 1

    return step


@torch.no_grad()
def validate(backbone, head, loader, accumulator, device, n_gate_bins=10, cfg=None, viz_dir=None, tag="", max_viz=0):
    head.eval()
    backbone.eval()
    accumulator.reset()

    gate_curves, diag_sums, n, saved = [], {}, 0, 0
    for batch in loader:
        batch = to_device(batch, device)
        with autocast(device_type=device.type, dtype=torch.bfloat16):
            features, aux = backbone(batch, return_aux=True)
            preds = head(features)

        accumulator.update(preds, batch["gt"]["cones"])

        if viz_dir is not None and saved < max_viz:
            saved += save_bev_visualizations(
                preds, batch["gt"]["cones"], batch["sample_id"], viz_dir,
                grid=backbone.grid,
                K=batch["camera"]["K"], T=batch["camera"]["T"],
                threshold=cfg.detection_threshold,
                color_conf_threshold=cfg.color_conf_threshold,
                max_to_save=max_viz - saved, tag=tag)
        for k, v in gate_diagnostics(backbone, aux).items():
            diag_sums[k] = diag_sums.get(k, 0.0) + v
        gate_curves.append(backbone.gate_by_range(aux, n_bins=n_gate_bins))
        aux = None
        n += 1

    out = accumulator.compute()
    out.update({f"val_{k}": v / max(n, 1) for k, v in diag_sums.items()})

    #curva del gate contro la distanza: e' la lettura diretta di dove il sistema
    #si fida della camera invece che del LiDAR. Va nel csv come colonne separate
    #cosi' si plotta dopo senza rilanciare nulla
    curves = torch.stack([torch.as_tensor(c[1]) for c in gate_curves]).mean(0)
    centers = torch.as_tensor(gate_curves[0][0])
    for c, g in zip(centers.tolist(), curves.tolist()):
        out[f"gate_r{c:.0f}m"] = float(g)

    if viz_dir is not None and saved:
        save_confusion_matrix(accumulator.confusion_matrix(), viz_dir, tag)

    return out


def make_dataset(cfg, split_file: str, lidar_processor):
    from .fusion_dataset import FusionDatasetConfig
    return FusionDataset(
        FusionDatasetConfig(dataset_root=cfg.dataset_root, split_file=split_file,
                            image_size=cfg.image_size),
        lidar_processor,
    )


def run_test_split(backbone, head, cfg, device, out_dir, val_metrics, lidar_processor, checkpoint=None):
    #il test si fa sul checkpoint MIGLIORE, non sui pesi dell'ultima epoca
    best = Path(checkpoint) if checkpoint else out_dir / f"{cfg.phase}_best.pth"
    if not best.exists():
        print(f"{best} non trovato, test saltato")
        return None
    split = Path(cfg.dataset_root) / cfg.test_split_file
    if not split.exists():
        print(f"{split} non trovato, test saltato")
        return None

    ckpt = torch.load(best, map_location=device, weights_only=False)
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])

    loader = DataLoader(
        make_dataset(cfg, cfg.test_split_file, lidar_processor),
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        collate_fn=collate_fusion, shuffle=False, pin_memory=True)
    acc = BEVValidationAccumulator(backbone.grid, cfg.detection_threshold,
                                   cfg.match_radius_m)

    viz_dir = Path(cfg.visualization_dir) / "test" if cfg.save_visualizations else None
    m = validate(backbone, head, loader, acc, device, cfg=cfg, viz_dir=viz_dir, tag="test", max_viz=cfg.num_visualizations_test)
    m["from_epoch"] = ckpt.get("epoch", -1)

    #una riga per split così il confronto val/test si legge da un solo file
    path = out_dir / "test_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["split"] + list(m.keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        if val_metrics:
            w.writerow({"split": "val", **val_metrics})
        w.writerow({"split": "test", **m})

    print(f"\n==== TEST (checkpoint epoca {m['from_epoch']}) ====")
    for k in ("precision", "recall", "f1", "loc_p50_cm", "loc_p90_cm","loc_over_20cm", "color_acc"):
        v = val_metrics.get(k, float("nan")) if val_metrics else float("nan")
        print(f"  {k:<16} val {v:.4f}   test {m.get(k, float('nan')):.4f}")
    print(f"  metriche complete in {path}")
    if viz_dir is not None:
        print(f"  visualizzazioni in {viz_dir}\n")
    return m


def main(cfg: FusionTrainConfig):
    from camera_detection.bev_config import BEVConfig

    from .encoders import CameraEncoder, LidarEncoder
    from .fusion_backbone import FusionBackbone, FusionBackboneConfig
    from .fusion_dataset import FusionDatasetConfig, LidarPointProcessor
    from .priors import CameraPriorConfig

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.phase == "phase2" and not cfg.allow_encoder_finetune:
        raise RuntimeError(
            "phase2 sblocca gli encoder: F_L diventa mobile e la garanzia "
            "F_out = F_L + delta non vale piu'. Serve --allow-encoder-finetune."
        )
    if cfg.phase != "phase0" and cfg.resume_from is None and not cfg.test_only:
        raise RuntimeError(f"{cfg.phase} deve partire da un checkpoint (--resume-from)")

    #LidarPointProcessor espone class_names, grid_size, voxel_size,
    #point_cloud_range e num_point_features: e' esattamente l'interfaccia che
    #build_network di OpenPCDet si aspetta come "dataset"
    lidar_processor = LidarPointProcessor(cfg.lidar_cfg_file, training=False)
    camera_cfg = cfg.camera_cfg or BEVConfig()

    backbone = FusionBackbone(
        LidarEncoder(cfg.lidar_cfg_file, cfg.lidar_checkpoint, lidar_processor),
        CameraEncoder(camera_cfg, cfg.camera_checkpoint),
        FusionBackboneConfig(camera_prior=CameraPriorConfig(fx=cfg.fx, baseline=cfg.baseline, image_width=cfg.image_size[1], image_height=cfg.image_size[0])),
    ).to(device)
    head = FusionHead(FusionHeadConfig(in_channels=backbone.out_channels)).to(device)


    if cfg.resume_from is not None:
        ckpt = torch.load(cfg.resume_from, map_location=device, weights_only=False)
        backbone.load_state_dict(ckpt["backbone"])
        head.load_state_dict(ckpt["head"])
        print(f"ripreso da {cfg.resume_from} (phase {ckpt['phase']})")

    phase = setup_phase(backbone, head, cfg.phase)
    loss_fn = FusionLoss(FusionLossConfig(
        focal_weight=cfg.focal_weight, offset_weight=cfg.offset_weight,
        color_weight=phase["color_weight"],
        focal_alpha=cfg.focal_alpha, focal_beta=cfg.focal_beta,
    ))
    optimizer = torch.optim.AdamW(phase["groups"], weight_decay=cfg.weight_decay)
    min_pts_target = 2 if cfg.phase == "phase0" else 0
    tcfg = TargetConfig(sigma=cfg.gaussian_sigma, min_points=min_pts_target)


    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, collate_fn=collate_fusion, pin_memory=True)
    train_loader = DataLoader(make_dataset(cfg, cfg.train_split_file, lidar_processor), shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(make_dataset(cfg, cfg.val_split_file, lidar_processor), shuffle=False, **common)


    accumulator = BEVValidationAccumulator(
        backbone.grid, cfg.detection_threshold, cfg.match_radius_m, min_lidar_points=2 if cfg.phase == "phase0" else 0)
    out_dir = Path(cfg.output_dir) / cfg.phase


    #--test: solo inferenza del best model sull'intero test split
    if cfg.test_only:
        run_test_split(backbone, head, cfg, device, out_dir, {}, lidar_processor, checkpoint=cfg.resume_from)
        return
    

    logger = TrainingLogger(out_dir, log_every_n_steps=cfg.log_every_n_steps)
    print(f"phase {cfg.phase}: color_weight {phase['color_weight']}, "
          f"log in {out_dir}")

    #la baseline B vive fuori dalla cartella della fase, cosi' phase1 la trova
    baseline_path = Path(cfg.output_dir) / "baseline_B.json"
    if cfg.phase != "phase0" and not baseline_path.exists():
        print(f"ATTENZIONE: {baseline_path} assente, nessun confronto con B")


    best_f1, step, last_val = -1.0, 0, {}
    for epoch in range(cfg.num_epochs):
        t0 = time.time()
        step = train_one_epoch(backbone, head, train_loader, loss_fn, optimizer, cfg, device, tcfg, logger, phase, epoch, step)
        viz_dir = None
        if cfg.save_visualizations:
            viz_dir = Path(cfg.visualization_dir) / "validation" / f"epoch_{epoch:03d}"
        m = validate(backbone, head, val_loader, accumulator, device, cfg=cfg, viz_dir=viz_dir, tag=f"epoch {epoch:03d}", max_viz=cfg.num_visualizations_per_val)
        m["epoch_time_s"] = time.time() - t0

        #differenze contro la baseline B (checkpoint di fase 0): una regressione si vede subito invece che a fine training
        
        m.update(compare_to_baseline(m, accumulator.instance_hit, baseline_path))
        logger.log_epoch(epoch, m)
        last_val = m

        #in phase0 delta deve essere rimasto esattamente 0
        if cfg.phase == "phase0":
            backbone.assert_zero_init()

        state = {"backbone": backbone.state_dict(), "head": head.state_dict(),
                 "epoch": epoch, "phase": cfg.phase, "metrics": m}
        torch.save(state, out_dir / f"{cfg.phase}_last.pth")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save(state, out_dir / f"{cfg.phase}_best.pth")
            print(f"  nuovo best F1 {best_f1:.4f}")


    logger.close()
    

    #la fase 0 con delta = 0 è la configurazione B: le sue metriche sono il riferimento fisso di ogni claim sulla fusione
    if cfg.phase == "phase0":
        save_baseline(last_val, accumulator.instance_hit, baseline_path)
        print(f"baseline B salvata in {baseline_path}")
    print(f"fine training. best F1 in validation {best_f1:.4f}")
    run_test_split(backbone, head, cfg, device, out_dir, last_val, lidar_processor)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="phase0", choices=["phase0", "phase1", "phase2"])
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument("--allow-encoder-finetune", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--test", action="store_true", help="salta il training: solo inferenza del best model sul test split")
    p.add_argument("--no-visualizations", action="store_true")
    a = p.parse_args()

    cfg = FusionTrainConfig(phase=a.phase, resume_from=a.resume_from, allow_encoder_finetune=a.allow_encoder_finetune)
    cfg.test_only = a.test
    if a.no_visualizations:
        cfg.save_visualizations = False
    main(cfg)