import random
import sys

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

#script principale per il warmup 2d di HRNet-W32 sul task di detection dei coni

from warmup_config import WarmupConfig
from dataset import WarmupDataset
from model import HRNet_with_detection_head
from losses import WarmupLoss
from metrics import ValidationAccumulator
from logger import TrainingLogger
from visualization import save_visualization_batch


def set_seed(seed: int):

    #fissa i seed per riproducibilità
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg: WarmupConfig):
    """Crea i DataLoader di train e val."""
    color_jitter_params = {
        "brightness": cfg.color_jitter_brightness,
        "contrast": cfg.color_jitter_contrast,
        "saturation": cfg.color_jitter_saturation,
        "hue": cfg.color_jitter_hue,
    }
    
    train_dataset = WarmupDataset(
        dataset_root=cfg.dataset_root,
        split_file=cfg.train_split_file,
        image_size=cfg.image_size,
        heatmap_stride=cfg.heatmap_stride,
        num_classes=cfg.num_classes,
        gaussian_sigma=cfg.gaussian_sigma,
        augment=True,
        color_jitter_params=color_jitter_params,
        gaussian_noise_std=cfg.gaussian_noise_std,
    )
    val_dataset = WarmupDataset(
        dataset_root=cfg.dataset_root,
        split_file=cfg.val_split_file,
        image_size=cfg.image_size,
        heatmap_stride=cfg.heatmap_stride,
        num_classes=cfg.num_classes,
        gaussian_sigma=cfg.gaussian_sigma,
        augment=False,
    )
    
    print(f"Train: {len(train_dataset)} sample, Val: {len(val_dataset)} sample")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    
    return train_loader, val_loader


def build_scheduler(optimizer , cfg: WarmupConfig, steps_per_epoch: int ):

    #crea lo scheduler: warmup lineare nelle prime warmup_epochs, successivamente cosine annealing fino alla fine

    total_steps = cfg.num_epoch * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    def lr(step):
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr)


def train_one_epoch(
        model,
        loader,
        optimizer,
        scheduler,
        loss_fn,
        cfg: WarmupConfig,
        epoch: int,
        global_step_start: int,
        logger: TrainingLogger,
):
    
    #esegue un epoca di training e ritorna il global_step finale

    model.train()
    device = "cuda"

    global_step = global_step_start

    epoch_losses = {"loss_total":0.0, "loss_focal": 0.0, "loss_offset": 0.0}
    num_batches = 0


    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = {
            "heatmap": batch["heatmap"].to(device, non_blocking=True),
            "offset": batch["offset"].to(device, non_blocking=True),
            "offset_mask": batch["offset_mask"].to(device, non_blocking=True),
        }

        optimizer.zero_grad(set_to_none=True)

        #rete con precisione bf16
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = model(images)
            loss, log_dict = loss_fn(predictions, targets)

        #calcolo gradienti
        loss.backward()
        #taglio del gradiente per evitare la sua esplosione
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        #aggiornamento pesi
        optimizer.step()


        scheduler.step()

        for k, v in log_dict.items():
            epoch_losses[k] += v
        
        num_batches += 1

        #estraiamo il valore del LR dal modello
        lrs = [pg["lr"] for pg in optimizer.param_groups]
        lr_backbone = lrs[0]
        lr_head = lrs[1] if len(lrs) > 1 else lrs[0]

        #chiamiamo il logger
        logger.log_step(epoch, global_step, log_dict, lr_backbone, lr_head)

        global_step += 1

    
    for k in epoch_losses:
        epoch_losses[k] /= max(num_batches, 1)
    
    return global_step, epoch_losses


@torch.no_grad()
def validate(
    model,
    loader,
    loss_fn,
    val_accumulator: ValidationAccumulator,
    cfg: WarmupConfig,
    epoch: int,
):
    
    #eseguiamo validation: loss media + metriche

    model.eval()
    device = "cuda"

    val_accumulator.reset()

    sum_losses = {"loss_total": 0.0, "loss_focal": 0.0, "loss_offset": 0.0}
    num_batches =0 

    vis_batch_idx = 0

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        targets = {
            "heatmap": batch["heatmap"].to(device, non_blocking=True),
            "offset": batch["offset"].to(device, non_blocking=True),
            "offset_mask": batch["offset_mask"].to(device, non_blocking=True),
        }

        sample_ids =  batch["sample_id"]

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = model(images)
            loss, log_dict = loss_fn(predictions, targets)


        for k, v in log_dict.items():
            sum_losses[k] += v
        num_batches += 1

        val_accumulator.update(
            predictions["heatmap_logits"].float(),
            predictions["offset_pred"].float(),
            sample_ids,
        )

        #salva visualizzazioni del primo batch
        if cfg.save_visualizations and batch_idx == vis_batch_idx:
            save_visualization_batch(
                images=images,
                heatmaps_gt=targets["heatmap"],
                heatmaps_pred_logits=predictions["heatmap_logits"].float(),
                sample_ids=list(sample_ids),
                output_dir="../visualizations",
                stride=cfg.heatmap_stride,
                epoch=epoch,
                max_to_save=cfg.num_visualizations_per_val,
            )

    for k in sum_losses:
        sum_losses[k] /= max(num_batches, 1)
    
    metrics = val_accumulator.compute()

    result = {f"val_{k}": v for k, v in sum_losses.items()}
    result.update({f"val_{k}": v for k, v in metrics.items()})

    return result


def save_checkpoint(model, optimizer, scheduler, epoch: int, cfg: WarmupConfig, name:str):
    
    checkpoint_path = cfg.output_dir / name
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": vars(cfg) if not isinstance(cfg, dict) else cfg,
    }, checkpoint_path)
    print(f"[Checkpoint] Saved: {checkpoint_path}")


def save_backbone(model, cfg:WarmupConfig, name: str = "backbone.pth"):
    
    #salva i pesi solo del backbone

    backbone_state = {f"backbone.{k}": v for k, v in model.backbone.state_dict().items()}
    out_path = cfg.models_dir / name
    torch.save({
        "backbone_state_dict": backbone_state,
        "backbone_name": cfg.backbone_name,
        "feature_index": cfg.feature_index,
    }, out_path)
    print(f"[Checkpoint] Backbone saved: {out_path}")
