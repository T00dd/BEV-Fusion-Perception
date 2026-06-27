import random
import sys
import argparse

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

    total_steps = cfg.num_epochs * steps_per_epoch
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

        #DEBUG
        print("max pred heatmap prob:", torch.sigmoid(predictions["heatmap_logits"]).max().item())


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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "backbone_state_dict": backbone_state,
        "backbone_name": cfg.backbone_name,
        "feature_index": cfg.feature_index,
    }, out_path)
    print(f"[Checkpoint] Backbone saved: {out_path}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default=None, help="Override del path dataset (altrimenti usa quello del config)")
    parser.add_argument("--output_dir", type=str, default=None, help="Override della directory di output")
    parser.add_argument("--resume", type=str, default=None, help="Path a checkpoint da cui riprendere")
    parser.add_argument("--overfit_test", action="store_true", help="Modalita' overfit test: usa pochi sample, tante epoche, verifica che la loss scenda a 0")
    args = parser.parse_args()

    cfg = WarmupConfig()

    if args.dataset_root:
        cfg.dataset_root = Path(args.dataset_root)
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)


    if args.overfit_test:
        print("[Mode] OVERFIT TEST active")
        #test per vedere se la rete impari
        #se la loss non scende c'è un bug nella pipeline
        cfg.num_epochs = 200
        cfg.batch_size = 4
        cfg.val_every_n_epochs = 5
        

    print(f"[Config] Dataset: {cfg.dataset_root}")
    print(f"[Config] Output: {cfg.output_dir}")
    print(f"[Config] Epochs number: {cfg.num_epochs}")
    print(f"[Config] Batch size: {cfg.batch_size}")

    set_seed(cfg.seed)

    #trova l'algoritmo + veloce per calcolare 640x640
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader = build_dataloaders(cfg)


    model = HRNet_with_detection_head(
        backbone_name=cfg.backbone_name,
        feature_index=cfg.feature_index,
        num_classes=cfg.num_classes,
        head_hidden_channels=cfg.head_hidden_channels,
        head_num_layers=cfg.head_num_layers,
        pretrained=True,
    ).to("cuda")

    #info modello
    total_params = sum(p.numel() for p in model.parameters())
    bb_params = sum(p.numel() for p in model.backbone.parameters())
    #hd_params = sum(p.numel() for p in model.head.parameters())
    print(f"[Model] Totale parametri: {total_params/1e6:.2f}M")
    print(f"[Model] -Backbone: {bb_params/1e6:.2f}M")
    #print(f"[Model] -Head: {hd_params/1e6:.2f}M")


    loss_fn = WarmupLoss(
        focal_weight=cfg.focal_loss_weight,
        offset_weight=cfg.offset_loss_weight,
        focal_alpha=cfg.focal_alpha,
        focal_beta=cfg.focal_beta,
    ).to("cuda")


    #optimizer differenziato tra backbone ed head
    param_groups = model.get_param(
        backbone_lr=cfg.backbone_lr,
        head_lr=cfg.head_lr,
        weight_decay=cfg.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)

    #scheduler
    scheduler= build_scheduler(optimizer, cfg, steps_per_epoch= len(train_loader))


    #logger
    logger = TrainingLogger(cfg.output_dir, log_every_n_steps=cfg.log_every_n_steps)

    #validation accumulator
    val_accumulator = ValidationAccumulator(
        dataset_root=cfg.dataset_root,
        stride=cfg.heatmap_stride,
        threshold=0.3,
        match_radius_px=10.0,
    )

    start_epoch = 0
    global_step = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=cfg.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = start_epoch * len(train_loader)
        print(f"[Resume] Restarting from epoch {start_epoch}")


    #training loop
    best_val_f1 = 0.0
    for epoch in range(start_epoch, cfg.num_epochs):
        print(f"\n============ Epoch {epoch}/{cfg.num_epochs} ============")

        global_step, train_losses = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            cfg=cfg,
            epoch=epoch,
            global_step_start=global_step,
            logger=logger,
        )


        #validation
        if (epoch + 1) % cfg.val_every_n_epochs == 0 or epoch == cfg.num_epochs - 1:
            val_metrics = validate(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                val_accumulator=val_accumulator,
                cfg=cfg,
                epoch=epoch,
            )
            
            #combina per logging
            epoch_summary = {f"train_{k}": v for k, v in train_losses.items()}
            epoch_summary.update(val_metrics)
            logger.log_epoch(epoch, epoch_summary)
            
            #salva il miglior modello in base a F1
            if val_metrics.get("val_f1", 0.0) > best_val_f1:
                best_val_f1 = val_metrics["val_f1"]
                save_checkpoint(model, optimizer, scheduler, epoch, cfg, "best_model.pth")
                save_backbone(model, cfg, "backbone.pth")

        else: 
            epoch_summary = {f"train_{k}": v for k, v in train_losses.items()}
            logger.log_epoch(epoch, epoch_summary)

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, cfg, f"checkpoint_epoch_{epoch:03d}.pth")


    save_checkpoint(model, optimizer, scheduler, cfg.num_epochs - 1, cfg, "full_model_final.pth")
    save_backbone(model, cfg, "backbone_final.pth")
    
    logger.close()
    print("\n[Done] Warm-up completato.")
    print(f"Best val F1: {best_val_f1:.4f}")
    print(f"Deliverable per script BEV: {cfg.output_dir}/backbone_only_best.pth")
 
 
if __name__ == "__main__":
    main()



