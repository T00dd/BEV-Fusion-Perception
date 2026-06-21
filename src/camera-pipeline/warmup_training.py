import random
import sys

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

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




