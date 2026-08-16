from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = ["FusionConfig"]


@dataclass
class FusionConfig:

    # "phase0": geometry only, fusion frozen at delta = 0
    # "phase1": fusion released, colour enabled
    # "phase2": encoders unfrozen
    phase: str = "phase0"
    resume_from: Optional[Path] = None
    allow_encoder_finetune: bool = False

    #paths
    dataset_root: Path = Path("../../dataset")
    output_dir: Path = Path("./checkpoints/fusion")
    lidar_cfg_file: Path = Path("../lidar_detection/cfgs/cone_centerpoint.yaml")
    lidar_checkpoint: Path = Path("../models/lidar_encoder.pth")
    camera_checkpoint: Path = Path("../models/camera_bev.pth")
    train_split_file: str = "splits/train.txt"
    val_split_file: str = "splits/val.txt"
    test_split_file: str = "splits/test.txt"

    #training
    num_epochs: int = 25
    batch_size: int = 8
    num_workers: int = 12
    weight_decay: float = 1e-4
    grad_clip: float = 10.0

    #target: sigma in celle. 2.0 a 0.25 m diventa 2.5 a 0.2 m
    gaussian_sigma: float = 2.5

    #loss
    focal_weight: float = 1.0
    offset_weight: float = 0.1
    focal_alpha: float = 2.0
    focal_beta: float = 4.0

    #validation
    detection_threshold: float = 0.3
    match_radius_m: float = 0.5
    log_every_n_steps: int = 50
    seed: int = 0