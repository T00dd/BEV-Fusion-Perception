from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class BEVConfig:

    #paths
    dataset_root: Path = Path("../../carla_dataset_three_classses")
    output_dir: Path = Path("./checkpoints/bev")
    models_dir: Path = Path("../models")
    depth_dir: Path = Path("../../data/depth_sgbm")
    
    #backbone allenato nel warmup
    backbone_checkpoint: Path = Path("../models/backbone.pth")

    #depth
    #"precomputed": legge depth_sgbm/*.npy generati offline 
    #"online": calcola SGBM dentro __getitem__ (comodo per debug)
    #"carla_gt": usa la depth GT di CARLA 
    depth_source: str = "carla_gt"  # "precomputed", "online", "carla_gt"
    depth_dir: str = "depth_sgbm"      # sottocartella scena per "precomputed"
    depth_gt_dir: str = "depth"        # depth CARLA, riferimento
    min_depth_m: float = 0.3
    max_depth_m: float = 60.0

    #parametri SGBM (usati con depth_source="online" e da precompute_depth.py)
    sgbm_min_disp: int = 0
    sgbm_num_disp: int = 192           #deve coprire la disparità a distanza minima
    sgbm_block_size: int = 5
    sgbm_uniqueness_ratio: int = 10
    sgbm_speckle_window_size: int = 100
    sgbm_speckle_range: int = 2
    sgbm_disp12_max_diff: int = 1
    sgbm_lr_consistency: bool = True   #left-right consistency check
    sgbm_lr_max_diff: float = 1.0      #px di tolleranza tra le due disparità

    #dataset
    image_size: Tuple[int, int] = (640, 640)
    train_split_file: str = "splits/train.txt"
    val_split_file: str = "splits/val.txt"
    test_split_file: str = "splits/test.txt"

    #griglia BEV
    x_min: float = 0.0
    x_max: float = 50.0
    y_min: float = -25.0
    y_max: float = 25.0
    resolution: float = 0.20 

    #sigma FISSO in celle: la griglia BEV e' metrica, non prospettica,
    #quindi l'adattivita' che serviva nel warmup 2D qui non ha motivo
    gaussian_sigma: float = 2.5

    #modello (si prende che il backbone sia stato allenato nel warmup)
    backbone_name: str = "hrnet_w32.ms_in1k"
    feature_index: int = 1
    feature_stride: int = 4
    num_classes: int = 3
    head_hidden_channels: int = 128
    head_num_layers: int = 5

    #loss
    focal_loss_weight: float = 1.0
    offset_loss_weight: float = 0.1
    focal_alpha: float = 2.0
    focal_beta: float = 4.0

    #training
    num_epochs: int = 15
    batch_size: int = 16
    num_workers: int = 12
    backbone_lr: float = 2e-5
    head_lr: float = 6e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2

    lift_subsamples: int = 3

    #augmentation: (non geometrica in quanto romperebbe la corrispondenza pixel <-> depth <-> calibrazione)
    color_jitter_brightness: float = 0.3
    color_jitter_contrast: float = 0.3
    color_jitter_saturation: float = 0.3
    color_jitter_hue: float = 0.05
    gaussian_noise_std: float = 0.01

    #validation / logging
    val_every_n_epochs: int = 1
    log_every_n_steps: int = 50
    save_visualizations: bool = True
    num_visualizations_per_val: int = 8
    detection_threshold: float = 0.2
    detection_threshold_val: float = 0.05
    match_radius_m: float = 0.5

    grad_clip_norm: float = 1.0
    seed: int = 14

    #wandb for monitoring
    use_wandb: bool = False
    wandb_project: str = "HRNet-bev-Cone-Detection"
    wandb_run_name: str = "bev-run-02"

    @property
    def bev_H(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def bev_W(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution))

    def sgbm_params(self) -> dict:
        return {
            "min_disp": self.sgbm_min_disp,
            "num_disp": self.sgbm_num_disp,
            "block_size": self.sgbm_block_size,
            "uniqueness_ratio": self.sgbm_uniqueness_ratio,
            "speckle_window_size": self.sgbm_speckle_window_size,
            "speckle_range": self.sgbm_speckle_range,
            "disp12_max_diff": self.sgbm_disp12_max_diff,
            "lr_consistency": self.sgbm_lr_consistency,
            "lr_max_diff": self.sgbm_lr_max_diff,
        }

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)