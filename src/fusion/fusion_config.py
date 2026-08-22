import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = ["FusionTrainConfig", "check_paths", "repo_cwd"]
 

SRC = Path(__file__).resolve().parents[1]
REPO = SRC.parent



@dataclass
class FusionTrainConfig:

    # "phase0": geometry only, fusion frozen at delta = 0
    # "phase1": fusion released, colour enabled
    # "phase2": encoders unfrozen
    phase: str = "phase0"
    resume_from: Optional[Path] = None
    allow_encoder_finetune: bool = False
    test_only: bool = False

    #paths
    dataset_root: Path = REPO / "carla_dataset_three_classses"
    output_dir: Path = SRC / "checkpoints/fusion"
    lidar_cfg_file: Path = SRC / "lidar_detection/configs/noise/second_centerpoint_agnostic_noise.yaml"
    lidar_checkpoint: Path = REPO / "lib/OpenPCDet/output/lidar_detection/configs/noise/second_centerpoint_agnostic_noise/default/ckpt/checkpoint_epoch_80.pth"
    camera_checkpoint: Path = REPO / "models_bev/camera_branch.pth"
    camera_cfg: Optional[object] = None
    train_split_file: str = "splits/train.txt"
    val_split_file: str = "splits/val.txt"
    test_split_file: str = "splits/test.txt"


    #training
    num_epochs: int = 15
    batch_size: int = 8
    num_workers: int = 12
    weight_decay: float = 1e-4
    lr_min: float = 1e-5          # eta_min del cosine scheduler
    grad_clip: float = 10.0

    #camera: servono a CameraPriorConfig e a FusionDatasetConfig
    #"carla_gt" legge da depth/, qualsiasi altro valore da depth_sgbm/
    depth_source: str = "sgbm"
    fx: float = 381.36
    baseline: float = 1.0
    image_size: tuple = (640, 640)   #(H, W)

    #target: sigma in celle. 2.0 a 0.25 m diventa 2.5 a 0.2 m
    gaussian_sigma: float = 0.85   # = (2*MIN_RADIUS+1)/6 come il baseline lidar

    #loss
    focal_weight: float = 1.0
    offset_weight: float = 0.1
    focal_alpha: float = 2.0
    focal_beta: float = 4.0
 
    #visualization: NB la cartella e' src/fusion/visualization/, per questo il
    #modulo si chiama bev_visualization.py e non visualization.py (altrimenti
    #Python troverebbe la cartella al posto del modulo)
    save_visualizations: bool = True
    visualization_dir: Path = SRC / "fusion/visualization"
    num_visualizations_per_val: int = 8
    num_visualizations_test: int = 500
    color_conf_threshold: float = 0.45
 
    #validation
    detection_threshold: float = 0.3
    match_radius_m: float = 0.5
    log_every_n_steps: int = 50
    seed: int = 0