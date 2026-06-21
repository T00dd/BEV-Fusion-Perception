from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class WarmupConfig:
    #path
    dataset_root: Path = Path("/path/to/fsae_dataset")  #DAMODIFICARE QUANDO ABBIAMO IL DATASET
    output_dir: Path = Path("./checkpoints/warmup")
    
    #dataset
    image_size: Tuple[int, int] = (640, 640)
    train_split_file: str = "splits/train.txt"
    val_split_file: str = "splits/val.txt"
    
    #heatmap 2d gt
    #HRNet-W32 branch ad alta risoluzione a stride 4
    heatmap_stride: int = 4
    #sigma della gaussiana per generare i picchi nella heatmap (unita' di feature map)
    gaussian_sigma: float = 2.0
    
    #modello
    backbone_name: str = "hrnet_w32.ms_in1k"

    feature_index: int = 1  #feature map a stride 4
    
    #numero di classi per la detection 2d: blu, giallo
    num_classes: int = 2
    
    #head di detection 2d
    head_hidden_channels: int = 64
    head_num_layers: int = 3
    
    #loss
    #pesi nella loss combinata
    focal_loss_weight: float = 1.0
    offset_loss_weight: float = 0.1  #L1 loss su offset sub-pixel
    
    #parametri focal loss (CenterNet-style)
    focal_alpha: float = 2.0
    focal_beta: float = 4.0
    
    #training
    num_epochs: int = 30
    batch_size: int = 16
    num_workers: int = 8
    
    #learning rate differenziato: basso sul backbone, alto sulla head
    backbone_lr: float = 1e-5
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    
    #scheduler
    warmup_epochs: int = 2  # warmup lineare del lr nelle prime epoche
    
    #augmentation

    color_jitter_brightness: float = 0.3
    color_jitter_contrast: float = 0.3
    color_jitter_saturation: float = 0.3
    color_jitter_hue: float = 0.05
    gaussian_noise_std: float = 0.01
    
    
    #Validation / Logging
    val_every_n_epochs: int = 1
    log_every_n_steps: int = 50
    save_visualizations: bool = True
    num_visualizations_per_val: int = 8

    grad_clip_norm: float = 1.0
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)