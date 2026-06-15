import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def gaussian_2d(shape: tuple[int, int], center: tuple[float, float], sigma: float) -> np.ndarray:
    
    #genera una gaussiana 2d con centro su center, con deviazione sigma e dimensione shape

    H, W = shape
    cx, cy = center
    y = np.arange(H, dtype=np.float32)[:, None]
    x = np.arange(W, dtype=np.float32)[None, :]

    gauss = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    return gauss


def generate_heatmap_offset_mask(
    cones: List[Dict],
    image_size: Tuple[int, int],
    stride: int,
    num_classes: int,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    #genera heatmap gaussiana, offset e offset mask per il calcolo della loss

    #heatmap: gaaussiana in corrispondenza dei centri dei coni
    #offset: vettore di offset per recuperare precisione sul centro esatto del cono
    #offset_mask: maschera per calcolare la loss solo in corrispondenza dei centri dei coni

    H, W = image_size
    H_feat, W_feat = H // stride, W // stride

    heatmap = np.zeros((num_classes, H_feat, W_feat), dtype=np.float32)
    offset = np.zeros((2, H_feat, W_feat), dtype=np.float32)
    offset_mask = np.zeros((H_feat, W_feat), dtype=np.float32)

    color_to_class = {
        "red": 0,
        "blue": 1,
    }

    for cone in cones:
        if not cone.get("fully_in_image", True):
            continue

        color = cone["color"]
        if color not in color_to_class:
            continue

        class_idx = color_to_class[color]

        #centro del cono in coordinate immagine quindi pixel
        cx_img, cy_img = cone["center_px"]

        #coordinate del centro del cono in coordinate feature map quindi divido per stride
        cx_feat = cx_img / stride
        cy_feat = cy_img / stride

        #pixel intero più vicino al centro del cono in coordinate feature map
        cx_feat_int = int(round(cx_feat))
        cy_feat_int = int(round(cy_feat))

        if not (0 <= cx_feat_int < W_feat and 0 <= cy_feat_int < H_feat):
            continue

        #genera gaussiana 2d centrata sul cono
        gauss = gaussian_2d((H_feat, W_feat), (cx_feat, cy_feat), sigma)

        heatmap[class_idx] = np.maximum(heatmap[class_idx], gauss)

        #crea offset e offset mask
        offset[0, cy_feat_int, cx_feat_int] = cx_feat - cx_feat_int
        offset[1, cy_feat_int, cx_feat_int] = cy_feat - cy_feat_int
        offset_mask[cy_feat_int, cx_feat_int] = 1.0


class WarmupDataset(Dataset):

    def __init__(
        self,
        dataset_root: Path,
        split_file: str,
        image_size: Tuple[int, int] = (640, 640),
        heatmap_stride: int = 4,
        num_classes: int = 2,
        gaussian_sigma: float = 2.0,
        augment: bool = False,
        precomputed_heatmaps: bool = False,

        #parametri per augmentation (usati solo se augment=True)
        color_jitter_params: Optional[Dict] = None,
        gaussian_noise_std: float = 0.0,
    ):
        
        self.dataset_root = Path(dataset_root)
        self.image_size = image_size
        self.heatmap_stride = heatmap_stride
        self.num_classes = num_classes
        self.gaussian_sigma = gaussian_sigma
        self.augment = augment
        self.precomputed_heatmaps = precomputed_heatmaps


        #cerca lista dei sample dallo split file
        split_path = self.dataset_root / split_file
        with open(split_path, "r") as f: self.sample_ids = [line.strip() for line in f if line.strip()]

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.color_jitter_params = None
        if augment and color_jitter_params is not None:
            self.color_jitter_params = transforms.ColorJitter(**color_jitter_params)

        self.gaussian_noise_std = gaussian_noise_std if augment else 0.0

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _sample_path(self, sample_id: str) -> Path:
        #ritorna il path della cartella del sample dato l'id del sample
        return self.dataset_root / "sequences" / sample_id

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_id = self.sample_ids[idx]
        sample_dir = self._sample_path(sample_id)

        #carica immagine
        img_path = sample_dir / "camera_left.png"
        image = Image.open(img_path).convert("RGB") 


        #verifica dimensione immagine, se non 640x640 ridimensionamento
        if image.size != (self.image_size[1], self.image_size[0]):
            image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)

        #color jitter augmentation
        if self.color_jitter_params is not None:
            image = self.color_jitter_params(image)

        
        #pil to tensor e normalizzazione
        image_tensor = self.image_transform(image)

        #rumore gaussiano
        if self.gaussian_noise_std > 0.0:
            noise = torch.randn_like(image_tensor) * self.gaussian_noise_std
            image_tensor = image_tensor + noise

        
        #caricamento heatmap gt

        cones_path = sample_dir / "cones_camera_2d.json"
        with open(cones_path, "r") as f:
            cones_data = json.load(f)
        heatmap, offset, offset_mask = generate_heatmap_offset_mask(
            cones_data["cones_in_image"],
            self.image_size,
            self.heatmap_stride,
            self.num_classes,
            self.gaussian_sigma,
        )

        return {
            "image": image_tensor,
            "heatmap": torch.from_numpy(heatmap),
            "offset": torch.from_numpy(offset),
            "offset_mask": torch.from_numpy(offset_mask),
            "sample_id": sample_id,
        }
    


