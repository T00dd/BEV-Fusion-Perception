
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
 
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
 
from dataset import IMAGENET_MEAN, IMAGENET_STD, gaussian_2d  #riuso del warmup
from stereo_depth import compute_depth_from_stereo


COLOR_TO_CLASS = {"blue": 0, "yellow": 1, "orange_small": 2}


#griglia bev convenzione: 
#riga 0 = x = 0 / x_min (in basso), colonna 0 = y_max (a sinistra)
#origine (0, 0) della griglia si trova in basso a sinistra

def world_to_grid(x, y, cfg):
    x_min = getattr(cfg, "x_min", 0.0)
    row = (x - x_min) / cfg.resolution
    col = (cfg.y_max - y) / cfg.resolution
    return row, col
 
 
def grid_to_world(row, col, cfg):
    # coordinate griglia (row, col) -> metri (frame veicolo)
    x_min = getattr(cfg, "x_min", 0.0)
    x = x_min + row * cfg.resolution
    y = cfg.y_max - col * cfg.resolution
    return x, y
 


def load_calib(calib_path: Path) -> Dict:
    #calib.yaml atteso:
    #  cam_left: {fx, fy, cx, cy, width, height}
    #  baseline: float (metri)
    with open(calib_path, "r") as f:
        data = yaml.safe_load(f)
    cam = data["cam_left"]
 
    #se la baseline non viene trovata non si puo' calcolare la depth stereo
    if "baseline" in data:
        baseline = float(data["baseline"])
    elif "stereo" in data and "baseline_m" in data["stereo"]:
        baseline = float(data["stereo"]["baseline_m"])
    elif "T_cam_left_to_cam_right" in data:
        T_lr = np.array(data["T_cam_left_to_cam_right"], dtype=np.float32).reshape(4, 4)
        baseline = float(abs(T_lr[0, 3]))   # traslazione lungo x della camera
    else:
        raise KeyError(
            f"Baseline stereo non trovata in {calib_path}. Attesa una di: "
            f"'baseline', 'stereo.baseline_m', 'T_cam_left_to_cam_right'."
        )
 
    return {
        "K": np.array([cam["fx"], cam["fy"], cam["cx"], cam["cy"]], dtype=np.float32),
        "calib_size": (int(cam["height"]), int(cam["width"])),
        "baseline": baseline,
        "T": np.array(data["T_cam_left_to_lidar"], dtype=np.float32).reshape(4, 4),
    }



def load_cones_3d(labels_path: Path) -> List[Dict]:
    #labels/frame_NNNNNN.json: GT 3D nel frame veicolo/LiDAR
    with open(labels_path, "r") as f:
        data = json.load(f)
    raw = data["cones"] if isinstance(data, dict) and "cones" in data else data
 
    cones = []
    for c in raw:
        if "position" in c:
            x, y = c["position"][0], c["position"][1]
        else:
            x, y = c["x"], c["y"]
        cones.append({"x": float(x), "y": float(y), "color": c.get("color", "unknown")})
    return cones


def generate_bev_heatmap_offset_mask(cones: List[Dict], cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #stessa struttura del warmup 2D, ma nel piano BEV
    #piano BEV con origine (0,0) in basso a sinistra:
    # row 0 = x_min (vicino al veicolo), row H-1 = x_max (lontano)
    # col 0 = y_max (sinistra), col W-1 = y_min (destra)
    H, W = cfg.bev_H, cfg.bev_W

    heatmap = np.zeros((cfg.num_classes, H, W), dtype=np.float32)
    offset = np.zeros((2, H, W), dtype=np.float32)
    offset_mask = np.zeros((H, W), dtype=np.float32)

    for cone in cones:
        class_idx = COLOR_TO_CLASS.get(cone["color"])
        if class_idx is None or class_idx >= cfg.num_classes:
            continue

        row_f, col_f = world_to_grid(cone["x"], cone["y"], cfg)
        row_i, col_i = int(np.floor(row_f)), int(np.floor(col_f))
        
        #scarta coni fuori dai confini metrici della griglia
        if not (0 <= row_i < H and 0 <= col_i < W):
            continue

        gauss = gaussian_2d((H, W), (col_f, row_f), cfg.gaussian_sigma)
        gauss[row_i, col_i] = 1.0
        heatmap[class_idx] = np.maximum(heatmap[class_idx], gauss)

        #offset sotto-pixel: canale 0 = d_row, canale 1 = d_col (entrambi in [0, 1))
        offset[0, row_i, col_i] = row_f - row_i
        offset[1, row_i, col_i] = col_f - col_i
        offset_mask[row_i, col_i] = 1.0

    return heatmap, offset, offset_mask


class BEVDataset(Dataset):

    def __init__(self, cfg, split_file: str, augment: bool = False, color_jitter_params: Optional[Dict] = None)
        self.cfg = cfg
        self.dataset_root = Path(cfg.dataset_root)
        self.image_size = cfg.image_size

        #stessa logica warmup 2d

        with open(self.dataset_root / split_file, "r") as f:
            scene_ids = [line.strip() for line in f if line.strip()]
 
        self.sample_ids = []
        for scene_id in scene_ids:
            img_dir = self.dataset_root / "scenes" / scene_id / "images"
            if not img_dir.is_dir():
                continue
            for img_path in sorted(img_dir.glob("*_cam_left.png")):
                self.sample_ids.append((scene_id, img_path.name.replace("_cam_left.png", "")))
 
        self._calib_cache: Dict[str, Dict] = {}
 
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.color_jitter = (transforms.ColorJitter(**color_jitter_params) if augment and color_jitter_params else None)
        self.gaussian_noise_std = cfg.gaussian_noise_std if augment else 0.0


    def __len__(self) -> int:
        return len(self.sample_ids)
 
    def _calib(self, scene_id: str) -> Dict:
        if scene_id not in self._calib_cache:
            path = self.dataset_root / "scenes" / scene_id / "calib.yaml"
            self._calib_cache[scene_id] = load_calib(path)
        return self._calib_cache[scene_id]


    def _load_depth (self, scene_dir: Path, frame_stem: str, fx_work: float, baseline: float) -> np.ndarray:

        cfg.self.cfg

        if cfg.depth_source == "precomputed":
            #depth pregenerate e salvate in file .npy
            path = scene_dir / cfg.depth_dir / f"{frame_stem}.npy"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Depth SGBM non trovata: {path}. "
                    f"Eseguire prima: python precompute_depth.py"
                )
            return np.load(path).astype(np.float32)
        
        if cfg.depth_source == "online":
            #calcolo depth al volo dalle immagini stereo
            left = np.asarray(Image.open(scene_dir / "images" / f"{frame_stem}_cam_left.png").convert("RGB"))
            right = np.asarray(Image.open(scene_dir / "images" / f"{frame_stem}_cam_right.png").convert("RGB"))
            
            scale = left.shape[1] / self.image_size[1]
            return compute_depth_from_stereo(
                left, right, fx_work * scale, baseline,
                sgbm_params=cfg.sgbm_params(),
                min_depth_m=cfg.min_depth_m, max_depth_m=cfg.max_depth_m,
            )

        if cfg.depth_source == "carla_gt":
            #depth presa da CARLA
            depth = np.load(scene_dir / cfg.depth_gt_dir / f"{frame_stem}.npy").astype(np.float32)
            depth[(depth < cfg.min_depth_m) | (depth > cfg.max_depth_m)] = 0.0
            return depth
        
        raise ValueError(f"depth_source non valido: {cfg.depth_source}")
    

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        scene_id, frame_stem = self.sample_ids[idx]
        scene_dir = self.dataset_root / "scenes" / scene_id

        #immagine
        image = Image.open(scene_dir / "images" / f"{frame_stem}_cam_left.png").convert("RGB")

        if image.size != (self.image_size[1], self.image_size[0]):
            image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        image_tensor = self.image_transform(image)
        if self.gaussian_noise_std > 0.0:
            image_tensor = image_tensor + torch.randn_like(image_tensor) * self.gaussian_noise_std
 
        #calibrazione
        calib = self._calib(scene_id)
        fx, fy, cx, cy = calib["K"]
        baseline = calib["baseline"]
        calib_h, calib_w = calib["calib_size"]
        sx, sy = self.image_size[1] / calib_w, self.image_size[0] / calib_h

        fx_work = fx * sx
        
        K = torch.tensor([fx_work, fy * sy, cx * sx, cy * sy], dtype=torch.float32)

        #pixel invalidi (occlusioni, LR check fallito) = 0.0, il lifting li scarta
        depth = self._load_depth(scene_dir, frame_stem, fx_work, baseline)
        depth_t = torch.from_numpy(depth)
        if depth.shape != tuple(self.image_size):
            #NEAREST: interpolando la depth si creano profondita' "fantasma" a metà strada sui bordi e si spalmano i buchi sui pixel validi
            depth_t = F.interpolate(depth_t[None, None], size=self.image_size, mode="nearest")[0, 0]
 
        #gt bev dai label 3d (già nel frame veicolo)
        cones = load_cones_3d(scene_dir / "labels" / f"{frame_stem}.json")
        heatmap, offset, offset_mask = generate_bev_heatmap_offset_mask(cones, self.cfg)
 
        return {
            "image": image_tensor,
            "depth": depth_t,
            "K": K,
            "T": torch.from_numpy(calib["T"]),
            "heatmap": torch.from_numpy(heatmap),
            "offset": torch.from_numpy(offset),
            "offset_mask": torch.from_numpy(offset_mask),
            "sample_id": f"{scene_id}/{frame_stem}",
        }
