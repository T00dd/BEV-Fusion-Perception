
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
 
from .dataset import IMAGENET_MEAN, IMAGENET_STD, gaussian_2d  #riuso del warmup
from .stereo_depth import compute_depth_from_stereo


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
    with open(calib_path, "r") as f:
        data = yaml.safe_load(f)
    
    cams = data["cameras"]
    cam_left = cams["left"]
    K_mat = cam_left["intrinsic_K"]
    fx, fy = float(K_mat[0][0]), float(K_mat[1][1])
    cx, cy = float(K_mat[0][2]), float(K_mat[1][2])

    #camera -> ego
    T_cam = np.array(cam_left["extrinsic_cam_from_ego_carla"], dtype=np.float64)
    T_lid = np.array(data["extrinsic_lidar_from_ego_carla"], dtype=np.float64)
    R_cam, t_cam, t_lid = T_cam[:3, :3], T_cam[:3, 3], T_lid[:3, 3]

    #ottico -> corpo carla -> ego carla -> lidar -> frame label
    #M: inversa di (x,y,z)->(y,-z,x)
    M = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]], dtype=np.float64)
    #N: carla y-destra -> frame label y-sinistra
    N = np.diag([1.0, -1.0, 1.0])

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = N @ R_cam @ M
    T[:3, 3] = N @ (t_cam - t_lid)      #origine sul LIDAR non su ego

    t_right = np.array(cams["right"]["extrinsic_cam_from_ego_carla"], dtype=np.float64)[:3, 3]
    baseline = float(abs(t_right[1] - t_cam[1]))

    return {
        "K": np.array([fx, fy, cx, cy], dtype=np.float32),
        "calib_size": (int(cam_left["height"]), int(cam_left["width"])),
        "baseline": baseline,
        "T": T.astype(np.float32),
        "T_inv": np.linalg.inv(T).astype(np.float32),   # label -> ottico
    }


def cone_visible(cone, calib, cone_height: float = 0.32) -> bool:

    T_inv = calib["T_inv"]
    fx, fy, cx, cy = calib["K"]
    H, W = calib["calib_size"]

    for dz in (0.0, cone_height):
        p = T_inv[:3, :3] @ np.array([cone["x"], cone["y"], cone["z"] + dz]) + T_inv[:3, 3]
        if p[2] <= 1e-6:            #dietro la camera
            continue
        u = fx * p[0] / p[2] + cx
        v = fy * p[1] / p[2] + cy
        if 0 <= u < W and 0 <= v < H:
            return True
    return False


def load_cones_3d(labels_path: Path) -> List[Dict]:
    #labels/frame_NNNNNN.json: GT 3D nel frame veicolo/LiDAR
    with open(labels_path, "r") as f:
        data = json.load(f)
    raw = data["cones"] if isinstance(data, dict) and "cones" in data else data
 
    cones = []
    for c in raw:
        if "position" in c:
            x, y, z = c["position"][0], c["position"][1], c["position"][2]
        else:
            x, y, z = c["x"], c["y"], c.get("z", 0.0) # Fallback a 0.0 se z non esiste
        
        cones.append({
            "x": float(x), 
            "y": float(y), 
            "z": float(z), 
            "color": c.get("class", c.get("color", "unknown"))
        })
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

    def __init__(self, cfg, split_file: str, augment: bool = True, color_jitter_params: Optional[Dict] = None):
        self.cfg = cfg
        self.dataset_root = Path(cfg.dataset_root)
        self.image_size = cfg.image_size
        self.augment = augment

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

        cfg = self.cfg

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
        
        #filtro FOV: scarta i coni che la telecamera non può vedere
        visible_cones = [c for c in cones if cone_visible(c, calib)]

        #bev augmentation
        T = calib["T"].copy()
        if self.augment:
            yaw = np.random.uniform(-8, 8) * np.pi / 180
            dx, dy = np.random.uniform(-1.0, 1.0, 2)
            c, s = np.cos(yaw), np.sin(yaw)
            A = np.array([[c, -s, 0, dx],
                          [s,  c, 0, dy],
                          [0,  0, 1, 0.0],
                          [0,  0, 0, 1.0]], dtype=np.float32)
            T = A @ T
            for cone in visible_cones:
                cone["x"], cone["y"] = (c*cone["x"] - s*cone["y"] + dx,
                                        s*cone["x"] + c*cone["y"] + dy)
        
        #genera le mappe per la loss solo sui coni visibili
        heatmap, offset, offset_mask = generate_bev_heatmap_offset_mask(visible_cones, self.cfg)
 
        return {
            "image": image_tensor,
            "depth": depth_t,
            "K": K,
            "T": torch.from_numpy(T),
            "heatmap": torch.from_numpy(heatmap),
            "offset": torch.from_numpy(offset),
            "offset_mask": torch.from_numpy(offset_mask),
            "sample_id": f"{scene_id}/{frame_stem}",
        }
