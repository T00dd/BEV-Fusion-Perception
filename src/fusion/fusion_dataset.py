import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from camera_detection.bev_dataset import COLOR_TO_CLASS, load_calib, load_cones_3d
from camera_detection.dataset import IMAGENET_MEAN, IMAGENET_STD

__all__ = [
    "FusionDatasetConfig",
    "LidarPointProcessor",
    "FusionDataset",
    "collate_fusion",
]


class LidarPointProcessor:

    def __init__(self, cfg_file: Path, training: bool = False):
        from pcdet.config import cfg as pcdet_cfg, cfg_from_yaml_file
        from pcdet.datasets.processor.data_processor import DataProcessor
        from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder

        cfg_from_yaml_file(str(cfg_file), pcdet_cfg)
        data_cfg = pcdet_cfg.DATA_CONFIG

        self.class_names = list(pcdet_cfg.CLASS_NAMES)
        self.point_cloud_range = np.array(data_cfg.POINT_CLOUD_RANGE, dtype=np.float32)

        self.point_feature_encoder = PointFeatureEncoder(
            data_cfg.POINT_FEATURE_ENCODING, point_cloud_range=self.point_cloud_range
        )
        self.data_processor = DataProcessor(
            data_cfg.DATA_PROCESSOR,
            point_cloud_range=self.point_cloud_range,
            training=training,
            num_point_features=self.point_feature_encoder.num_point_features,
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.depth_downsample_factor = None

    @property
    def num_point_features(self) -> int:
        return self.point_feature_encoder.num_point_features

    def __call__(self, points: np.ndarray) -> Dict:
        d = self.point_feature_encoder.forward({"points": points})
        return self.data_processor.forward(d)


@dataclass
class FusionDatasetConfig:
    dataset_root: Path
    split_file: str = "splits/train.txt"

    image_size: Tuple[int, int] = (640, 640)
    depth_source: str = "carla_gt"
    depth_dir: str = "depth_sgbm"
    depth_gt_dir: str = "depth"
    min_depth_m: float = 0.5
    max_depth_m: float = 60.0

    # must match the LiDAR branch YAML, or the two branches see different points
    point_cloud_range: Sequence[float] = (0.0, -25.0, -3.0, 50.0, 25.0, 1.0)

    colour_jitter: Optional[Dict] = None
    gaussian_noise_std: float = 0.0


class FusionDataset(Dataset):

    def __init__(
        self,
        cfg: FusionDatasetConfig,
        lidar_processor: Optional[LidarPointProcessor] = None,
    ):
        self.cfg = cfg
        self.root = Path(cfg.dataset_root)
        self.lidar_processor = lidar_processor

        with open(self.root / cfg.split_file) as f:
            scenes = [ln.strip() for ln in f if ln.strip()]

        self.samples = self._build_index(scenes)
        if not self.samples:
            raise RuntimeError(f"no complete frames found for {cfg.split_file}")

        self._calib_cache: Dict[str, Dict] = {}

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.jitter = (transforms.ColorJitter(**cfg.colour_jitter)
                       if cfg.colour_jitter else None)

    def _build_index(self, scenes: List[str]) -> List[Tuple[str, str]]:

        out = []
        for scene in scenes:
            d = self.root / "scenes" / scene
            if not d.is_dir():
                continue

            sub = (self.cfg.depth_gt_dir if self.cfg.depth_source == "carla_gt"
                   else self.cfg.depth_dir)
            stems = (
                {p.stem for p in (d / "lidar").glob("frame_*.bin")}
                & {p.name.replace("_cam_left.png", "")
                   for p in (d / "images").glob("frame_*_cam_left.png")}
                & {p.stem for p in (d / "labels").glob("frame_*.json")}
                & {p.stem for p in (d / sub).glob("frame_*.npy")}
            )
            out.extend((scene, s) for s in sorted(stems))
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def _calib(self, scene: str) -> Dict:
        if scene not in self._calib_cache:
            self._calib_cache[scene] = load_calib(
                self.root / "scenes" / scene / "calib.yaml"
            )
        return self._calib_cache[scene]

    def _load_points(self, scene_dir: Path, stem: str) -> torch.Tensor:
        pts = np.fromfile(scene_dir / "lidar" / f"{stem}.bin", dtype=np.float32)
        pts = pts.reshape(-1, 4)
        x0, y0, z0, x1, y1, z1 = self.cfg.point_cloud_range
        keep = (
            (pts[:, 0] >= x0) & (pts[:, 0] < x1)
            & (pts[:, 1] >= y0) & (pts[:, 1] < y1)
            & (pts[:, 2] >= z0) & (pts[:, 2] < z1)
        )
        return torch.from_numpy(pts[keep])

    def _load_depth(self, scene_dir: Path, stem: str) -> torch.Tensor:
        cfg = self.cfg
        sub = cfg.depth_gt_dir if cfg.depth_source == "carla_gt" else cfg.depth_dir
        path = scene_dir / sub / f"{stem}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"depth not found: {path}")

        d = np.load(path).astype(np.float32)
        d[(d < cfg.min_depth_m) | (d > cfg.max_depth_m)] = 0.0
        d = torch.from_numpy(d)

        if d.shape != tuple(cfg.image_size):
            # nearest, never bilinear: interpolating depth invents surfaces
            # halfway across occlusion boundaries and smears holes into valid pixels
            d = F.interpolate(d[None, None], size=cfg.image_size, mode="nearest")[0, 0]
        return d

    def _load_image(self, scene_dir: Path, stem: str) -> torch.Tensor:
        img = Image.open(scene_dir / "images" / f"{stem}_cam_left.png").convert("RGB")
        h, w = self.cfg.image_size
        if img.size != (w, h):
            img = img.resize((w, h), Image.BILINEAR)
        if self.jitter is not None:
            img = self.jitter(img)
        t = self.to_tensor(img)
        if self.cfg.gaussian_noise_std > 0:
            t = t + torch.randn_like(t) * self.cfg.gaussian_noise_std
        return t

    def _scaled_K(self, calib: Dict) -> torch.Tensor:
        fx, fy, cx, cy = calib["K"]
        ch, cw = calib["calib_size"]
        sx, sy = self.cfg.image_size[1] / cw, self.cfg.image_size[0] / ch
        return torch.tensor([fx * sx, fy * sy, cx * sx, cy * sy], dtype=torch.float32)

    def _load_cones(self, scene_dir: Path, stem: str, points: Optional[torch.Tensor] = None) -> torch.Tensor:

        cones = load_cones_3d(scene_dir / "labels" / f"{stem}.json")
        x0, y0, _, x1, y1, _ = self.cfg.point_cloud_range

        rows = []
        for c in cones:
            k = COLOR_TO_CLASS.get(c["color"])
            if k is None:
                continue
            if not (x0 <= c["x"] < x1 and y0 <= c["y"] < y1):
                continue

            # Conta quanti punti LiDAR cadono attorno al cono
            n_pts = 0
            if points is not None and len(points) > 0:
                dx = points[:, 0] - c["x"]
                dy = points[:, 1] - c["y"]
                dist_2d = torch.hypot(dx, dy)
                dz = points[:, 2] - c["z"]
                mask = (dist_2d <= 0.35) & (dz >= -0.5) & (dz <= 0.6)
                n_pts = int(mask.sum().item())

            rows.append([c["x"], c["y"], c["z"], float(k), float(n_pts)])

        return (torch.tensor(rows, dtype=torch.float32) if rows
                else torch.zeros(0, 5, dtype=torch.float32))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        scene, stem = self.samples[idx]
        scene_dir = self.root / "scenes" / scene
        calib = self._calib(scene)

        points = self._load_points(scene_dir, stem)
        sample = {
            "sample_id": f"{scene}/{stem}",
            "points": points,
            "image": self._load_image(scene_dir, stem),
            "depth": self._load_depth(scene_dir, stem),
            "K": self._scaled_K(calib),
            "T": torch.from_numpy(calib["T"]),
            "cones": self._load_cones(scene_dir, stem, points=points),
        }

        if self.lidar_processor is not None:
            # voxels are what MeanVFE actually reads; points are kept because the
            # LiDAR priors scatter them directly and check_alignment needs them
            processed = self.lidar_processor(points.numpy())
            for k in ("voxels", "voxel_num_points", "voxel_coords"):
                sample[k] = torch.from_numpy(processed[k])

        return sample


def collate_fusion(batch: List[Dict]) -> Dict:

    def with_batch_index(key: str) -> torch.Tensor:
        parts = [
            torch.cat([torch.full((s[key].shape[0], 1), i, dtype=torch.float32),
                       s[key]], dim=1)
            for i, s in enumerate(batch)
        ]
        return torch.cat(parts, dim=0)

    lidar = {
        "batch_size": len(batch),
        "points": with_batch_index("points"),
    }

    if "voxels" in batch[0]:
        lidar["voxels"] = torch.cat([s["voxels"] for s in batch], dim=0)
        lidar["voxel_num_points"] = torch.cat(
            [s["voxel_num_points"] for s in batch], dim=0)
        # voxel_coords are (n, 3) as [z, y, x]; the batch index goes in FRONT,
        # giving the (n, 4) layout the sparse backbone indexes with
        lidar["voxel_coords"] = torch.cat([
            F.pad(s["voxel_coords"], (1, 0), value=i)
            for i, s in enumerate(batch)
        ], dim=0)

    return {
        "lidar": lidar,
        "camera": {
            "images": torch.stack([s["image"] for s in batch]),
            "depth": torch.stack([s["depth"] for s in batch]),
            "K": torch.stack([s["K"] for s in batch]),
            "T": torch.stack([s["T"] for s in batch]),
        },
        "gt": {
            "cones": with_batch_index("cones"),
        },
        "sample_id": [s["sample_id"] for s in batch],
    }