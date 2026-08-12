from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from .grid_alignment import CAMERA_GRID, LIDAR_GRID, GridSpec

__all__ = ["FrozenEncoder", "LidarEncoder", "CameraEncoder"]


class FrozenEncoder(nn.Module):

    out_channels: int
    grid: GridSpec

    def freeze(self) -> "FrozenEncoder":
        self.requires_grad_(False)
        self.eval()
        return self

    def train(self, mode: bool = True) -> "FrozenEncoder":
        # avoid re-activating bathcnorm in a frozen encoder
        return super().train(False)

    @torch.no_grad()
    def forward(self, batch: Dict) -> torch.Tensor:
        raise NotImplementedError

    def check_output(self, feat: torch.Tensor, expected_shape: tuple) -> torch.Tensor:
        if feat.shape[1] != self.out_channels:
            raise RuntimeError(
                f"{type(self).__name__}: expected {self.out_channels} channels, "
                f"received {feat.shape[1]}"
            )
        if tuple(feat.shape[-2:]) != expected_shape:
            raise RuntimeError(
                f"{type(self).__name__}: expected spatial shape {expected_shape}, "
                f"received {tuple(feat.shape[-2:])}. Check VOXEL_SIZE / "
                f"UPSAMPLE_STRIDES in config, or resolution in GridSpec."
            )
        return feat

class LidarEncoder(FrozenEncoder):

    def __init__(
        self,
        cfg_file: Path,
        ckpt_file: Path,
        dataset,
        grid: GridSpec = LIDAR_GRID,
        logger=None,
    ):
        super().__init__()
        from pcdet.config import cfg as pcdet_cfg, cfg_from_yaml_file
        from pcdet.models import build_network
        from pcdet.utils import common_utils

        cfg_from_yaml_file(str(cfg_file), pcdet_cfg)
        logger = logger or common_utils.create_logger()

        self.net = build_network(
            model_cfg=pcdet_cfg.MODEL,
            num_class=len(pcdet_cfg.CLASS_NAMES),
            dataset=dataset,
        )
        self.net.load_params_from_file(str(ckpt_file), logger=logger, to_cpu=True)

        self.grid = grid
        self.out_channels = self._infer_out_channels(pcdet_cfg)
        self.freeze()

    @staticmethod
    def _infer_out_channels(pcdet_cfg) -> int:
        
        bb = pcdet_cfg.MODEL.BACKBONE_2D
        if "NUM_UPSAMPLE_FILTERS" in bb:
            return int(sum(bb.NUM_UPSAMPLE_FILTERS))
        return int(sum(bb.NUM_FILTERS))

    @torch.no_grad()
    def forward(self, batch_dict: Dict) -> torch.Tensor:
        
        for module in self.net.module_list:
            batch_dict = module(batch_dict)
            if "spatial_features_2d" in batch_dict:
                break
        else:
            raise RuntimeError(
                "spatial_features_2d not generated: config does not contain a 2D backbone?"
            )

        feat = batch_dict["spatial_features_2d"]
        return self.check_output(feat, self.grid.lidar_shape)


class CameraEncoder(FrozenEncoder):

    def __init__(
        self,
        cfg,
        ckpt_file: Optional[Path] = None,
        grid: GridSpec = CAMERA_GRID,
        map_location: str = "cpu",
    ):
        super().__init__()
        from camera_detection.bev_model import CameraBEVNet

        self.net = CameraBEVNet(cfg, pretrained=False)
        if ckpt_file is not None:
            self._load_checkpoint(Path(ckpt_file), map_location)

        self.cfg = cfg
        self.grid = self._grid_from_cfg(cfg, grid)
        self.out_channels = self.net.backbone.feature_info.channels()[0]
        self.freeze()

    @staticmethod
    def _grid_from_cfg(cfg, fallback: GridSpec) -> GridSpec:

        derived = GridSpec(
            x_min=cfg.x_min, x_max=cfg.x_max,
            y_min=cfg.y_min, y_max=cfg.y_max,
            resolution=cfg.resolution,
        )
        if derived.camera_shape != (cfg.bev_H, cfg.bev_W):
            raise RuntimeError(
                f"GridSpec {derived.camera_shape} incoerente con la cfg camera "
                f"({cfg.bev_H}, {cfg.bev_W})"
            )
        return derived

    def _load_checkpoint(self, path: Path, map_location: str):
        if not path.is_file():
            raise FileNotFoundError(f"camera checkpoint not found: {path}")
        state = torch.load(path, map_location=map_location)
        state = state.get("model_state_dict", state.get("state_dict", state))
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"missing keys in camera checkpoint: {missing}")
        if unexpected:
            print(f"[CameraEncoder] unexpected keys: {unexpected}")

    @torch.no_grad()
    def forward(self, batch: Dict) -> torch.Tensor:

        feature_map = self.net.backbone(batch["images"])[0]
        bev = self.net.lift_to_bev(
            feature_map, batch["depth"], batch["K"], batch["T"]
        )
        return self.check_output(bev, self.grid.camera_shape)