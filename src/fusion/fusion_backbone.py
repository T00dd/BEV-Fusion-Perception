from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .encoders import FrozenEncoder
from .grid_alignment import (
    CAMERA_GRID,
    FUSION_GRID,
    GridSpec,
    align_camera_features,
    align_lidar_features,
)
from .fusion_node import FusionConfig, FusionNode
from .priors import CameraPriorConfig, CameraPriors, LidarPriorConfig, lidar_priors

__all__ = ["FusionBackboneConfig", "FusionBackbone"]


@dataclass
class FusionBackboneConfig:
    camera_prior: CameraPriorConfig
    lidar_prior: LidarPriorConfig = field(default_factory=LidarPriorConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    grid: GridSpec = FUSION_GRID


class FusionBackbone(nn.Module):
    # encoders injection
    def __init__(
        self,
        lidar_encoder: FrozenEncoder,
        camera_encoder: FrozenEncoder,
        cfg: FusionBackboneConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.grid = cfg.grid
        self.lidar_encoder = lidar_encoder
        self.camera_encoder = camera_encoder

        self._check_encoder_grids()

        self.camera_priors = CameraPriors(
            cfg.camera_prior, grid=cfg.grid, source_grid=camera_encoder.grid
        )

        fusion_cfg = FusionConfig(
            **{**cfg.fusion.__dict__,
               "lidar_channels": lidar_encoder.out_channels,
               "camera_channels": camera_encoder.out_channels}
        )
        self.fusion = FusionNode(fusion_cfg)
        self.out_channels = lidar_encoder.out_channels

    def _check_encoder_grids(self):

        for name, enc in (("lidar", self.lidar_encoder), ("camera", self.camera_encoder)):
            if not enc.grid.same_extent_as(self.grid):
                raise ValueError(
                    f"{name} encoder grid extent does not match the fusion grid"
                )


    def forward(
        self, batch: Dict, return_aux: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        lidar_batch, camera_batch = batch["lidar"], batch["camera"]
        batch_size = lidar_batch["batch_size"]

        f_lidar = align_lidar_features(
            self.lidar_encoder(lidar_batch), self.lidar_encoder.grid, self.grid
        )
        f_camera_raw, counts = self.camera_encoder(camera_batch)
        f_camera = align_camera_features(
            f_camera_raw, self.camera_encoder.grid, self.grid
        )

        prior_lidar = lidar_priors(
            lidar_batch["points"], batch_size, self.grid, self.cfg.lidar_prior
        )
        prior_camera = self.camera_priors(
            camera_batch["K"], camera_batch["T"], counts
        )

        return self.fusion(f_lidar, f_camera, prior_lidar, prior_camera, return_aux)


    def assert_zero_init(self):
        # check that the fusion node is zero-initialised, so that the identity guarantee holds at step 0
        self.fusion.assert_zero_init()

    def param_groups(self, lr: float, weight_decay: float = 1e-4):
        # optimiser parameter groups: the fusion node has a no-decay group for the gate and context convs
        no_decay = {id(p) for p in self.fusion.no_decay_parameters()}
        decayed, undecayed = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            if id(p) in no_decay:
                undecayed.append(p)
            else:
                decayed.append(p)
        return [
            {"params": decayed, "lr": lr, "weight_decay": weight_decay},
            {"params": undecayed, "lr": lr, "weight_decay": 0.0},
        ]

    @torch.no_grad()
    def diagnostics(self, aux: Dict[str, torch.Tensor]) -> Dict[str, float]:
        # diagnostics for logging and plotting: gate mean/std, delta norm, dead camera channels, modality weight norms
        g = aux["gate"]
        out = {
            "gate_mean": g.mean().item(),
            "gate_std": g.std().item(),
            "delta_norm": aux["delta_norm"].item(),
            "camera_dead_channels": float(aux["camera_dead_channels"]),
        }
        out.update(self.fusion.modality_weight_norms())
        return out

    @torch.no_grad()
    def gate_by_range(self, aux: Dict[str, torch.Tensor], n_bins: int = 10):
        
        r = self.camera_priors.range_norm * self.grid.x_max
        g = aux["gate"][:, 0].mean(dim=0)
        edges = torch.linspace(0.0, self.grid.x_max, n_bins + 1, device=g.device)
        r = r.to(g.device)

        centres, means = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (r >= lo) & (r < hi)
            if m.any():
                centres.append(((lo + hi) / 2).item())
                means.append(g[m].mean().item())
        return centres, means