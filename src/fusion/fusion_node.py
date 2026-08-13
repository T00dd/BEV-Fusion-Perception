from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

__all__ = ["FusionConfig", "FusionNode"]

# group order in the concatenation. Every weight-norm diagnostic depends on this
LIDAR_GROUP = slice(None, None)

@dataclass(frozen=True)
class FusionConfig:
    lidar_channels: int = 256
    camera_channels: int = 128
    lidar_priors: int = 4
    camera_priors: int = 4
    hidden_channels: int = 128
    gate_channels: int = 32
    gate_bias: float = 2.0
    context_kernel: int = 3
    modality_dropout: float = 0.0


def _conv_bn_relu(cin: int, cout: int, k: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class FusionNode(nn.Module):
    def __init__(self, cfg: FusionConfig = FusionConfig()):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_channels

        self.proj_lidar = _conv_bn_relu(cfg.lidar_channels + cfg.lidar_priors, h)
        self.proj_camera = _conv_bn_relu(cfg.camera_channels + cfg.camera_priors, h)

        gate_in = 2 * h + cfg.lidar_priors + cfg.camera_priors
        self.gate = nn.Sequential(
            nn.Conv2d(gate_in, cfg.gate_channels, 1, bias=False),
            nn.BatchNorm2d(cfg.gate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.gate_channels, 1, 3, padding=1),
        )

        self.context = _conv_bn_relu(2 * h, h, k=cfg.context_kernel)

        # zero convolution step
        self.zero_conv = nn.Conv2d(h, cfg.lidar_channels, 1, bias=True)

        self.lidar_slice = slice(0, h)
        self.camera_slice = slice(h, 2 * h)
        self.reset_fusion_parameters()


    def reset_fusion_parameters(self):

        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, self.cfg.gate_bias)

    def assert_zero_init(self):

        assert torch.all(self.zero_conv.weight == 0), "zero conv weight was overwritten"
        assert torch.all(self.zero_conv.bias == 0), "zero conv bias was overwritten"

    def no_decay_parameters(self):
        # parameters that should not be wheight-decayed
        return list(self.zero_conv.parameters()) + list(self.gate[-1].parameters())

    def forward(
        self,
        f_lidar: torch.Tensor,
        f_camera: torch.Tensor,
        prior_lidar: torch.Tensor,
        prior_camera: torch.Tensor,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:

        self._check_shapes(f_lidar, f_camera, prior_lidar, prior_camera)

        if self.training and self.cfg.modality_dropout > 0:
            keep = (torch.rand(f_camera.shape[0], 1, 1, 1, device=f_camera.device)
                    >= self.cfg.modality_dropout).float()
            f_camera = f_camera * keep
            prior_camera = prior_camera * keep

        # step 2
        p_lidar = self.proj_lidar(torch.cat([f_lidar, prior_lidar], dim=1))
        p_camera = self.proj_camera(torch.cat([f_camera, prior_camera], dim=1))

        # step 3
        g = torch.sigmoid(self.gate(
            torch.cat([p_lidar, p_camera, prior_lidar, prior_camera], dim=1)
        ))

        # step 4
        fused = torch.cat([g * p_lidar, (1.0 - g) * p_camera], dim=1)
        context = self.context(fused)

        # step 5
        delta = self.zero_conv(context)
        out = f_lidar + delta

        if not return_aux:
            return out, None
        return out, {
            "gate": g,
            "delta": delta,
            "delta_norm": delta.norm(),
            "camera_dead_channels": (p_camera.abs().amax(dim=(0, 2, 3)) == 0).sum(),
        }

    def _check_shapes(self, f_l, f_c, p_l, p_c):
        c = self.cfg
        for name, t, expected in (
            ("f_lidar", f_l, c.lidar_channels),
            ("f_camera", f_c, c.camera_channels),
            ("prior_lidar", p_l, c.lidar_priors),
            ("prior_camera", p_c, c.camera_priors),
        ):
            if t.shape[1] != expected:
                raise ValueError(f"{name}: expected {expected} channels, got {t.shape[1]}")
        ref = f_l.shape[-2:]
        for name, t in (("f_camera", f_c), ("prior_lidar", p_l), ("prior_camera", p_c)):
            if t.shape[-2:] != ref:
                raise ValueError(
                    f"{name} spatial shape {tuple(t.shape[-2:])} != f_lidar {tuple(ref)}; "
                    "did you forget grid.align_*_features?"
                )

    @torch.no_grad()
    def modality_weight_norms(self) -> Dict[str, float]:
        
        w = self.context[0].weight
        return {
            "w_lidar": w[:, self.lidar_slice].norm().item(),
            "w_camera": w[:, self.camera_slice].norm().item(),
        }