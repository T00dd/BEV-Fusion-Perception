from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from camera_detection.losses import CenterNetFocalLoss, OffsetL1Loss  # reused unchanged from the camera branch
from .targets import IGNORE_INDEX

__all__ = ["FusionLossConfig", "ColorCrossEntropy", "FusionLoss"]


@dataclass(frozen=True)
class FusionLossConfig:
    focal_weight: float = 1.0
    offset_weight: float = 0.1
    color_weight: float = 0.0  #stays 0 (color is ignored) in phase 0 
    focal_alpha: float = 2.0
    focal_beta: float = 4.0


class ColorCrossEntropy(nn.Module):
    
    #check only at the center of the only if the cone is present
    
    def forward(self, color_logits: torch.Tensor, color_target: torch.Tensor) -> torch.Tensor:
        valid = (color_target != IGNORE_INDEX).sum()
        if valid == 0:
            return color_logits.sum() * 0.0
        return F.cross_entropy(color_logits.float(), color_target, ignore_index=IGNORE_INDEX)


class FusionLoss(nn.Module):

    def __init__(self, cfg: FusionLossConfig = FusionLossConfig()):
        super().__init__()
        self.cfg = cfg
        self.focal_loss = CenterNetFocalLoss(alpha=cfg.focal_alpha, beta=cfg.focal_beta)
        self.offset_loss = OffsetL1Loss()
        self.color_loss = ColorCrossEntropy()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        c = self.cfg
        loss_focal = self.focal_loss(predictions["presence_logits"], targets["heatmap"])
        loss_offset = self.offset_loss(
            predictions["offset_pred"], targets["offset"], targets["offset_mask"]
        )
        loss_color = self.color_loss(predictions["color_logits"], targets["color"])

        total = (c.focal_weight * loss_focal
                 + c.offset_weight * loss_offset
                 + c.color_weight * loss_color)

        return total, {
            "loss_focal": loss_focal.item(),
            "loss_offset": loss_offset.item(),
            "loss_color": loss_color.item(),
            "loss_total": total.item(),
        }