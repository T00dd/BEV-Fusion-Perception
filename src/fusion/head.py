from dataclasses import dataclass
from typing import Dict, Iterator

import torch
import torch.nn as nn

__all__ = ["FusionHeadConfig", "FusionHead"]


@dataclass(frozen=True)
class FusionHeadConfig:
    in_channels: int = 256
    hidden_channels: int = 64
    num_colors: int = 3
    trunk_layers: int = 2
    color_stem_layers: int = 1
    prior_prob: float = 0.01


def _conv_bn_relu(cin: int, cout: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


def _stem(cin: int, cout: int, n_layers: int) -> nn.Sequential:
    layers, c = [], cin
    for _ in range(n_layers):
        layers.append(_conv_bn_relu(c, cout))
        c = cout
    return nn.Sequential(*layers)


class FusionHead(nn.Module):
    #convolutional stem + parallel 1x1 branches
    #presence is class-agnostic 
    # colour gets its own stem reading the full fused map

    def __init__(self, cfg: FusionHeadConfig = FusionHeadConfig()):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_channels

        self.trunk = _stem(cfg.in_channels, h, cfg.trunk_layers)
        self.color_stem = _stem(cfg.in_channels, h, cfg.color_stem_layers)

        self.presence_head = nn.Conv2d(h, 1, 1)
        self.offset_head = nn.Conv2d(h, 2, 1)
        self.color_head = nn.Conv2d(h, cfg.num_colors, 1)

        self.reset_head_parameters()

    def reset_head_parameters(self):
        p = self.cfg.prior_prob
        focal_bias = -float(torch.log(torch.tensor((1.0 - p) / p)))

        nn.init.normal_(self.presence_head.weight, std=0.01)
        nn.init.constant_(self.presence_head.bias, focal_bias)

        # offset targets live in [0, 1), so 0.5 is the zero-information prediction
        nn.init.normal_(self.offset_head.weight, std=0.01)
        nn.init.constant_(self.offset_head.bias, 0.5)

        # zero bias on colour = uniform prior over the classes
        nn.init.normal_(self.color_head.weight, std=0.01)
        nn.init.zeros_(self.color_head.bias)


    def geometry_parameters(self) -> Iterator[nn.Parameter]:
        for m in (self.trunk, self.presence_head, self.offset_head):
            yield from m.parameters()



    def color_parameters(self) -> Iterator[nn.Parameter]:
        for m in (self.color_stem, self.color_head):
            yield from m.parameters()

    def freeze_color(self, frozen: bool = True) -> "FusionHead":
        for p in self.color_parameters():
            p.requires_grad_(not frozen)
        return self

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if features.shape[1] != self.cfg.in_channels:
            raise ValueError(
                f"expected {self.cfg.in_channels} channels, got {features.shape[1]}"
            )
        t = self.trunk(features)
        return {
            "presence_logits": self.presence_head(t),
            "offset_pred": self.offset_head(t),
            "color_logits": self.color_head(self.color_stem(features)),
        }