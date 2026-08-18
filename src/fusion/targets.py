from dataclasses import dataclass
from typing import Dict

import torch

from .grid_alignment import FUSION_GRID, GridSpec

__all__ = ["TargetConfig", "build_targets"]

IGNORE_INDEX = -1


@dataclass(frozen=True)
class TargetConfig:
    #sigma in cells
    #2.0 cells at 0.25m becomes 2.5 cells at 0.2 m for the same physical extent
    sigma: float = 2.5
    num_colors: int = 3
    min_points: int = 0


@torch.no_grad()
def build_targets(
    cones: torch.Tensor,
    batch_size: int,
    grid: GridSpec = FUSION_GRID,
    cfg: TargetConfig = TargetConfig(),
) -> Dict[str, torch.Tensor]:
    
    #cones: (M, 5) as [batch_idx, x, y, z, class] from collate_fusion

    if cones.dim() != 2 or cones.shape[1] < 5:
        raise ValueError(f"expected (M, 5) [b,x,y,z,cls] cones, got {tuple(cones.shape)}")

    device = cones.device
    H, W = grid.camera_shape

    heatmap = torch.zeros(batch_size, 1, H, W, device=device)
    offset = torch.zeros(batch_size, 2, H, W, device=device)
    offset_mask = torch.zeros(batch_size, 1, H, W, device=device)
    color = torch.full((batch_size, H, W), IGNORE_INDEX, dtype=torch.long, device=device)

    if cones.numel() == 0:
        return {"heatmap": heatmap, "offset": offset,
                "offset_mask": offset_mask, "color": color}

    b = cones[:, 0].long()
    row_f, col_f = grid.world_to_grid(cones[:, 1], cones[:, 2])
    row_i, col_i = row_f.floor().long(), col_f.floor().long()

    keep = (row_i >= 0) & (row_i < H) & (col_i >= 0) & (col_i < W)
    if cfg.min_points > 0 and cones.shape[1] >= 6:
        keep = keep & (cones[:, 5] >= cfg.min_points)
    b, row_f, col_f = b[keep], row_f[keep], col_f[keep]
    row_i, col_i = row_i[keep], col_i[keep]
    cls = cones[keep, 4].long()

    if b.numel() == 0:
        return {"heatmap": heatmap, "offset": offset,
                "offset_mask": offset_mask, "color": color}

    #gaussian rendered on a local window instead of the whole grid

    r = int(round(3.0 * cfg.sigma))
    d = torch.arange(-r, r + 1, device=device)
    win_rows = row_i.view(-1, 1) + d.view(1, -1)
    win_cols = col_i.view(-1, 1) + d.view(1, -1)

    dr = (win_rows.float() - row_f.view(-1, 1)) ** 2
    dc = (win_cols.float() - col_f.view(-1, 1)) ** 2
    gauss = torch.exp(-(dr.unsqueeze(2) + dc.unsqueeze(1)) / (2.0 * cfg.sigma ** 2))

    inside = ((win_rows >= 0) & (win_rows < H)).unsqueeze(2) \
        & ((win_cols >= 0) & (win_cols < W)).unsqueeze(1)

    lin = (b.view(-1, 1, 1) * H + win_rows.clamp(0, H - 1).unsqueeze(2)) * W \
        + win_cols.clamp(0, W - 1).unsqueeze(1)

    flat = heatmap.view(-1)
    flat.index_reduce_(0, lin[inside].reshape(-1), gauss[inside].reshape(-1),
                       "amax", include_self=True)

    #the centre cell is the positive we force it to 1 so the focal pos_mask always fires

    centre = (b * H + row_i) * W + col_i
    flat.index_fill_(0, centre, 1.0)

    offset.view(batch_size, 2, -1)[b, 0, row_i * W + col_i] = row_f - row_i.float()
    offset.view(batch_size, 2, -1)[b, 1, row_i * W + col_i] = col_f - col_i.float()
    offset_mask.view(-1).index_fill_(0, centre, 1.0)
    color.view(-1)[centre] = cls

    return {"heatmap": heatmap, "offset": offset,
            "offset_mask": offset_mask, "color": color}