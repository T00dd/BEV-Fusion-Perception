from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .grid_alignment import CAMERA_GRID, FUSION_GRID, GridSpec, resample_camera_convention

__all__ = ["LidarPriorConfig", "CameraPriorConfig", "lidar_priors", "CameraPriors"]

N_LIDAR_PRIORS = 4
N_CAMERA_PRIORS = 4

# lidar priors are: log1p(count)/4, max height (from z_min to z_max mapped to [0,1]), height spread/4, mean intensity
# camera priors are: frustum mask, range/x_max, expected stereo depth error (soft-saturated to [0,1))

@dataclass(frozen=True)
class LidarPriorConfig:
    z_min: float = -3.0
    z_max: float = 1.0
    count_scale: float = 4.0
    spread_scale: float = 4.0
    intensity_scale: float = 1.0


@dataclass(frozen=True)
class CameraPriorConfig:
    fx: float
    baseline: float
    image_width: int
    image_height: int
    disparity_error_px: float = 0.5
    depth_error_scale: float = 1.0
    count_scale: float = 4.0

# lidar priors

@torch.no_grad()
def lidar_priors(points: torch.Tensor, batch_size: int, grid: GridSpec = FUSION_GRID, cfg: LidarPriorConfig = LidarPriorConfig()) -> torch.Tensor:
    
    if points.shape[1] < 5:
        raise ValueError(f"expected at least 5 columns [b,x,y,z,i], got {points.shape[1]}")

    b = points[:, 0].long()
    x, y, z, intensity = points[:, 1], points[:, 2], points[:, 3], points[:, 4]

    row, col = grid.world_to_grid(x, y)
    row, col = row.floor().long(), col.floor().long()
    keep = (row >= 0) & (row < grid.n_x) & (col >= 0) & (col < grid.n_y)

    b, z, intensity = b[keep], z[keep], intensity[keep]
    lin = (b * grid.n_x + row[keep]) * grid.n_y + col[keep]
    n_cells = batch_size * grid.n_x * grid.n_y

    count = torch.zeros(n_cells, device=points.device, dtype=torch.float32)
    count.index_add_(0, lin, torch.ones_like(z))

    # index_reduce_ needs an identity element that loses to every real value
    z_max = torch.full((n_cells,), cfg.z_min, device=points.device, dtype=torch.float32)
    z_min = torch.full((n_cells,), cfg.z_max, device=points.device, dtype=torch.float32)
    z_max.index_reduce_(0, lin, z, "amax", include_self=True)
    z_min.index_reduce_(0, lin, z, "amin", include_self=True)

    intensity_sum = torch.zeros(n_cells, device=points.device, dtype=torch.float32)
    intensity_sum.index_add_(0, lin, intensity)

    occupied = count > 0
    safe_count = count.clamp(min=1.0)

    out = torch.stack([
        torch.log1p(count) / cfg.count_scale,
        torch.where(occupied, (z_max - cfg.z_min) / (cfg.z_max - cfg.z_min), 
                    torch.zeros_like(z_max)),
        torch.where(occupied, (z_max - z_min) / cfg.spread_scale, torch.zeros_like(z_max)),
        (intensity_sum / safe_count) / cfg.intensity_scale,
    ], dim=0)

    return out.view(N_LIDAR_PRIORS, batch_size, grid.n_x, grid.n_y) \
              .permute(1, 0, 2, 3).contiguous().clamp(0.0, 1.0)


# camera priors

class CameraPriors(nn.Module):

    def __init__(self, cfg: CameraPriorConfig, grid: GridSpec = FUSION_GRID, source_grid: GridSpec = CAMERA_GRID):
        super().__init__()
        self.cfg = cfg
        self.grid = grid
        self.source_grid = source_grid

        gx, gy = grid.cell_centers_camera_convention()
        r = torch.sqrt(gx ** 2 + gy ** 2)

        depth_err = (r ** 2) * cfg.disparity_error_px / (cfg.fx * cfg.baseline)
        depth_err_norm = depth_err / (depth_err + cfg.depth_error_scale)

        self.register_buffer("cell_x", gx, persistent=False)
        self.register_buffer("cell_y", gy, persistent=False)
        self.register_buffer("range_norm", (r / grid.x_max).clamp(0.0, 1.0), persistent=False)
        self.register_buffer("depth_err", depth_err, persistent=False)
        self.register_buffer("depth_err_norm", depth_err_norm, persistent=False)
        self._frustum_cache: Optional[torch.Tensor] = None
        self._cached_T: Optional[torch.Tensor] = None

    @torch.no_grad()
    def frustum_mask(self, K: torch.Tensor, T: torch.Tensor) -> torch.Tensor:

        B = T.shape[0]
        if self._cached_T is not None and self._cached_T.shape == T.shape \
                and torch.equal(self._cached_T, T):
            return self._frustum_cache

        gx, gy = self.cell_x, self.cell_y
        pts_v = torch.stack([gx.reshape(-1), gy.reshape(-1),
                             torch.zeros_like(gx.reshape(-1))], dim=0)     # (3, n_cells)
        pts_v = pts_v.unsqueeze(0).expand(B, 3, -1)

        R_inv = T[:, :3, :3].transpose(1, 2)
        pts_c = torch.bmm(R_inv, pts_v - T[:, :3, 3:4])

        fx, fy, cx, cy = (K[:, i].view(B, 1) for i in range(4))
        z = pts_c[:, 2]
        in_front = z > 1e-3
        u = fx * pts_c[:, 0] / z.clamp(min=1e-3) + cx
        v = fy * pts_c[:, 1] / z.clamp(min=1e-3) + cy

        inside = (in_front
                  & (u >= 0) & (u < self.cfg.image_width)
                  & (v >= 0) & (v < self.cfg.image_height))

        mask = inside.float().view(B, self.grid.n_x, self.grid.n_y)
        self._frustum_cache, self._cached_T = mask, T.clone()
        return mask

    @torch.no_grad()
    def occupancy(self, counts: torch.Tensor) -> torch.Tensor:
        
        if counts.dim() != 4 or counts.shape[1] != 1:
            raise ValueError(f"expected (B, 1, H, W) counts, got {tuple(counts.shape)}")
 
        if tuple(counts.shape[-2:]) != self.grid.camera_shape:
            counts = resample_camera_convention(
                counts.float(), self.source_grid, self.grid, mode="nearest"
            )
        return (torch.log1p(counts.float()) / self.cfg.count_scale).clamp(0.0, 1.0)

    @torch.no_grad()
    def forward(self, K: torch.Tensor, T: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:

        B = T.shape[0]
        mask = self.frustum_mask(K, T)
        rng = self.range_norm.unsqueeze(0).expand(B, -1, -1)
        err = self.depth_err_norm.unsqueeze(0).expand(B, -1, -1)
        return torch.stack([mask, rng, err], dim=1).contiguous()