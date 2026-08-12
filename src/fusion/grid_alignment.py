from dataclasses import dataclass
import torch
import torch.nn.functional as F

__all__ = [
    "GridSpec",
    "LIDAR_GRID",
    "CAMERA_GRID",
    "FUSION_GRID",
    "lidar_raster_to_camera_convention",
    "camera_convention_to_lidar_raster",
    "resample_camera_convention",
    "align_lidar_features",
    "align_camera_features",
    "occupancy_from_points",
]

@dataclass(frozen=True)
class GridSpec:
    # grid extension and resolution in BEV
    # n_x/n_y are the number of cells along the x y axis, respectively
    # independently from the convention used.
    # the shapes are: (n_x, n_y) for camera convention and (n_y, n_x) for lidar raster convention.

    x_min: float = 0.0
    x_max: float = 50.0
    y_min: float = -25.0
    y_max: float = 25.0
    resolution: float = 0.2

    @property
    def n_x(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution))
 
    @property
    def n_y(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution))
 
    @property
    def camera_shape(self) -> tuple:
        return (self.n_x, self.n_y)
 
    @property
    def lidar_shape(self) -> tuple:
        return (self.n_y, self.n_x)
 
    def same_extent_as(self, other: "GridSpec", tol: float = 1e-6) -> bool:
        return (
            abs(self.x_min - other.x_min) < tol
            and abs(self.x_max - other.x_max) < tol
            and abs(self.y_min - other.y_min) < tol
            and abs(self.y_max - other.y_max) < tol
        )

    # meters to index conversion (camera convention)
    def world_to_grid(self, x, y):
        row = (x - self.x_min) / self.resolution
        col = (self.y_max - y) / self.resolution
        return row, col
 
    def grid_to_world(self, row, col):
        x = self.x_min + row * self.resolution
        y = self.y_max - col * self.resolution
        return x, y

    # meters to index conversion (lidar raster convention, just for diagnostic purposes)
    def world_to_lidar_raster(self, x, y):
        i_y = (y - self.y_min) / self.resolution
        i_x = (x - self.x_min) / self.resolution
        return i_y, i_x

    def cell_centers_camera_convention(self, device=None, dtype=torch.float32):
        # metric coordinates from each cell center in camera convention (x, y) with shape (n_x, n_y)
        rows = torch.arange(self.n_x, device=device, dtype=dtype) + 0.5
        cols = torch.arange(self.n_y, device=device, dtype=dtype) + 0.5
        x = self.x_min + rows * self.resolution
        y = self.y_max - cols * self.resolution
        return x.view(-1, 1).expand(self.n_x, self.n_y), y.view(1, -1).expand(self.n_x, self.n_y)

LIDAR_GRID = GridSpec(resolution=0.2)
CAMERA_GRID = GridSpec(resolution=0.25)
FUSION_GRID = LIDAR_GRID

# index permutation
def lidar_raster_to_camera_convention(t: torch.Tensor) -> torch.Tensor:
    
    if t.dim() < 2:
        raise ValueError(f"expected a tensor with at least 2 dimensions, received {t.dim()}")
    return t.transpose(-2, -1).flip(-1).contiguous()
 
 
def camera_convention_to_lidar_raster(t: torch.Tensor) -> torch.Tensor:
    
    if t.dim() < 2:
        raise ValueError(f"expected a tensor with at least 2 dimensions, received {t.dim()}")
    return t.flip(-1).transpose(-2, -1).contiguous()

# resampling 
def resample_camera_convention(t: torch.Tensor, src: GridSpec, dst: GridSpec, mode: str = "bilinear") -> torch.Tensor:
    # change of resolution
    if t.dim() != 4:
        raise ValueError(f"expected shape (B, C, H, W), received {tuple(t.shape)}")
    if not src.same_extent_as(dst):
        raise ValueError(
            "the two grids do not cover the same extent: "
            f"src=({src.x_min},{src.x_max},{src.y_min},{src.y_max}) "
            f"dst=({dst.x_min},{dst.x_max},{dst.y_min},{dst.y_max})"
        )
    if tuple(t.shape[-2:]) != src.camera_shape:
        raise ValueError(
            f"spatial shape {tuple(t.shape[-2:])} is inconsistent with src {src.camera_shape}"
        )

    if src.camera_shape == dst.camera_shape:
        return t
 
    kwargs = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
    return F.interpolate(t, size=dst.camera_shape, mode=mode, **kwargs)

def align_lidar_features(feat: torch.Tensor, src: GridSpec = LIDAR_GRID, dst: GridSpec = FUSION_GRID) -> torch.Tensor:
    
    if tuple(feat.shape[-2:]) != src.lidar_shape:
        raise ValueError(
            f"shape spaziale {tuple(feat.shape[-2:])} incoerente con il raster "
            f"LiDAR atteso {src.lidar_shape}"
        )
    out = lidar_raster_to_camera_convention(feat)
    return resample_camera_convention(out, src, dst)

def align_camera_features(feat: torch.Tensor, src: GridSpec = CAMERA_GRID, dst: GridSpec = FUSION_GRID) -> torch.Tensor:
    
    return resample_camera_convention(feat, src, dst)

# diagnostic function to compute occupancy from point cloud
def occupancy_from_points(points_xy: torch.Tensor, grid: GridSpec = FUSION_GRID) -> torch.Tensor:
    
    row, col = grid.world_to_grid(points_xy[:, 0], points_xy[:, 1])
    row = row.floor().long()
    col = col.floor().long()
    keep = (row >= 0) & (row < grid.n_x) & (col >= 0) & (col < grid.n_y)
    row, col = row[keep], col[keep]
 
    occ = torch.zeros(grid.n_x * grid.n_y, dtype=torch.float32, device=points_xy.device)
    occ.index_add_(0, row * grid.n_y + col, torch.ones_like(row, dtype=torch.float32))
    return occ.view(grid.n_x, grid.n_y)