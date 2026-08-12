import torch

from fusion.grid_alignment import FUSION_GRID
from fusion.priors import (
    CameraPriorConfig,
    CameraPriors,
    LidarPriorConfig,
    lidar_priors,
)

CAM = CameraPriorConfig(fx=700.0, baseline=0.12, image_width=1280, image_height=720)

def test_lidar_priors_shape_and_bounds():
    pts = torch.tensor([
        [0.0, 40.1, 20.1, 0.3, 0.5],
        [0.0, 40.1, 20.1, 0.1, 0.7],
        [1.0, 10.1, -5.1, -0.2, 0.2],
    ])
    out = lidar_priors(pts, batch_size=2)
    assert out.shape == (2, 4, *FUSION_GRID.camera_shape)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_lidar_priors_land_in_the_expected_cell():
    pts = torch.tensor([[0.0, 40.1, 20.1, 0.3, 0.5],
                        [0.0, 40.1, 20.1, 0.1, 0.7]])
    out = lidar_priors(pts, batch_size=1)

    count_channel = out[0, 0]
    assert count_channel[200, 24] > 0
    assert count_channel.gt(0).sum() == 1, "points leaked into other cells"


def test_height_spread_reflects_the_points():
    cfg = LidarPriorConfig()
    pts = torch.tensor([[0.0, 10.1, 0.1, 0.5, 0.0],
                        [0.0, 10.1, 0.1, -0.5, 0.0]])
    out = lidar_priors(pts, batch_size=1, cfg=cfg)
    row, col = 50, 124
    assert abs(out[0, 2, row, col].item() - 1.0 / cfg.spread_scale) < 1e-5


def test_empty_cells_are_zero_not_sentinel():

    pts = torch.tensor([[0.0, 10.1, 0.1, 0.5, 1.0]])
    out = lidar_priors(pts, batch_size=1)
    assert out[0, 1, 0, 0].item() == 0.0
    assert out[0, 2, 0, 0].item() == 0.0


def test_out_of_range_points_are_dropped():
    pts = torch.tensor([[0.0, -5.0, 0.0, 0.0, 1.0],
                        [0.0, 100.0, 0.0, 0.0, 1.0]])
    out = lidar_priors(pts, batch_size=1)
    assert out[0, 0].sum() == 0.0


def identity_T(b=1):

    T = torch.zeros(b, 4, 4)
    T[:, 0, 2] = 1.0
    T[:, 1, 0] = -1.0
    T[:, 2, 1] = -1.0
    T[:, 3, 3] = 1.0
    return T


def test_camera_priors_shape_and_bounds():
    p = CameraPriors(CAM)
    K = torch.tensor([[700.0, 700.0, 640.0, 360.0]])
    out = p(K, identity_T())
    assert out.shape == (1, 3, *FUSION_GRID.camera_shape)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_raw_depth_error_grows_quadratically():

    p = CameraPriors(CAM)
    col = FUSION_GRID.n_y // 2
    near = p.depth_err[int(10 / FUSION_GRID.resolution), col].item()
    far = p.depth_err[int(20 / FUSION_GRID.resolution), col].item()
    assert abs(far / near - 4.0) < 0.05


def test_depth_error_matches_the_closed_form():
    p = CameraPriors(CAM)
    row, col = int(20 / FUSION_GRID.resolution), FUSION_GRID.n_y // 2
    r = torch.sqrt(p.cell_x[row, col] ** 2 + p.cell_y[row, col] ** 2)
    expected = (r ** 2) * CAM.disparity_error_px / (CAM.fx * CAM.baseline)
    assert abs(p.depth_err[row, col].item() - expected.item()) < 1e-4


def test_depth_error_never_saturates():

    p = CameraPriors(CAM)
    col = FUSION_GRID.n_y // 2
    profile = p.depth_err_norm[:, col]
    assert profile.max() < 1.0
    assert torch.all(profile[1:] - profile[:-1] > 0), "must stay strictly monotone"


def test_depth_error_normalisation_is_invertible():

    p = CameraPriors(CAM)
    row, col = int(35 / FUSION_GRID.resolution), FUSION_GRID.n_y // 2
    n = p.depth_err_norm[row, col].item()
    recovered = CAM.depth_error_scale * n / (1.0 - n)
    assert abs(recovered - p.depth_err[row, col].item()) < 1e-3


def test_frustum_mask_is_binary_and_partial():
    p = CameraPriors(CAM)
    K = torch.tensor([[700.0, 700.0, 640.0, 360.0]])
    mask = p.frustum_mask(K, identity_T())

    assert torch.all((mask == 0) | (mask == 1)), "mask must stay binary"
    covered = mask.mean().item()
    assert 0.3 < covered < 0.95, f"suspicious frustum coverage: {covered:.2f}"


def test_frustum_is_symmetric_about_the_vehicle_axis():

    p = CameraPriors(CAM)
    K = torch.tensor([[700.0, 700.0, 640.0, 360.0]])
    mask = p.frustum_mask(K, identity_T())[0]
    assert torch.equal(mask, mask.flip(-1))


def test_frustum_cache_is_reused():
    p = CameraPriors(CAM)
    K = torch.tensor([[700.0, 700.0, 640.0, 360.0]])
    T = identity_T()
    first = p.frustum_mask(K, T)
    assert p.frustum_mask(K, T) is first


def test_frustum_cache_invalidated_by_new_extrinsics():
    p = CameraPriors(CAM)
    K = torch.tensor([[700.0, 700.0, 640.0, 360.0]])
    first = p.frustum_mask(K, identity_T())

    rotated = identity_T()
    rotated[:, :3, :3] = torch.tensor([[0.0, 0.0, 1.0],
                                       [0.0, -1.0, 0.0],
                                       [1.0, 0.0, 0.0]])
    second = p.frustum_mask(K, rotated)
    assert not torch.equal(first, second)