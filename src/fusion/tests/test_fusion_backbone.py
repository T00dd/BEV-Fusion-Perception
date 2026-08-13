"""End-to-end tests of the fusion backbone, with stub encoders.

No OpenPCDet, no checkpoints, no dataset: the encoders are injected, so the
composition can be tested on its own.

    python -m pytest fusion/tests/test_fusion_backbone.py -v
"""

import pytest
import torch
import torch.nn as nn

from fusion.encoders import FrozenEncoder
from fusion.fusion_backbone import FusionBackbone, FusionBackboneConfig
from fusion.grid_alignment import CAMERA_GRID, FUSION_GRID, LIDAR_GRID, GridSpec
from fusion.priors import CameraPriorConfig

CAM = CameraPriorConfig(fx=700.0, baseline=0.12, image_width=1280, image_height=720)
H, W = FUSION_GRID.camera_shape


class StubLidarEncoder(FrozenEncoder):
    def __init__(self, grid: GridSpec = LIDAR_GRID, channels: int = 256):
        super().__init__()
        self.out_channels, self.grid = channels, grid
        self.bn = nn.BatchNorm2d(channels)
        self.freeze()

    @torch.no_grad()
    def forward(self, batch_dict):
        b = batch_dict["batch_size"]
        return torch.randn(b, self.out_channels, *self.grid.lidar_shape)


class StubCameraEncoder(FrozenEncoder):
    def __init__(self, grid: GridSpec = CAMERA_GRID, channels: int = 128):
        super().__init__()
        self.out_channels, self.grid = channels, grid
        self.bn = nn.BatchNorm2d(channels)
        self.freeze()

    @torch.no_grad()
    def forward(self, batch):
        b = batch["K"].shape[0]
        feats = torch.randn(b, self.out_channels, *self.grid.camera_shape)
        counts = torch.randint(0, 12, (b, 1, *self.grid.camera_shape)).float()
        return feats, counts


def make_backbone(**kwargs):
    return FusionBackbone(
        StubLidarEncoder(), StubCameraEncoder(),
        FusionBackboneConfig(camera_prior=CAM, **kwargs),
    )


def make_batch(b=2, n_points=4000):
    torch.manual_seed(0)
    points = torch.stack([
        torch.randint(0, b, (n_points,)).float(),
        torch.rand(n_points) * 50.0,
        torch.rand(n_points) * 50.0 - 25.0,
        torch.rand(n_points) * 4.0 - 3.0,
        torch.rand(n_points),
    ], dim=1)

    T = torch.zeros(b, 4, 4)
    T[:, 0, 2], T[:, 1, 0], T[:, 2, 1], T[:, 3, 3] = 1.0, -1.0, -1.0, 1.0

    return {
        "lidar": {"batch_size": b, "points": points},
        "camera": {
            "K": torch.tensor([[700.0, 700.0, 640.0, 360.0]]).repeat(b, 1),
            "T": T,
        },
    }


def test_output_shape_and_channels():
    model = make_backbone()
    out, _ = model(make_batch())
    assert out.shape == (2, 256, H, W)


def test_delta_is_exactly_zero_at_init():

    model = make_backbone().eval()
    torch.manual_seed(7)
    _, aux = model(make_batch(), return_aux=True)
    assert torch.all(aux["delta"] == 0)
    assert aux["delta_norm"].item() == 0.0


def test_gate_is_uniform_at_init():
    model = make_backbone().eval()
    _, aux = model(make_batch(), return_aux=True)
    g = aux["gate"]
    assert abs(g.mean().item() - 0.8808) < 1e-3
    assert g.std().item() < 1e-6, "gate must start uniform, not noisy"


def test_gradients_reach_the_zero_conv():

    model = make_backbone()
    out, _ = model(make_batch())
    out.sum().backward()

    assert model.fusion.zero_conv.weight.grad is not None
    assert model.fusion.zero_conv.weight.grad.abs().sum() > 0


def test_frozen_encoders_get_no_gradient():
    model = make_backbone()
    out, _ = model(make_batch())
    out.sum().backward()

    for p in model.lidar_encoder.parameters():
        assert p.grad is None
    for p in model.camera_encoder.parameters():
        assert p.grad is None


def test_train_does_not_wake_the_encoders():
    model = make_backbone()
    model.train()
    assert model.fusion.training
    assert not model.lidar_encoder.training
    assert not model.camera_encoder.training


def test_trainable_budget():
    model = make_backbone()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 350_000 < trainable < 420_000, f"unexpected budget: {trainable}"


def test_param_groups_exclude_the_zero_conv_from_decay():
    model = make_backbone()
    groups = model.param_groups(lr=1e-3, weight_decay=1e-4)

    assert groups[1]["weight_decay"] == 0.0
    undecayed = {id(p) for p in groups[1]["params"]}
    assert id(model.fusion.zero_conv.weight) in undecayed
    assert id(model.fusion.zero_conv.bias) in undecayed
    assert id(model.fusion.context[0].weight) not in undecayed


def test_param_groups_cover_every_trainable_parameter():
    model = make_backbone()
    groups = model.param_groups(lr=1e-3)
    covered = sum(p.numel() for g in groups for p in g["params"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert covered == trainable


def test_diagnostics_keys():
    model = make_backbone().eval()
    _, aux = model(make_batch(), return_aux=True)
    d = model.diagnostics(aux)
    assert set(d) == {
        "gate_mean", "gate_std", "delta_norm",
        "camera_dead_channels", "w_lidar", "w_camera",
    }
    assert abs(d["w_lidar"] - d["w_camera"]) < 0.5, "groups must start balanced"


def test_gate_by_range_is_monotone_in_the_bins():
    model = make_backbone().eval()
    _, aux = model(make_batch(), return_aux=True)
    centres, means = model.gate_by_range(aux, n_bins=8)

    assert len(centres) == len(means) > 0
    assert all(c2 > c1 for c1, c2 in zip(centres, centres[1:]))
    assert all(abs(m - 0.8808) < 1e-3 for m in means), "flat at init by construction"


def test_assert_zero_init_catches_an_overwrite():
    model = make_backbone()
    model.assert_zero_init()

    nn.init.kaiming_uniform_(model.fusion.zero_conv.weight)  # simulate model.apply
    with pytest.raises(AssertionError):
        model.assert_zero_init()


def test_mismatched_encoder_extent_is_rejected():

    bad = StubCameraEncoder(grid=GridSpec(x_max=60.0, resolution=0.25))
    with pytest.raises(ValueError, match="extent"):
        FusionBackbone(StubLidarEncoder(), bad, FusionBackboneConfig(camera_prior=CAM))


def test_channel_widths_follow_the_encoders():
    model = FusionBackbone(
        StubLidarEncoder(channels=384), StubCameraEncoder(channels=64),
        FusionBackboneConfig(camera_prior=CAM),
    )
    assert model.fusion.cfg.lidar_channels == 384
    assert model.fusion.cfg.camera_channels == 64
    out, _ = model(make_batch())
    assert out.shape[1] == 384


def test_priors_stay_bounded_on_real_shaped_input():
    model = make_backbone().eval()
    batch = make_batch()
    from fusion.priors import lidar_priors

    p_l = lidar_priors(batch["lidar"]["points"], 2, FUSION_GRID, model.cfg.lidar_prior)
    p_c = model.camera_priors(batch["camera"]["K"], batch["camera"]["T"],
                              torch.randint(0, 20, (2, 1, 200, 200)).float())
    assert p_l.shape == (2, 4, H, W) and 0.0 <= p_l.min() and p_l.max() <= 1.0
    assert p_c.shape == (2, 4, H, W) and 0.0 <= p_c.min() and p_c.max() <= 1.0