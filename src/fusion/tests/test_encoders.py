import pytest
import torch
import torch.nn as nn

from fusion.encoders import FrozenEncoder
from fusion.grid_alignment import LIDAR_GRID, GridSpec


class DummyEncoder(FrozenEncoder):

    def __init__(self, grid: GridSpec = LIDAR_GRID, channels: int = 8):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1), nn.BatchNorm2d(channels))
        self.out_channels = channels
        self.grid = grid
        self.freeze()

    @torch.no_grad()
    def forward(self, batch):
        return self.net(batch["x"])


def test_pesi_senza_gradiente():
    enc = DummyEncoder()
    assert all(not p.requires_grad for p in enc.parameters())


def test_parte_in_eval():
    enc = DummyEncoder()
    assert not enc.training
    assert not enc.net[1].training


def test_train_non_lo_risveglia():

    enc = DummyEncoder()
    enc.train()
    assert not enc.training
    assert not enc.net[1].training


def test_train_annidato_non_lo_risveglia():

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = DummyEncoder()
            self.fusion = nn.Conv2d(8, 8, 1)

    model = Wrapper()
    model.train()

    assert model.fusion.training, "la parte allenabile deve stare in train mode"
    assert not model.encoder.net[1].training, "la BN congelata si e' risvegliata"


def test_running_stats_non_si_muovono():
    
    enc = DummyEncoder()
    model = nn.ModuleDict({"encoder": enc, "head": nn.Conv2d(8, 1, 1)})
    model.train()

    x = torch.randn(2, 3, 16, 16)
    first = enc({"x": x}).clone()
    mean_before = enc.net[1].running_mean.clone()

    for _ in range(5):
        enc({"x": torch.randn(2, 3, 16, 16)})

    assert torch.equal(enc.net[1].running_mean, mean_before), "running stats derivate"
    assert torch.equal(enc({"x": x}), first), "uscita non deterministica"


def test_check_output_su_canali_sbagliati():
    enc = DummyEncoder(channels=8)
    with pytest.raises(RuntimeError, match="canali"):
        enc.check_output(torch.zeros(1, 16, 250, 250), LIDAR_GRID.lidar_shape)


def test_check_output_su_shape_sbagliata():

    enc = DummyEncoder(channels=8)
    with pytest.raises(RuntimeError, match="UPSAMPLE_STRIDES"):
        enc.check_output(torch.zeros(1, 8, 125, 125), LIDAR_GRID.lidar_shape)


def test_no_grad_nel_forward():
    enc = DummyEncoder()
    out = enc({"x": torch.randn(1, 3, 16, 16)})
    assert not out.requires_grad