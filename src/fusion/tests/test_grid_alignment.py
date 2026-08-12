import math
 
import pytest
import torch
 
from fusion.grid_alignment import (
    CAMERA_GRID,
    FUSION_GRID,
    LIDAR_GRID,
    GridSpec,
    align_camera_features,
    align_lidar_features,
    camera_convention_to_lidar_raster,
    lidar_raster_to_camera_convention,
    occupancy_from_points,
    resample_camera_convention,
)

# reference testing point
X_REF, Y_REF = 40.1, 20.1

def test_forme_delle_griglie():
    assert LIDAR_GRID.camera_shape == (250, 250)
    assert CAMERA_GRID.camera_shape == (200, 200)
    assert FUSION_GRID is LIDAR_GRID

def test_indici_del_punto_di_riferimento():
    
    i_y, i_x = FUSION_GRID.world_to_lidar_raster(X_REF, Y_REF)
    row, col = FUSION_GRID.world_to_grid(X_REF, Y_REF)
 
    assert math.floor(i_y) == 225
    assert math.floor(i_x) == 200
    assert math.floor(row) == 200
    assert math.floor(col) == 24
    assert math.floor(col) == (FUSION_GRID.n_y - 1) - math.floor(i_y)
 
 
def test_identita_degli_indici_su_punti_casuali():
    
    g = FUSION_GRID
    torch.manual_seed(0)
    x = torch.rand(20000) * (g.x_max - g.x_min) + g.x_min
    y = torch.rand(20000) * (g.y_max - g.y_min) + g.y_min
 
    i_y, i_x = g.world_to_lidar_raster(x, y)
    row, col = g.world_to_grid(x, y)
 
    u = (y - g.y_min) / g.resolution
    interior = (u - u.floor()).abs() > 1e-9
 
    assert torch.equal(row.floor()[interior], i_x.floor()[interior])
    assert torch.equal(col.floor()[interior], (g.n_y - 1) - i_y.floor()[interior])

def test_permutazione_su_tensore():

    g = FUSION_GRID
    L = torch.zeros(1, 1, *g.lidar_shape)
    L[0, 0, 225, 200] = 1.0
 
    A = lidar_raster_to_camera_convention(L)
 
    assert A.shape == (1, 1, *g.camera_shape)
    assert A[0, 0, 200, 24].item() == 1.0
    assert A.sum().item() == 1.0  # niente perso, niente duplicato

def test_non_e_uno_specchio():
    """Il fallimento tipico e' una riflessione: blu e giallo si scambiano di lato.
 
    Due punti simmetrici rispetto all'asse del veicolo devono restare su lati
    opposti, e nell'ordine giusto: y > 0 (sinistra) sta a colonna BASSA.
    """
    g = FUSION_GRID
    L = torch.zeros(1, 1, *g.lidar_shape)
 
    i_y_sx, i_x_sx = g.world_to_lidar_raster(30.1, 10.1)   # sinistra
    i_y_dx, i_x_dx = g.world_to_lidar_raster(30.1, -10.1)  # destra
    L[0, 0, int(i_y_sx), int(i_x_sx)] = 1.0
    L[0, 0, int(i_y_dx), int(i_x_dx)] = 2.0
 
    A = lidar_raster_to_camera_convention(L)[0, 0]
    col_sx = A.eq(1.0).nonzero()[0, 1].item()
    col_dx = A.eq(2.0).nonzero()[0, 1].item()
 
    assert col_sx < col_dx, "sinistra e destra invertite: la trasformazione specchia"
 
 
def test_permutazione_invertibile():
    torch.manual_seed(1)
    L = torch.randn(2, 5, *FUSION_GRID.lidar_shape)
    back = camera_convention_to_lidar_raster(lidar_raster_to_camera_convention(L))
    assert torch.equal(L, back)
 
 
def test_griglia_non_quadrata():
    
    g = GridSpec(x_min=0.0, x_max=40.0, y_min=-10.0, y_max=10.0, resolution=0.5)
    assert g.lidar_shape == (40, 80)
    assert g.camera_shape == (80, 40)
 
    L = torch.zeros(1, 1, *g.lidar_shape)
    i_y, i_x = g.world_to_lidar_raster(12.3, 4.3)
    L[0, 0, int(i_y), int(i_x)] = 1.0
 
    A = lidar_raster_to_camera_convention(L)
    row, col = g.world_to_grid(12.3, 4.3)
 
    assert A.shape == (1, 1, *g.camera_shape)
    assert A[0, 0, int(row), int(col)].item() == 1.0
 
 
def test_world_grid_roundtrip():
    g = FUSION_GRID
    row, col = g.world_to_grid(X_REF, Y_REF)
    x, y = g.grid_to_world(row, col)
    assert abs(x - X_REF) < 1e-9
    assert abs(y - Y_REF) < 1e-9
 
 
def test_resample_e_metricamente_esatto():
    
    src, dst = CAMERA_GRID, FUSION_GRID
 
    cols = torch.arange(src.n_y, dtype=torch.float32) + 0.5
    y_src = src.y_max - cols * src.resolution
    t = y_src.view(1, 1, 1, -1).expand(1, 1, src.n_x, src.n_y).contiguous()
 
    out = resample_camera_convention(t, src, dst)
 
    cols_dst = torch.arange(dst.n_y, dtype=torch.float32) + 0.5
    y_expected = dst.y_max - cols_dst * dst.resolution
 
    interior = slice(1, dst.n_y - 1)
    err = (out[0, 0, dst.n_x // 2, interior] - y_expected[interior]).abs().max()
    assert err < 1e-4, f"errore metrico {err:.5f} m: controlla align_corners"
 
 
def test_align_corners_true_sarebbe_peggio():

    import torch.nn.functional as F
 
    src, dst = CAMERA_GRID, FUSION_GRID
    cols = torch.arange(src.n_y, dtype=torch.float32) + 0.5
    y_src = src.y_max - cols * src.resolution
    t = y_src.view(1, 1, 1, -1).expand(1, 1, src.n_x, src.n_y).contiguous()
 
    wrong = F.interpolate(t, size=dst.camera_shape, mode="bilinear", align_corners=True)
    cols_dst = torch.arange(dst.n_y, dtype=torch.float32) + 0.5
    y_expected = dst.y_max - cols_dst * dst.resolution
 
    err = (wrong[0, 0, dst.n_x // 2, :] - y_expected).abs().max().item()
    assert 0.02 < err < 0.03, f"scarto atteso ~0.025 m, misurato {err:.4f}"
 
 
def test_align_lidar_features_end_to_end():
    feat = torch.randn(2, 256, *LIDAR_GRID.lidar_shape)
    out = align_lidar_features(feat)
    assert out.shape == (2, 256, 250, 250)
    assert out.is_contiguous()
 
 
def test_align_camera_features_end_to_end():
    feat = torch.randn(2, 128, *CAMERA_GRID.camera_shape)
    out = align_camera_features(feat)
    assert out.shape == (2, 128, 250, 250)
 
 
def test_shape_sbagliate_falliscono_rumorosamente():

    with pytest.raises(ValueError):
        align_camera_features(torch.randn(2, 128, 250, 250))  # gia' a 0.2 m
    with pytest.raises(ValueError):
        resample_camera_convention(
            torch.randn(2, 8, 200, 200),
            CAMERA_GRID,
            GridSpec(x_max=60.0, resolution=0.2),  # estensione diversa
        )
 
 
def test_occupancy_scatter():
    pts = torch.tensor([[X_REF, Y_REF], [X_REF, Y_REF], [-5.0, 0.0]])
    occ = occupancy_from_points(pts)
    assert occ.shape == FUSION_GRID.camera_shape
    assert occ[200, 24].item() == 2.0  # due punti nella stessa cella
    assert occ.sum().item() == 2.0     # il terzo e' fuori griglia