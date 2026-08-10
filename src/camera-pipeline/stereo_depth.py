#calcolo della depth stereo con SGBM + left right consistency check




from typing import Dict, Optional, Tuple
 
import cv2
import numpy as np

_DEFAULTS = {
    "min_disp": 0,
    "num_disp": 128,                #deve coprire la disparità a distanza MINIMA
    "block_size": 5,                #dimensione del blocco di matching
    "uniqueness_ratio": 10,         #percentuale di differenza tra il miglior match e il secondo miglior match
    "speckle_window_size": 100,     #dimensione della finestra per la rimozione dei pixel speckle
    "speckle_range": 2,             #range di disparità per la rimozione dei pixel speckle
    "disp12_max_diff": 1,           #differenza massima tra la disparità sinistra e destra per considerare un match valido
    "lr_consistency": True,         #attiva/disattiva left-right consistency check
    "lr_max_diff": 1.0,             #px di tolleranza tra le due disparità
}
 
 
def compute_depth_from_stereo(
    left: np.ndarray,
    right: np.ndarray,
    fx: float,
    baseline_m: float,
    sgbm_params: Optional[Dict] = None,
    min_depth_m: float = 0.3,
    max_depth_m: float = 100.0,
) -> np.ndarray:
    
    #coppia stereo rettificata -> depth in metri (invalidi = 0.0)
    p = {**_DEFAULTS, **(sgbm_params or {})}
 
    gl = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY) if left.ndim == 3 else left
    gr = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY) if right.ndim == 3 else right
 
    bs = p["block_size"]
    num_disp = int(np.ceil(p["num_disp"] / 16.0) * 16)   #deve essere multiplo di 16

    matcher = cv2.StereoSGBM_create(
        minDisparity=p["min_disp"], numDisparities=num_disp, blockSize=bs,
        P1=8 * bs * bs, P2=32 * bs * bs,                 #penalità smoothness standard
        disp12MaxDiff=p["disp12_max_diff"],
        uniquenessRatio=p["uniqueness_ratio"],
        speckleWindowSize=p["speckle_window_size"],
        speckleRange=p["speckle_range"],
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
 
    #SGBM restituisce la disparita in fixed-point 1/16
    disp = matcher.compute(gl, gr).astype(np.float32) / 16.0
    valid = disp > (p["min_disp"] - 0.5)
 
    if p["lr_consistency"]:
        disp_r = matcher.compute(gr[:, ::-1], gl[:, ::-1]).astype(np.float32)[:, ::-1] / 16.0
        H, W = disp.shape
        u = np.arange(W)[None, :].repeat(H, 0)
        u_right = np.round(u - disp).astype(np.int32)
        in_range = (u_right >= 0) & (u_right < W)
        rows = np.arange(H)[:, None].repeat(W, 1)
        disp_r_at = np.full_like(disp, np.nan)
        disp_r_at[in_range] = disp_r[rows[in_range], u_right[in_range]]
        valid &= in_range & np.isfinite(disp_r_at) & (np.abs(disp - disp_r_at) <= p["lr_max_diff"])
 

    depth = np.zeros_like(disp)
    ok = valid & (disp > 1e-3)
    depth[ok] = fx * baseline_m / disp[ok]
    depth[(depth < min_depth_m) | (depth > max_depth_m)] = 0.0   # scarta depth instabili
    return depth
 
 
def compare_depth(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    max_depth_m: float = 50.0,
    bins: Tuple[Tuple[float, float], ...] = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 50)),
) -> Dict[str, float]:
    
    #confronta la depth SGBM con quella GT di CARLA sui soli pixel dove entrambe sono valide
    #la coverage è riportata a parte: una depth accurata sul 20% dei pixel è inutile per il lifting
    #stratifica per distanza perchè l'errore stereo cresce con z^2

    vp = np.isfinite(depth_pred) & (depth_pred > 0)
    vg = np.isfinite(depth_gt) & (depth_gt > 0) & (depth_gt <= max_depth_m)
    both = vp & vg
 
    stats = {"coverage": float(vp[vg].mean()) if vg.any() else 0.0, "num_valid_px": int(both.sum())}

    if not both.any():
        return {**stats, "mae_m": float("nan"), "rmse_m": float("nan"), "bias_m": float("nan"), "pct_within_0.5m": 0.0}
 
    err = depth_pred[both] - depth_gt[both]
    gt = depth_gt[both]
    stats["mae_m"] = float(np.mean(np.abs(err)))
    stats["rmse_m"] = float(np.sqrt(np.mean(err ** 2)))
    stats["bias_m"] = float(np.mean(err))                 # + = sovrastima la distanza
    stats["pct_within_0.5m"] = float(np.mean(np.abs(err) < 0.5))
 
    for lo, hi in bins:
        m = (gt >= lo) & (gt < hi)
        stats[f"mae_{lo}-{hi}m"] = float(np.mean(np.abs(err[m]))) if m.any() else float("nan")
        gb = vg & (depth_gt >= lo) & (depth_gt < hi)
        stats[f"coverage_{lo}-{hi}m"] = float(vp[gb].mean()) if gb.any() else 0.0
 
    return stats
