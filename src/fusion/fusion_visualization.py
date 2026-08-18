from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .decode import decode_detections
from .grid_alignment import FUSION_GRID

__all__ = ["save_bev_visualizations", "save_confusion_matrix"]

CLASS_COLORS = {0: "#2a78d6", 1: "#eda100", 2: "#eb6834"}
NO_COLOR = "#9a9a9a" #cono localizzato ma senza informazione cromatica
CLASS_NAMES = ("blue", "yellow", "orange")


def _draw_bev_panel(ax, cones, grid, title, gt_cones=None, fov_params=None):
    for r in range(10, int(grid.x_max) + 1, 10):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color="0.85", lw=0.8, zorder=0))
        ax.text(0, r, f"{r}m", color="0.55", fontsize=7, ha="center", va="bottom", zorder=1)
    ax.plot(0, 0, marker="^", color="0.15", markersize=11, zorder=6)

    if fov_params is not None:
        cx, cy, slope = fov_params["cam_x"], fov_params["cam_y"], fov_params["slope"]
        x_end = grid.x_max
        y_left = cy + (x_end - cx) * slope
        y_right = cy - (x_end - cx) * slope
        ax.plot([cy, y_left], [cx, x_end], color="lime", linestyle="--", linewidth=1.5, alpha=0.8, zorder=2)
        ax.plot([cy, y_right], [cx, x_end], color="lime", linestyle="--", linewidth=1.5, alpha=0.8, zorder=2)
        ax.fill_betweenx([cx, x_end], [cy, y_right], [cy, y_left], color="lime", alpha=0.06, zorder=1)

    if gt_cones is not None:
        for c in gt_cones:
            ax.scatter(c["y"], c["x"], s=110, facecolors="none", edgecolors=CLASS_COLORS.get(c["cls"], NO_COLOR), linewidths=1.3, alpha=0.45, zorder=3)

    for c in cones:
        #grigio = la head ha localizzato il cono ma non ha informazione cromatica affidabile (fase 0 oppure cella fuori dal fov camera)
        face = CLASS_COLORS.get(c["cls"], NO_COLOR) if c.get("colored", True) else NO_COLOR
        ax.scatter(c["y"], c["x"], s=42, color=face, edgecolors="k", linewidths=0.4, zorder=4)

    ax.set_xlim(grid.y_max, grid.y_min)
    ax.set_ylim(grid.x_min, grid.x_max)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("y laterale (m)")
    ax.set_ylabel("x avanti (m)")
    ax.grid(True, color="0.93", lw=0.5)


@torch.no_grad()
def save_bev_visualizations(
    predictions: Dict[str, torch.Tensor],
    cones: torch.Tensor,
    sample_ids: List[str],
    output_dir,
    grid=FUSION_GRID,
    K: Optional[torch.Tensor] = None,
    T: Optional[torch.Tensor] = None,
    threshold: float = 0.3,
    color_conf_threshold: float = 0.45,
    max_to_save: int = 8,
    tag: str = "",
) -> int:
    #restituisce quante figure ha salvato
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dets_batch = decode_detections(predictions, grid, threshold=threshold)
    cones_np = cones.detach().cpu().numpy()
    n = min(max_to_save, len(sample_ids))

    for i in range(n):
        rows = cones_np[cones_np[:, 0] == i]
        gt_cones = [{"x": float(r[1]), "y": float(r[2]), "cls": int(r[4])} for r in rows]


        pred_cones = []
        for d in dets_batch[i]:
            #il colore è considerato affidabile solo se la softmax è chiaramente sopra il caso (1/3)
            # in fase 0 il ramo colore è congelato e resta a 0.333 quindi i coni escono tutti grigi
            pred_cones.append({
                "x": d["x"], "y": d["y"], "cls": d["class_id"],
                "colored": d["color_score"] >= color_conf_threshold,
            })

        fov = None
        if K is not None and T is not None:
            k, t = K[i].detach().cpu().numpy(), T[i].detach().cpu().numpy()
            # K e' [fx, fy, cx, cy], non una 3x3: la pendenza del FOV e' cx/fx
            fov = {"cam_x": float(t[0, 3]), "cam_y": float(t[1, 3]),
                   "slope": float(k[2] / k[0])}

        fig, axs = plt.subplots(1, 2, figsize=(11, 6.5))
        _draw_bev_panel(axs[0], gt_cones, grid, "BEV GT (tutti i coni)",fov_params=fov)
        _draw_bev_panel(axs[1], pred_cones, grid, "BEV pred (GT in trasparenza, grigio = senza colore)", gt_cones=gt_cones, fov_params=fov)


        fig.suptitle(f"{sample_ids[i]}  -  {tag}", fontsize=11)
        fig.tight_layout()
        fig.savefig(output_dir / f"{sample_ids[i].replace('/', '_')}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    return n


def save_confusion_matrix(matrix: np.ndarray, output_dir, tag: str = ""):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n = matrix.shape[0]
    norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(1.6 * n + 2.2, 1.6 * n + 1.8))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for a in range(n):
        for b in range(n):
            ax.text(b, a, f"{matrix[a, b]}\n{norm[a, b]:.2f}", ha="center", va="center", fontsize=9, color="white" if norm[a, b] > 0.5 else "black")
    ax.set_xticks(range(n), CLASS_NAMES[:n])
    ax.set_yticks(range(n), CLASS_NAMES[:n])
    ax.set_xlabel("predetto")
    ax.set_ylabel("ground truth")
    ax.set_title(f"colore, solo sui TP  -  {tag}", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=110, bbox_inches="tight")
    plt.close(fig)