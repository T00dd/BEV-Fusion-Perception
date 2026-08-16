from typing import Dict, List

import torch
import torch.nn.functional as F

from camera_detection.metrics import extract_peaks_from_heatmap  # reused unchanged from the camera branch
from .grid_alignment import FUSION_GRID, GridSpec

__all__ = ["decode_detections"]


@torch.no_grad()
def decode_detections(
    predictions: Dict[str, torch.Tensor],
    grid: GridSpec = FUSION_GRID,
    threshold: float = 0.3,
    color_pool: int = 1,
    max_detections: int = 200,
) -> List[List[Dict]]:
    
    #one list of detections per sample
    #positions in metres in the label frame

    presence = torch.sigmoid(predictions["presence_logits"].float()).cpu()
    offset = predictions["offset_pred"].float().cpu()
    color_logits = predictions["color_logits"].float().cpu()

    #colour is read at the peak cell only
    if color_pool > 1:
        color_logits = F.avg_pool2d(
            color_logits, color_pool, stride=1, padding=color_pool // 2
        )
    color_prob = torch.softmax(color_logits, dim=1)

    out = []
    for i in range(presence.shape[0]):
        peaks = extract_peaks_from_heatmap(
            presence[i], offset[i], stride=1,
            threshold=threshold, max_detections=max_detections,
        )

        dets = []
        for p in peaks:
            row_f, col_f = p["y"], p["x"]  # extractor returns (col, row) as (x, y)
            x, y = grid.grid_to_world(row_f, col_f)
            probs = color_prob[i, :, int(row_f), int(col_f)]
            dets.append({
                "x": float(x),
                "y": float(y),
                "score": p["score"],
                "class_id": int(probs.argmax()),
                "color_score": float(probs.max()),
            })
        out.append(dets)

    return out