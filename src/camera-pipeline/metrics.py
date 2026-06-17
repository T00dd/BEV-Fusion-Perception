#metriche di validation per il warmup 2d

#precision, recall su detection di coni e distanza usando l'informazione di depth dai cones_2d.json


from pathlib import Path
from typing import Dict, List
 
import numpy as np
import torch
import torch.nn.functional as F


def extract_peaks_from_heatmap(
        heatmap: torch.Tensor,
        offset: torch.Tensor,
        stride: int,
        threshold: float = 0.5,
        nms_kernel_size: int = 3,
        max_detections: int = 100,
) -> List[Dict]:
    
    #estrazione dei picchi da una heatmap predetta

    C, H, W = heatmap.shape

    #nms via max pooling
    padding = nms_kernel_size // 2
    pooling = F.max_pool2d(heatmap.unsqueeze(0), kernel_size=nms_kernel_size, stride=1, padding=padding).squeeze(0)
    keep_mask = (heatmap == pooling).float() * heatmap #0 dove non è un picco locale


    #estrazione di tutti i pixel sopra la soglia 
    detections = []

    for c in range(C):
        map  = keep_mask[c] #(H, W)
        scores_flat = map.flatten() #(H*W) per rendere più facile l'estrazione degli indici

        top_scores, top_indices = torch.topk(scores_flat, k=min(max_detections, scores_flat.numel())) #max_detections

        for score, idx in zip(top_scores, top_indices):
            score_value = score.item()
            if score_value < threshold:
                break  #salta se sono sotto la soglia

            iy = (idx // W).item()
            ix = (idx % W).item()


            dy = offset[0, iy, ix].item()  #offset in y
            dx = offset[1, iy, ix].item()  #offset in x

            #convertimento in coordinate immagine
            x = (ix + dx) * stride
            y = (iy + dy) * stride

            detections.append({
                "class_id": c,
                "score": score_value,
                "x": x,
                "y": y,
            })

    return detections

