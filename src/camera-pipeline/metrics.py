#metriche di validation per il warmup 2d

#precision, recall su detection di coni e distanza usando l'informazione di depth dai cones_2d.json


from pathlib import Path
from typing import Dict, List, Tuple
 
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


def match_detections_to_gt(
        detections: List[Dict],
        ground_truth: List[Dict],
        match_radius_px: float = 10.0
) -> Tuple[List[dict], List[dict], List[dict]]:
    
    #asxocia detections a coni nel gt usando una soglia minima di distanza
    #restituisce TP, FP, FN

    color_to_class = {"blue": 0, "yellow" : 1}

    gt_items = []

    for cone in ground_truth:
        
        if not cone.get("fully_in_image", True):
            continue

        cls = color_to_class.get(cone["color"])

        if cls is None:
            continue

        x, y = cone["center_px"]

        gt_items.append({
            "class": cls,
            "x_px": x,
            "y_px" : y,
            "depth_m": cone.get("depth_m", -1),
            "matched": False,
        })

    #ordina detection per score decrescente

    detections_sorted = sorted(detections, key=lambda d: -d["score"])

    true_positives = []
    false_positives = []


    for det in detections_sorted:
        best_idx = -1
        best_dist = float("inf")
        for i, gt in enumerate(gt_items):
            if gt["matched"]:
                continue
            if gt["class"] != det["class"]:
                continue
            dist = np.sqrt((det["x_px"] - gt["x_px"]) ** 2 + (det["y_px"] - gt["y_px"]) ** 2)
            if dist < match_radius_px and dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx >= 0:
            gt_items[best_idx]["matched"] = True
            true_positives.append({**det, "gt_depth_m": gt_items[best_idx]["depth_m"]})
        else:
            false_positives.append(det)
    
    false_negatives = [gt for gt in gt_items if not gt["matched"]]

    return true_positives, false_positives, false_negatives


def compute_metrics(
        all_tp: List[Dict],
        all_fp: List[Dict], 
        all_fn: List[Dict],
) -> Dict[str, float]:

    #calcolo di precision, recall, F1 e l'errore in base alla distanza

    tp_count = len(all_tp)
    fp_count = len(all_fp)
    fn_count = len(all_fn)

    precision = tp_count / max(tp_count + fp_count, 1)
    recall = tp_count / max(tp_count + fn_count, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
    }

    #stratificazione per distanza

    areas = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 100)]
    for lo, hi in areas:
        tp_in_bin = sum(1 for d in all_tp if lo <= d.get("gt_depth_m", -1) < hi)
        fn_in_bin = sum(1 for d in all_fn if lo <= d.get("depth_m", -1) < hi)
        total_gt = tp_in_bin + fn_in_bin
        recall_bin = tp_in_bin / max(total_gt, 1)
        metrics[f"recall_{lo}-{hi}m"] = recall_bin
        metrics[f"num_gt_{lo}-{hi}m"] = total_gt

    return metrics






