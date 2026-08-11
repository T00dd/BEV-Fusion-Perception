#metriche di validation per il warmup 2d

#precision, recall su detection di coni e distanza usando l'informazione di depth dai cones_2d.json

import json
from pathlib import Path
from typing import Dict, List, Tuple
 
import numpy as np
import torch
import torch.nn.functional as F


DISTANCE_BINS_2D = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 100)]
DISTANCE_BINS_BEV = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 50)]


def extract_peaks_from_heatmap(
        heatmap: torch.Tensor,
        offset: torch.Tensor,
        stride: int,
        threshold: float = 0.2,
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
        match_radius_px: float = 10.0,
        color_to_class: Dict[str, int] = None,
        class_agnostic: bool = False
) -> Tuple[List[dict], List[dict], List[dict]]:
    
    #asxocia detections a coni nel gt usando una soglia minima di distanza
    #restituisce TP, FP, FN

    if color_to_class is None:
        color_to_class = {
            "blue": 0,
            "yellow": 1, 
            "orange_small": 2
        }
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
            if not class_agnostic and gt["class"] != det["class_id"]:
                continue
            dist = np.sqrt((det["x"] - gt["x_px"]) ** 2 + (det["y"] - gt["y_px"]) ** 2)
            if dist < match_radius_px and dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx >= 0:
            gt_items[best_idx]["matched"] = True
            true_positives.append({**det,
                                    "gt_depth_m": gt_items[best_idx]["depth_m"],
                                    "gt_class": gt_items[best_idx]["class"]})
        else:
            false_positives.append(det)
    
    false_negatives = [gt for gt in gt_items if not gt["matched"]]

    return true_positives, false_positives, false_negatives


def compute_metrics(
        all_tp: List[Dict],
        all_fp: List[Dict], 
        all_fn: List[Dict],
        distance_bins: List[Tuple[float, float]] = None,
        num_classes: int = 2,
) -> Dict[str, float]:

    #calcolo di precision, recall, F1 e l'errore in base alla distanza

    if distance_bins is None:
        distance_bins = DISTANCE_BINS_2D

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

    for lo, hi in distance_bins:
        tp_in_bin = sum(1 for d in all_tp if lo <= d.get("gt_depth_m", -1) < hi)
        fn_in_bin = sum(1 for d in all_fn if lo <= d.get("depth_m", -1) < hi)
        total_gt = tp_in_bin + fn_in_bin
        metrics[f"recall_{lo}-{hi}m"] = tp_in_bin / max(total_gt, 1)
        metrics[f"num_gt_{lo}-{hi}m"] = total_gt

    #stratificazione per classe

    for c in range(num_classes):
        tp_c = sum(1 for d in all_tp if d["class_id"] == c)
        fp_c = sum(1 for d in all_fp if d["class_id"] == c)
        fn_c = sum(1 for g in all_fn if g["class"] == c)
        p_c = tp_c / max(tp_c + fp_c, 1)
        r_c = tp_c / max(tp_c + fn_c, 1)
        metrics[f"precision_c{c}"] = p_c
        metrics[f"recall_c{c}"] = r_c
        metrics[f"f1_c{c}"] = 2 * p_c * r_c / max(p_c + r_c, 1e-6)
        metrics[f"num_gt_c{c}"] = tp_c + fn_c

    return metrics


def confusion_matrix_from_tp(tp_agnostic: List[Dict], num_classes: int) -> np.ndarray:
    #matrice di confusione dei colori
    #righe = classe GT
    #colonne = classe predetta

    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for d in tp_agnostic:
        matrix[d["gt_class"], d["class_id"]] += 1
    return matrix


def compute_color_metrics(tp_agnostic: List[Dict], num_classes: int) -> Dict[str, float]:
    matrix = confusion_matrix_from_tp(tp_agnostic, num_classes)

    total = max(matrix.sum(), 1)
    out = {"color_accuracy": float(np.trace(matrix)) / total}

    for gt_c in range(num_classes):
        denom = max(matrix[gt_c].sum(), 1)
        out[f"color_acc_c{gt_c}"] = float(matrix[gt_c, gt_c]) / denom

    #confusione blu e giallo: l'errore che inverte i lati del tracciato
    if num_classes >= 2:
        out["confusion_blue_as_yellow"] = int(matrix[0, 1])
        out["confusion_yellow_as_blue"] = int(matrix[1, 0])

    return out


def compute_ap(all_tp: List[Dict], all_fp: List[Dict], num_gt: int) -> float:
    #average precision
    items = [(d["score"], 1) for d in all_tp] + [(d["score"], 0) for d in all_fp]
    items.sort(key=lambda t: -t[0])

    tp_cum = fp_cum = 0
    ap = prev_recall = 0.0
    for score, is_tp in items:
        tp_cum += is_tp
        fp_cum += 1 - is_tp
        rec = tp_cum / max(num_gt, 1)
        prec = tp_cum / max(tp_cum + fp_cum, 1)
        ap += prec * (rec - prev_recall)
        prev_recall = rec
    return ap


class ValidationAccumulator:

    #accumula TP, FP, FN su tutti i batch di validation per calcolare metriche globali

    def __init__(self, dataset_root: Path, stride: int, threshold: float = 0.3, match_radius_px: float = 10.0, num_classes: int = 3):
        self.dataset_root = Path(dataset_root)
        self.stride = stride
        self.threshold = threshold
        self.match_radius_px = match_radius_px
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.all_tp = []
        self.all_fp = []
        self.all_fn = []
        self.all_tp_agnostic = []

    def update(
        self,
        heatmap_logits: torch.Tensor,
        offset_pred: torch.Tensor,
        sample_ids: List[str],
    ):
        #si processa un batch di predizioni

        heatmap_probs = torch.sigmoid(heatmap_logits)
        batch_dim = heatmap_probs.shape[0]


        for b in range(batch_dim):

            detections = extract_peaks_from_heatmap(
                heatmap_probs[b].cpu(),
                offset_pred[b].cpu(),
                stride = self.stride,
                threshold=self.threshold,
            )

            sample_id = sample_ids[b]
            scene_id, frame_stem = sample_id.split("/")
            cones_path = self.dataset_root / "scenes" / scene_id / "labels_2d" / f"{frame_stem}_cam_left.json"
            with open(cones_path, "r") as f:
                cones_data = json.load(f)


            tp, fp, fn = match_detections_to_gt(
                detections, 
                cones_data["cones_in_image"],
                match_radius_px=self.match_radius_px,
            )

            self.all_tp.extend(tp)
            self.all_fp.extend(fp)
            self.all_fn.extend(fn)

            tp_agn, _, _ = match_detections_to_gt(
                detections,
                cones_data["cones_in_image"],
                match_radius_px=self.match_radius_px,
                class_agnostic=True,
            )
            self.all_tp_agnostic.extend(tp_agn)


    def compute(self) -> Dict[str, float]:
        m = compute_metrics(self.all_tp, self.all_fp, self.all_fn, distance_bins=DISTANCE_BINS_2D, num_classes=self.num_classes)
        m.update(compute_color_metrics(self.all_tp_agnostic, self.num_classes))
        m["ap"] = compute_ap(self.all_tp, self.all_fp, num_gt=len(self.all_tp) + len(self.all_fn))
        return m