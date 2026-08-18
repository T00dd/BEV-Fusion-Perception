import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .decode import decode_detections
from .grid_alignment import FUSION_GRID

__all__ = ["BEVValidationAccumulator", "RANGE_BANDS", "CLASS_NAMES", "compare_to_baseline"]

RANGE_BANDS = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 50))
CLASS_NAMES = ("blue", "yellow", "orange")
LOC_PERCENTILES = (50, 90, 95, 99)
LOC_THRESHOLDS_CM = (5, 10, 20)   #20 cm = una cella della griglia


class BEVValidationAccumulator:
    #accumula TP/FP/FN in metri e restituisce un dict piatto

    def __init__(self, grid=FUSION_GRID, threshold: float = 0.3, match_radius_m: float = 0.5, min_lidar_points: int = 0):
        self.grid = grid
        self.threshold = threshold
        self.radius = match_radius_m
        #min_lidar_points > 0 replica il filtro gt del ramo lidar: i coni con meno ritorni sono invisibili al sensore e il baseline li esclude dal denominatore
        self.min_lidar_points = min_lidar_points
        self.reset()

    def reset(self):
        self.tp: List[Dict] = []
        self.fp: List[Dict] = []
        self.fn: List[Dict] = []
        self.n_frames = 0
        #esito per istanza, chiave "frame|gt_idx"
        #val loader ha shuffle=False quindi il contatore di frame è stabile fra run diverse
        self.instance_hit: Dict[str, int] = {}

    @torch.no_grad()
    def update(self, predictions: Dict[str, torch.Tensor], cones: torch.Tensor):
        #cones: (M, 5) come [batch_idx, x, y, z, classe]
        batch = decode_detections(predictions, self.grid, threshold=self.threshold)
        cones = cones.detach().cpu().numpy()

        for i, dets in enumerate(batch):
            gt = cones[cones[:, 0] == i]
            if self.min_lidar_points > 0 and gt.shape[1] > 5:
                gt = gt[gt[:, 5] >= self.min_lidar_points]
            taken = np.zeros(len(gt), dtype=bool)
            frame = self.n_frames
            self.n_frames += 1

            #greedy per punteggio decrescente
            #CLASS-AGNOSTIC: un errore di colore non deve trasformare un cono ben localizzato in FP + FN
            for d in sorted(dets, key=lambda x: -x["score"]):
                best, best_dist = -1, self.radius

                for j in range(len(gt)):
                    if taken[j]:
                        continue
                    dist = float(np.hypot(d["x"] - gt[j, 1], d["y"] - gt[j, 2]))
                    if dist < best_dist:
                        best, best_dist = j, dist

                if best < 0:
                    self.fp.append({"range": float(np.hypot(d["x"], d["y"])), "score": d["score"]})
                    continue

                taken[best] = True
                self.tp.append({
                    "dist": best_dist,
                    "range": float(np.hypot(gt[best, 1], gt[best, 2])),
                    "gt_class": int(gt[best, 4]),
                    "pred_class": d["class_id"],
                })

            for j in range(len(gt)):
                if not taken[j]:
                    self.fn.append({"range": float(np.hypot(gt[j, 1], gt[j, 2])), "gt_class": int(gt[j, 4])})

            for j in range(len(gt)):
                self.instance_hit[f"{frame}|{j}"] = int(taken[j])

    def confusion_matrix(self) -> np.ndarray:
        cm = np.zeros((3, 3), dtype=int)
        for t in self.tp:
            cm[t["gt_class"], t["pred_class"]] += 1
        return cm

    def compute(self) -> Dict[str, float]:
        out = {"frames": self.n_frames}
        out.update(self._block(self.tp, self.fp, self.fn, ""))

        for lo, hi in RANGE_BANDS:
            sel = lambda xs: [x for x in xs if lo <= x["range"] < hi]
            out.update(self._block(sel(self.tp), sel(self.fp), sel(self.fn), f"_{lo}-{hi}m"))

        #matrice di confusione 3x3 sui soli TP: separa "non l'ho visto"
        #da "l'ho visto ma sbaglio colore"
        cm = self.confusion_matrix()
        for a, gt_name in enumerate(CLASS_NAMES):
            for b, pred_name in enumerate(CLASS_NAMES):
                out[f"cm_{gt_name}_as_{pred_name}"] = int(cm[a, b])

        #controllo: i conteggi per banda devono sommare al totale
        out["gt_total"] = len(self.tp) + len(self.fn)
        out["gt_accounted"] = sum(out[f"n_gt_{lo}-{hi}m"] for lo, hi in RANGE_BANDS)
        return out

    @staticmethod
    def _block(tp, fp, fn, suffix) -> Dict[str, float]:
        n_tp, n_fp, n_fn = len(tp), len(fp), len(fn)
        #bande senza GT restano nan,non 0

        nan = float("nan")
        p = n_tp / (n_tp + n_fp) if n_tp + n_fp else nan
        r = n_tp / (n_tp + n_fn) if n_tp + n_fn else nan
        f1 = 2 * p * r / (p + r) if (n_tp + n_fp and n_tp + n_fn and p + r) else nan
        out = {
            f"precision{suffix}": p,
            f"recall{suffix}": r,
            f"f1{suffix}": f1,
            f"n_tp{suffix}": n_tp, f"n_fp{suffix}": n_fp, f"n_fn{suffix}": n_fn,
            f"n_gt{suffix}": n_tp + n_fn,
        }

        if n_tp:
            d = np.array([x["dist"] for x in tp]) * 100
            out[f"loc_mean_cm{suffix}"] = float(d.mean())
            for q in LOC_PERCENTILES:
                out[f"loc_p{q}_cm{suffix}"] = float(np.percentile(d, q))

            #frazione oltre soglie fisiche fisse
            #i percentili alti sono vicini al raggio di matching (50 cm) quindi censurati: un cono spinto
            #oltre i 50 cm esce dai TP e i percentili dei rimanenti migliorano queste frazioni non hanno lo stesso artefatto e si leggono meglio

            for thr in LOC_THRESHOLDS_CM:
                out[f"loc_over_{thr}cm{suffix}"] = float((d > thr).mean())
            #il colore e' valutato solo sui TP
            out[f"color_acc{suffix}"] = float(
                np.mean([x["pred_class"] == x["gt_class"] for x in tp]))
        else:
            out[f"loc_mean_cm{suffix}"] = nan
            for q in LOC_PERCENTILES:
                out[f"loc_p{q}_cm{suffix}"] = nan
            for thr in LOC_THRESHOLDS_CM:
                out[f"loc_over_{thr}cm{suffix}"] = nan
            out[f"color_acc{suffix}"] = nan
        return out


def save_baseline(metrics: Dict, instance_hit: Dict[str, int], path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"metrics": metrics,
                                      "instance_hit": instance_hit}))


def compare_to_baseline(metrics: Dict, instance_hit: Dict[str, int], baseline_path: Path) -> Dict[str, float]:

    #differenze contro la baseline B da loggare a ogni epoca così una regressione si può notare all'epoca 5 invece che a fine corsa
    p = Path(baseline_path)
    if not p.exists():
        return {}
    base = json.loads(p.read_text())
    bm, bh = base["metrics"], base["instance_hit"]

    out = {}
    keys = ["recall", "precision", "f1", "n_fp"] + \
           [f"recall_{lo}-{hi}m" for lo, hi in RANGE_BANDS]
    for k in keys:
        if k in bm and k in metrics:
            out[f"d_{k}"] = metrics[k] - bm[k]
    if "frames" in bm and bm["frames"]:
        out["d_fp_per_frame"] = (metrics["n_fp"] / max(metrics["frames"], 1)
                                 - bm["n_fp"] / bm["frames"])

    #recupero condizionale: è la metrica che porta segnale quando il recall aggregato non può muoversi perchè la baseline è già a soffitto
    common = [k for k in bh if k in instance_hit]
    missed = [k for k in common if not bh[k]]
    hit = [k for k in common if bh[k]]
    out["recovered"] = sum(1 for k in missed if instance_hit[k])
    out["lost"] = sum(1 for k in hit if not instance_hit[k])
    out["missed_by_baseline"] = len(missed)
    out["recovery_rate"] = out["recovered"] / max(len(missed), 1)
    return out