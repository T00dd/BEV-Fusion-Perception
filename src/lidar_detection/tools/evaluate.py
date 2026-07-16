"""
evaluate.py -- valutazione GEOMETRICA del rilevatore di coni class-agnostic.

Fa TRE cose in un colpo:
  1) REPORT testuale: separa CAPACITA' della rete (coni rilevabili, ignore-region) da
     COPERTURA del sensore (quanti coni sono davvero rilevabili).
  2) FIGURE da tesi: heatmap BEV di recall, recall distanza x punti, curva di
     detectabilita', copertura-vs-capacita', scene BEV facile/difficile.
  3) PERSISTENZA: salva TUTTE le metriche in un file cumulativo (JSON + CSV riepilogo),
     etichettate per configurazione (--run_name) e soglia punti (--min_gt_points), cosi'
     confronti facilmente le varie run (es. agnostic_nogts vs gtsampling vs minpts2).

USO
---
  python evaluate.py --dump val_predictions.pkl --run_name agnostic_nogts --min_gt_points 3
  python evaluate.py --dump val_predictions.pkl --run_name agnostic_gtsampling --min_gt_points 3
  python evaluate.py --dump val_predictions.pkl --run_name agnostic_nogts --min_gt_points 2
  # -> tutte le metriche si accumulano in results/eval_results.json + eval_summary.csv
"""
import argparse
import csv
import datetime
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DIST_BANDS = [(0, 15), (15, 30), (30, 50)]
NPTS_BUCKETS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 10 ** 9)]
NPTS_LABELS = ['0', '1-2', '3-5', '6-10', '11-20', '21+']
PC_RANGE = (0, -25, 50, 25)          # xmin, ymin, xmax, ymax
BEV_EXTENT = (0, 50, -25, 25)        # x_min, x_max, y_min, y_max per le figure
MIN_POINTS_TRAIN = 3


def jsonable(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, float) and (x != x):
        return None
    return x


# --------------------------------------------------------------------------- #
# validita' + matching ignore-aware (per la CAPACITA')
# --------------------------------------------------------------------------- #
def in_range(xy, r):
    if len(xy) == 0:
        return np.zeros(0, bool)
    return (xy[:, 0] >= r[0]) & (xy[:, 0] < r[2]) & (xy[:, 1] >= r[1]) & (xy[:, 1] < r[3])


def valid_mask(gb, gnp, min_pts, r):
    if len(gb) == 0:
        return np.zeros(0, bool)
    return (gnp >= min_pts) & in_range(gb[:, :2], r)


def match_frame(pb, ps, gb, gvalid, dist_thr, score_thr=-1e9):
    P, M = len(pb), len(gb)
    is_tp = np.zeros(P, bool); ignore = np.zeros(P, bool)
    considered = ps >= score_thr
    taken = np.zeros(M, bool)
    order = np.where(considered)[0]; order = order[np.argsort(-ps[order])]
    for pi in order:
        if M == 0:
            continue
        d = np.linalg.norm(gb[:, :2] - pb[pi, :2], axis=1); d[taken] = np.inf
        j = int(np.argmin(d))
        if d[j] <= dist_thr:
            taken[j] = True
            (is_tp if gvalid[j] else ignore)[pi] = True
    return is_tp, ignore, considered, taken


def average_precision(scores, is_tp, n_gt):
    if n_gt == 0:
        return float('nan')
    if len(scores) == 0:
        return 0.0
    o = np.argsort(-scores); tp = is_tp[o].astype(float); fp = 1.0 - tp
    tpc, fpc = np.cumsum(tp), np.cumsum(fp)
    rec = tpc / n_gt; prec = tpc / np.maximum(tpc + fpc, 1e-9)
    mrec = np.concatenate([[0], rec, [rec[-1]]]); mpre = np.concatenate([[1], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def band_mask(xy, band):
    if band is None:
        return np.ones(len(xy), bool)
    a, b = band; d = np.hypot(xy[:, 0], xy[:, 1]) if len(xy) else np.zeros(0)
    return (d >= a) & (d < b)


def ap(frames, thr, band=None):
    sc, tp, ngt = [], [], 0
    for fr in frames:
        pm = band_mask(fr['pred_boxes'][:, :2], band); gm = band_mask(fr['gt_boxes'][:, :2], band)
        is_tp, ign, cons, taken = match_frame(fr['pred_boxes'][pm], fr['pred_scores'][pm],
                                              fr['gt_boxes'][gm], fr['gt_valid'][gm], thr)
        sel = cons & ~ign
        sc.append(fr['pred_scores'][pm][sel]); tp.append(is_tp[sel]); ngt += int(fr['gt_valid'][gm].sum())
    sc = np.concatenate(sc) if sc else np.zeros(0); tp = np.concatenate(tp) if tp else np.zeros(0, bool)
    return average_precision(sc, tp, ngt)


def pr_curve(frames, thr):
    sc, tp, ngt = [], [], 0
    for fr in frames:
        is_tp, ign, cons, taken = match_frame(fr['pred_boxes'], fr['pred_scores'],
                                              fr['gt_boxes'], fr['gt_valid'], thr)
        sel = cons & ~ign
        sc.append(fr['pred_scores'][sel]); tp.append(is_tp[sel]); ngt += int(fr['gt_valid'].sum())
    sc = np.concatenate(sc); tp = np.concatenate(tp)
    o = np.argsort(-sc); tpc = np.cumsum(tp[o]); fpc = np.cumsum(~tp[o])
    rec = tpc / max(ngt, 1); prec = tpc / np.maximum(tpc + fpc, 1e-9)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return sc[o], rec, prec, f1, ngt


def collect_operating(frames, thr, score_thr):
    tp, fp, missed, per_frame, n_gt = [], [], [], [], 0
    for fr in frames:
        gb, gnp, gv = fr['gt_boxes'], fr['gt_npts'], fr['gt_valid']
        is_tp, ign, cons, taken = match_frame(fr['pred_boxes'], fr['pred_scores'], gb, gv, thr, score_thr)
        pb = fr['pred_boxes']
        for pi in np.where(cons & ~is_tp & ~ign)[0]:
            fp.append({'score': float(fr['pred_scores'][pi]), 'dist': float(np.hypot(*pb[pi, :2]))})
        n_gt += int(gv.sum())
        for j in range(len(gb)):
            if gv[j] and not taken[j]:
                missed.append({'dist': float(np.hypot(*gb[j, :2])), 'npts': int(gnp[j])})
        order = np.where(cons)[0]; order = order[np.argsort(-fr['pred_scores'][order])]
        takg = np.zeros(len(gb), bool)
        for pi in order:
            if len(gb) == 0:
                break
            d = np.linalg.norm(gb[:, :2] - pb[pi, :2], axis=1); d[takg] = np.inf
            j = int(np.argmin(d))
            if d[j] <= thr:
                takg[j] = True
                if gv[j]:
                    g = gb[j]; e = pb[pi, :2] - g[:2]; u = g[:2] / max(np.linalg.norm(g[:2]), 1e-9)
                    tp.append({'dist': float(np.hypot(*g[:2])), 'npts': int(gnp[j]),
                               'err': float(np.linalg.norm(e)), 'radial': float(e @ u),
                               'lateral': float(e[0] * u[1] - e[1] * u[0]),
                               'z_err': float(pb[pi, 2] - g[2])})
        nv = int(gv.sum()); rec = (takg & gv).sum() / nv if nv else float('nan')
        per_frame.append({'frame_id': fr['frame_id'], 'recall': rec, 'n_gt': nv,
                          'fp': int((cons & ~is_tp & ~ign).sum())})
    return dict(tp=tp, fp=fp, missed=missed, n_gt=n_gt, per_frame=per_frame)


# --------------------------------------------------------------------------- #
# matching FULL-GT (senza ignore) per le FIGURE (donut / copertura / detectability)
# --------------------------------------------------------------------------- #
def match_full(frames, dist_thr, score_thr):
    gx, gy, gn, gdet = [], [], [], []
    for fr in frames:
        pb, ps, gb, gnp = fr['pred_boxes'], fr['pred_scores'], fr['gt_boxes'], fr['gt_npts']
        keep = ps >= score_thr; pbb = pb[keep]; order = np.argsort(-ps[keep]); pbb = pbb[order]
        taken = np.zeros(len(gb), bool)
        for b in pbb:
            if len(gb) == 0:
                break
            d = np.linalg.norm(gb[:, :2] - b[:2], axis=1); d[taken] = np.inf
            j = int(np.argmin(d))
            if d[j] <= dist_thr:
                taken[j] = True
        for j in range(len(gb)):
            gx.append(gb[j, 0]); gy.append(gb[j, 1]); gn.append(int(gnp[j])); gdet.append(bool(taken[j]))
    return np.array(gx), np.array(gy), np.array(gn), np.array(gdet, bool)


def recall_by(vals_det, vals_all, buckets, labels):
    rows = []
    for lab, (lo, hi) in zip(labels, buckets):
        det = int(((vals_det >= lo) & (vals_det <= hi)).sum())
        tot = int(((vals_all >= lo) & (vals_all <= hi)).sum())
        rows.append([lab, det, tot, det / tot if tot else float('nan')])
    return rows


# --------------------------------------------------------------------------- #
# FIGURE
# --------------------------------------------------------------------------- #
def fig_bev_recall(gx, gy, gdet, out, cell=2.5, min_count=8):
    xe = np.arange(BEV_EXTENT[0], BEV_EXTENT[1] + cell, cell)
    ye = np.arange(BEV_EXTENT[2], BEV_EXTENT[3] + cell, cell)
    tot, _, _ = np.histogram2d(gx, gy, bins=[xe, ye])
    det, _, _ = np.histogram2d(gx[gdet], gy[gdet], bins=[xe, ye])
    with np.errstate(invalid='ignore'):
        rec = np.where(tot >= min_count, det / np.maximum(tot, 1), np.nan)
    ext = [BEV_EXTENT[2], BEV_EXTENT[3], BEV_EXTENT[0], BEV_EXTENT[1]]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    im = ax[0].imshow(rec, origin='lower', extent=ext, aspect='equal', cmap='RdYlGn', vmin=0, vmax=1)
    ax[0].scatter([0], [0], marker='^', s=120, color='k', zorder=5, label='LiDAR')
    ax[0].set_title('Recall per cella (vista dall\'alto)', fontweight='bold')
    ax[0].set_xlabel('y laterale [m]'); ax[0].set_ylabel('x avanti [m]'); ax[0].legend(loc='upper right')
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04, label='recall')
    im2 = ax[1].imshow(np.log10(tot + 1), origin='lower', extent=ext, aspect='equal', cmap='viridis')
    ax[1].scatter([0], [0], marker='^', s=120, color='w', zorder=5)
    ax[1].set_title('Densita\' di coni GT (log)', fontweight='bold')
    ax[1].set_xlabel('y laterale [m]'); ax[1].set_ylabel('x avanti [m]')
    fig.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04, label='log10(conteggio+1)')
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_dist_npts(gx, gy, gn, gdet, out):
    dist = np.hypot(gx, gy); Rm = np.full((len(NPTS_BUCKETS), len(DIST_BANDS)), np.nan); Nm = np.zeros_like(Rm)
    for i, (lo, hi) in enumerate(NPTS_BUCKETS):
        for j, (a, b) in enumerate(DIST_BANDS):
            m = (gn >= lo) & (gn <= hi) & (dist >= a) & (dist < b)
            if m.sum():
                Rm[i, j] = gdet[m].mean(); Nm[i, j] = m.sum()
    fig, ax = plt.subplots(figsize=(6.5, 5.5)); im = ax.imshow(Rm, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(DIST_BANDS))); ax.set_xticklabels([f'{a}-{b}m' for a, b in DIST_BANDS])
    ax.set_yticks(range(len(NPTS_LABELS))); ax.set_yticklabels(NPTS_LABELS)
    ax.set_xlabel('distanza'); ax.set_ylabel('punti sul cono')
    ax.set_title('Recall per distanza x densita\' di punti', fontweight='bold')
    for i in range(Rm.shape[0]):
        for j in range(Rm.shape[1]):
            if not np.isnan(Rm[i, j]):
                ax.text(j, i, f'{Rm[i,j]:.2f}\nn={int(Nm[i,j])}', ha='center', va='center',
                        fontsize=8, color='black' if Rm[i, j] > .4 else 'white')
    fig.colorbar(im, fraction=0.046, pad=0.04, label='recall'); fig.tight_layout()
    fig.savefig(out, dpi=140); plt.close(fig)


def fig_detectability(gn, gdet, out):
    edges = [0, 1, 3, 5, 7, 10, 15, 20, 30, 10 ** 9]
    labels = ['0', '1-2', '3-4', '5-6', '7-9', '10-14', '15-19', '20-29', '30+']
    rec, cnt = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (gn >= lo) & (gn < hi); rec.append(gdet[m].mean() if m.sum() else np.nan); cnt.append(int(m.sum()))
    x = np.arange(len(labels)); fig, ax = plt.subplots(figsize=(7.5, 4.5)); ax2 = ax.twinx()
    ax2.bar(x, cnt, color='lightgray', alpha=.6, width=.7); ax.plot(x, rec, 'o-', color='tab:blue', lw=2, ms=6)
    ax.axvline(1.5, color='red', ls='--', lw=1.5)
    ax.text(1.55, 0.05, f'soglia training\n(>= {MIN_POINTS_TRAIN} punti)', color='red', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_xlabel('numero di punti sul cono')
    ax.set_ylabel('recall'); ax.set_ylim(0, 1.03); ax2.set_ylabel('n. coni (barre)')
    ax.set_title('Soglia di detectabilita\' geometrica', fontweight='bold')
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_coverage_capability(gx, gy, gn, gdet, out):
    dist = np.hypot(gx, gy); bands = [f'{a}-{b}m' for a, b in DIST_BANDS]
    f0, f12, f3, r3 = [], [], [], []
    for a, b in DIST_BANDS:
        m = (dist >= a) & (dist < b); tot = m.sum()
        f0.append(((gn == 0) & m).sum() / tot); f12.append(((gn >= 1) & (gn <= 2) & m).sum() / tot)
        f3.append(((gn >= 3) & m).sum() / tot); m3 = m & (gn >= 3)
        r3.append(gdet[m3].mean() if m3.sum() else np.nan)
    x = np.arange(len(bands)); fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(x, f0, color='tab:red', label='0 punti (invisibili)')
    ax.bar(x, f12, bottom=f0, color='orange', label='1-2 punti (sotto soglia)')
    ax.bar(x, f3, bottom=np.array(f0) + np.array(f12), color='tab:green', label='>=3 punti (rilevabili)')
    for i, r in enumerate(r3):
        ax.text(i, 1.02, f'recall>=3pt\n{r:.3f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(bands); ax.set_ylim(0, 1.15); ax.set_ylabel('frazione dei coni')
    ax.set_title('Copertura del sensore vs capacita\' della rete', fontweight='bold')
    ax.legend(loc='lower center', ncol=3, fontsize=8, framealpha=.9); fig.tight_layout()
    fig.savefig(out, dpi=140); plt.close(fig)


def fig_scene(fr, dist_thr, score_thr, out):
    pb, ps, gb = fr['pred_boxes'], fr['pred_scores'], fr['gt_boxes']
    keep = ps >= score_thr; pbb = pb[keep]; order = np.argsort(-ps[keep]); pbb = pbb[order]
    taken = np.zeros(len(gb), bool)
    for b in pbb:
        if len(gb) == 0:
            break
        d = np.linalg.norm(gb[:, :2] - b[:2], axis=1); d[taken] = np.inf
        j = int(np.argmin(d))
        if d[j] <= dist_thr:
            taken[j] = True
    fig, ax = plt.subplots(figsize=(6.5, 7))
    if len(gb):
        ax.scatter(gb[taken, 1], gb[taken, 0], marker='o', s=70, facecolors='none',
                   edgecolors='tab:green', linewidths=2, label='GT rilevati')
        ax.scatter(gb[~taken, 1], gb[~taken, 0], marker='o', s=70, facecolors='none',
                   edgecolors='orange', linewidths=2, label='GT persi')
    if len(pbb):
        ax.scatter(pbb[:, 1], pbb[:, 0], marker='x', s=40, color='tab:red', label='predetti')
    ax.scatter([0], [0], marker='^', s=140, color='k', label='LiDAR')
    for r in (15, 30, 50):
        th = np.linspace(-np.pi / 2, np.pi / 2, 100)
        ax.plot(r * np.sin(th), r * np.cos(th), color='gray', lw=.5, ls=':')
    ax.set_xlim(BEV_EXTENT[2], BEV_EXTENT[3]); ax.set_ylim(BEV_EXTENT[0], BEV_EXTENT[1]); ax.set_aspect('equal')
    ax.set_xlabel('y laterale [m]'); ax.set_ylabel('x avanti [m]')
    rec = taken.mean() if len(gb) else float('nan')
    ax.set_title(f"{fr['frame_id']}  |  recall={rec:.2f}  (GT={len(gb)})", fontweight='bold')
    ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(out, dpi=140); plt.close(fig)


def bar(labels, vals, title, ylab, path, counts=None):
    fig, ax = plt.subplots(figsize=(6.5, 4)); xs = np.arange(len(labels))
    ax.bar(xs, vals, color='tab:blue', alpha=.85); ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel(ylab); ax.set_ylim(0, 1.05); ax.set_title(title, fontweight='bold'); ax.grid(alpha=.2, axis='y')
    if counts:
        for x, v, c in zip(xs, vals, counts):
            ax.text(x, (v if v == v else 0) + .02, f'n={c}', ha='center', fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_pr(rec, prec, path):
    fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(rec, prec, lw=2, color='tab:blue')
    ax.set_xlabel('recall'); ax.set_ylabel('precision'); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title('Precision-Recall (coni rilevabili)', fontweight='bold'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# CONFIDENZA & CALIBRAZIONE
# Sfrutta il fatto che le predizioni sono raccolte con score_thresh basso (0.01):
# abbiamo l'intero spettro di score per valutare quanto lo score sia affidabile.
# --------------------------------------------------------------------------- #
def _rankdata(a):
    """Ranghi medi (gestione ties) senza scipy."""
    order = np.argsort(a, kind='mergesort'); sa = a[order]
    ranks = np.empty(len(a), float); i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def confidence_metrics(frames, dist_thr, n_bins=10):
    """Match su TUTTE le predizioni (ignore-aware). Ritorna separazione TP/FP,
    reliability/ECE e score mediano per difficolta' (distanza / densita' punti)."""
    tp_s, fp_s, tp_rec = [], [], []   # tp_rec: (score, dist, npts)
    for fr in frames:
        pb, ps, gb, gnp, gv = fr['pred_boxes'], fr['pred_scores'], fr['gt_boxes'], fr['gt_npts'], fr['gt_valid']
        order = np.argsort(-ps); taken = np.zeros(len(gb), bool)
        for pi in order:
            if len(gb) == 0:
                fp_s.append(float(ps[pi])); continue
            d = np.linalg.norm(gb[:, :2] - pb[pi, :2], axis=1); d[taken] = np.inf
            j = int(np.argmin(d))
            if d[j] <= dist_thr:
                taken[j] = True
                if gv[j]:
                    tp_s.append(float(ps[pi]))
                    tp_rec.append((float(ps[pi]), float(np.hypot(*gb[j, :2])), int(gnp[j])))
                # matched a GT non valido -> ignore
            else:
                fp_s.append(float(ps[pi]))
    tp = np.array(tp_s); fp = np.array(fp_s)

    # AUROC: P(score TP > score FP). 0.5 = score inutile, 1 = separazione perfetta.
    auroc = None
    if len(tp) and len(fp):
        r = _rankdata(np.concatenate([tp, fp]))
        auroc = float((r[:len(tp)].sum() - len(tp) * (len(tp) + 1) / 2) / (len(tp) * len(fp)))

    # reliability diagram + ECE
    y = np.concatenate([np.ones(len(tp)), np.zeros(len(fp))])
    conf = np.concatenate([tp, fp]); N = len(conf)
    edges = np.linspace(0, 1, n_bins + 1); rel = []; ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi) if hi >= 1 else (conf >= lo) & (conf < hi)
        c = int(m.sum())
        if c:
            acc = float(y[m].mean()); cf = float(conf[m].mean())
            rel.append({'bin_lo': float(lo), 'bin_hi': float(hi), 'count': c, 'confidence': cf, 'accuracy': acc})
            ece += c / max(N, 1) * abs(acc - cf)

    # score mediano per difficolta' (solo TP)
    recs = np.array(tp_rec) if tp_rec else np.zeros((0, 3))

    def med_dist():
        out = {}
        for a, b in DIST_BANDS:
            m = (recs[:, 1] >= a) & (recs[:, 1] < b) if len(recs) else np.zeros(0, bool)
            out[f'{a}-{b}m'] = float(np.median(recs[m, 0])) if len(recs) and m.any() else None
        return out

    def med_npts():
        out = {}
        for lab, (lo, hi) in zip(NPTS_LABELS, NPTS_BUCKETS):
            m = (recs[:, 2] >= lo) & (recs[:, 2] <= hi) if len(recs) else np.zeros(0, bool)
            out[lab] = float(np.median(recs[m, 0])) if len(recs) and m.any() else None
        return out

    return {'tp_score_median': float(np.median(tp)) if len(tp) else None,
            'fp_score_median': float(np.median(fp)) if len(fp) else None,
            'auroc': auroc, 'ece': float(ece), 'reliability': rel,
            'score_by_distance': med_dist(), 'score_by_npts': med_npts(),
            '_tp': tp, '_fp': fp}


def plot_score_calib(tp, fp, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if len(tp):
        ax.hist(tp, bins=40, alpha=.6, density=True, color='tab:green', label=f'TP (mediana {np.median(tp):.2f})')
    if len(fp):
        ax.hist(fp, bins=40, alpha=.6, density=True, color='tab:red', label=f'FP (mediana {np.median(fp):.2f})')
    ax.set_xlabel('score'); ax.set_ylabel('densita\''); ax.legend()
    ax.set_title('Separazione dello score: TP vs FP', fontweight='bold'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_reliability(rel, ece, path):
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='calibrazione ideale')
    if rel:
        ax.plot([r['confidence'] for r in rel], [r['accuracy'] for r in rel], 'o-',
                color='tab:blue', lw=2, label='osservata')
    ax.set_xlabel('confidenza media (score)'); ax.set_ylabel('accuratezza (frazione TP)')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f'Reliability diagram (ECE={ece:.3f})', fontweight='bold'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# persistenza cumulativa
# --------------------------------------------------------------------------- #
SUMMARY_COLS = ['run_name', 'min_gt_points', 'dist_thresh', 'op_score', 'n_frames',
                'coverage_pct', 'ap_global', 'ap_0_15m', 'ap_15_30m', 'ap_30_50m',
                'recall', 'precision', 'fp_per_frame', 'loc_err_median_cm', 'loc_err_p90_cm',
                'score_auroc', 'ece', 'tp_score_med', 'fp_score_med']


def _num(x, nd):
    return round(x, nd) if (x is not None and x == x) else ''


def save_results(results_file, key, metrics):
    results_file = Path(results_file); results_file.parent.mkdir(parents=True, exist_ok=True)
    db = json.load(open(results_file)) if results_file.exists() else {}
    db[key] = metrics
    json.dump(jsonable(db), open(results_file, 'w'), indent=2, ensure_ascii=False)
    # CSV riepilogo (una riga per configurazione)
    csv_path = results_file.with_name('eval_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS); w.writeheader()
        for k, m in db.items():
            h = m['headline']
            w.writerow({
                'run_name': m['run_name'], 'min_gt_points': m['min_gt_points'],
                'dist_thresh': m['dist_thresh'], 'op_score': round(m['operating']['score'], 3),
                'n_frames': m['n_frames'], 'coverage_pct': round(m['coverage']['valid_pct'], 1),
                'ap_global': round(h['ap_global'], 3),
                'ap_0_15m': round(m['ap_by_band']['0-15m'], 3),
                'ap_15_30m': round(m['ap_by_band']['15-30m'], 3),
                'ap_30_50m': round(m['ap_by_band']['30-50m'], 3),
                'recall': round(h['recall'], 3), 'precision': round(h['precision'], 3),
                'fp_per_frame': round(h['fp_per_frame'], 3),
                'loc_err_median_cm': _num(h['loc_err_median_cm'], 2),
                'loc_err_p90_cm': _num(h['loc_err_p90_cm'], 2),
                'score_auroc': _num(m.get('confidence', {}).get('auroc'), 3),
                'ece': _num(m.get('confidence', {}).get('ece'), 3),
                'tp_score_med': _num(m.get('confidence', {}).get('tp_score_median'), 3),
                'fp_score_med': _num(m.get('confidence', {}).get('fp_score_median'), 3),
            })
    return results_file, csv_path


# --------------------------------------------------------------------------- #
def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--dump', required=True)
    ap_.add_argument('--run_name', default='run', help='nome configurazione, es. agnostic_nogts')
    ap_.add_argument('--out_dir', type=Path, default=None, help='default: eval_report/<run_name>_minpts<N>')
    ap_.add_argument('--results_file', default='results/eval_results.json',
                     help='file cumulativo JSON (+ eval_summary.csv accanto)')
    ap_.add_argument('--dist_thresh', type=float, default=0.5)
    ap_.add_argument('--min_gt_points', type=int, default=3)
    ap_.add_argument('--score_thresh', type=float, default=None)
    args = ap_.parse_args()
    T = args.dist_thresh
    key = f'{args.run_name}@minpts{args.min_gt_points}'
    out_dir = args.out_dir or Path('eval_report') / f'{args.run_name}_minpts{args.min_gt_points}'
    out_dir.mkdir(parents=True, exist_ok=True)

    data = pickle.load(open(args.dump, 'rb'))
    frames = data['frames']
    for fr in frames:
        fr['pred_boxes'] = np.asarray(fr['pred_boxes'], float).reshape(-1, 7)
        fr['gt_boxes'] = np.asarray(fr['gt_boxes'], float).reshape(-1, 7)
        fr['pred_scores'] = np.asarray(fr['pred_scores'], float)
        fr['gt_npts'] = np.asarray(fr['gt_npts'], int)
        fr['gt_valid'] = valid_mask(fr['gt_boxes'], fr['gt_npts'], args.min_gt_points, PC_RANGE)

    # coverage
    all_np = np.concatenate([f['gt_npts'] for f in frames]); all_xy = np.concatenate([f['gt_boxes'][:, :2] for f in frames])
    all_valid = np.concatenate([f['gt_valid'] for f in frames]); inr = in_range(all_xy, PC_RANGE)
    tot, n_inr, n_valid = len(all_np), int(inr.sum()), int(all_valid.sum())
    dist_all = np.hypot(all_xy[:, 0], all_xy[:, 1])
    cov_by_band = {}
    print(f'\n=== {len(frames)} frame | {key} | tolleranza {T} m ===')
    print(f'[0]  GT totali {tot} | in-range {n_inr} | RILEVABILI {n_valid} ({100*n_valid/max(tot,1):.1f}%)')
    print('[0b] COPERTURA sensore per fascia:')
    for a, b in DIST_BANDS:
        m = (dist_all >= a) & (dist_all < b) & inr
        cov = all_valid[m].sum() / m.sum() if m.sum() else float('nan')
        cov_by_band[f'{a}-{b}m'] = cov
        print(f'      {a:>2}-{b:<3}m: {int(all_valid[m].sum())}/{int(m.sum())}  copertura={cov:.3f}')

    # AP capacita'
    ap_glob = ap(frames, T); ap_band = {f'{a}-{b}m': ap(frames, T, (a, b)) for a, b in DIST_BANDS}
    print(f'\n[1] AP capacita\' globale {ap_glob:.3f} | ' + ' '.join(f'{k}={v:.3f}' for k, v in ap_band.items()))

    # punto operativo
    s, rec, prec, f1, ngt = pr_curve(frames, T)
    op = args.score_thresh if args.score_thresh is not None else (float(s[int(np.argmax(f1))]) if len(f1) else 0.15)
    print(f'\n[2] Punto operativo score>={op:.3f}')
    sweep = []
    for th in sorted(set(np.round(np.linspace(0.05, 0.6, 12), 3)) | {round(op, 3)}):
        k = min(np.searchsorted(-s, -th), len(rec) - 1)
        sweep.append({'thr': float(th), 'precision': float(prec[k]), 'recall': float(rec[k]), 'f1': float(f1[k])})
        print(f'      {th:.3f}  P={prec[k]:.3f} R={rec[k]:.3f} F1={f1[k]:.3f}' + ('  <-- op' if abs(th-op) < 1e-6 else ''))

    R = collect_operating(frames, T, op)
    tp, fp, missed = R['tp'], R['fp'], R['missed']
    recall = len(tp) / max(R['n_gt'], 1); precision = len(tp) / max(len(tp) + len(fp), 1)
    fp_pf = len(fp) / max(len(frames), 1)
    print(f"      TP={len(tp)} FP={len(fp)} MISSED={len(missed)} recall={recall:.3f} "
          f"precisione={precision:.3f} FP/frame={fp_pf:.2f}")

    # figure (match full-gt al punto operativo)
    gx, gy, gn, gdet = match_full(frames, T, op)
    fig_bev_recall(gx, gy, gdet, out_dir / 'bev_recall_heatmap.png')
    fig_dist_npts(gx, gy, gn, gdet, out_dir / 'recall_dist_npts.png')
    fig_detectability(gn, gdet, out_dir / 'detectability_curve.png')
    fig_coverage_capability(gx, gy, gn, gdet, out_dir / 'coverage_vs_capability.png')
    plot_pr(rec, prec, out_dir / 'precision_recall.png')

    # recall per distanza (capacita') + detectability (full)
    td = np.array([t['dist'] for t in tp]); md = np.array([m['dist'] for m in missed])
    rows_dist = recall_by(td, np.concatenate([td, md]) if len(td) + len(md) else np.zeros(0),
                          DIST_BANDS, [f'{a}-{b}m' for a, b in DIST_BANDS])
    det_n = gn[gdet]; rows_npts = recall_by(det_n, gn, NPTS_BUCKETS, NPTS_LABELS)
    bar([r[0] for r in rows_dist], [r[3] for r in rows_dist], 'Recall per distanza (rilevabili)',
        'recall', out_dir / 'recall_by_distance.png', counts=[r[2] for r in rows_dist])
    bar(NPTS_LABELS, [r[3] for r in rows_npts], 'Detectability per densita\' punti (tutti i GT)',
        'recall', out_dir / 'recall_by_npoints.png', counts=[r[2] for r in rows_npts])

    # errore di localizzazione
    loc = {}
    if tp:
        err = np.array([t['err'] for t in tp]); rad = np.array([t['radial'] for t in tp])
        lat = np.array([t['lateral'] for t in tp]); ze = np.array([t['z_err'] for t in tp])
        loc = {'median_cm': float(np.median(err) * 100), 'p90_cm': float(np.percentile(err, 90) * 100),
               'radial_med_cm': float(np.median(np.abs(rad)) * 100),
               'lateral_med_cm': float(np.median(np.abs(lat)) * 100),
               'z_med_cm': float(np.median(np.abs(ze)) * 100),
               'by_band_median_cm': {f'{a}-{b}m': (float(np.median(err[(td >= a) & (td < b)]) * 100)
                                                   if ((td >= a) & (td < b)).any() else None) for a, b in DIST_BANDS}}
        print(f"\n[6] loc_err mediana {loc['median_cm']:.1f} cm | p90 {loc['p90_cm']:.1f} cm | "
              f"radiale {loc['radial_med_cm']:.1f} | laterale {loc['lateral_med_cm']:.1f} | z {loc['z_med_cm']:.1f}")

    fp_d = np.array([f['dist'] for f in fp]) if fp else np.zeros(0)
    fp_by_band = {f'{a}-{b}m': int(((fp_d >= a) & (fp_d < b)).sum()) for a, b in DIST_BANDS}
    worst = sorted([p for p in R['per_frame'] if p['n_gt'] >= 5], key=lambda p: p['recall'])[:10]

    # ---------- confidenza & calibrazione ----------
    conf = confidence_metrics(frames, T)
    plot_score_calib(conf.pop('_tp'), conf.pop('_fp'), out_dir / 'score_calibration.png')
    plot_reliability(conf['reliability'], conf['ece'], out_dir / 'reliability_diagram.png')
    print('\n[10] Confidenza dello score')
    print(f"      TP mediano {conf['tp_score_median']} | FP mediano {conf['fp_score_median']} | "
          f"AUROC {conf['auroc']:.3f} | ECE {conf['ece']:.3f}" if conf['auroc'] is not None else
          f"      ECE {conf['ece']:.3f}")
    print(f"      score mediano TP per distanza: {conf['score_by_distance']}")

    # ---------- costruzione metriche + salvataggio ----------
    metrics = {
        'run_name': args.run_name, 'min_gt_points': args.min_gt_points, 'dist_thresh': T,
        'n_frames': len(frames), 'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'coverage': {'total': tot, 'in_range': n_inr, 'valid': n_valid,
                     'valid_pct': 100 * n_valid / max(tot, 1), 'by_band': cov_by_band},
        'ap_global': ap_glob, 'ap_by_band': ap_band,
        'operating': {'score': op, 'sweep': sweep},
        'headline': {'ap_global': ap_glob, 'recall': recall, 'precision': precision,
                     'tp': len(tp), 'fp': len(fp), 'missed': len(missed), 'fp_per_frame': fp_pf,
                     'loc_err_median_cm': loc.get('median_cm'), 'loc_err_p90_cm': loc.get('p90_cm')},
        'recall_by_distance': rows_dist, 'detectability_by_npts': rows_npts,
        'loc_error': loc, 'fp_by_band': fp_by_band, 'worst_frames': worst,
        'confidence': conf,
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(jsonable(metrics), f, indent=2, ensure_ascii=False)
    rf, cf = save_results(args.results_file, key, metrics)

    print('\n--- RIASSUNTO (i due numeri) ---')
    print(f"  CAPACITA' : recall={recall:.3f} AP={ap_glob:.3f} precisione={precision:.3f} "
          f"loc={loc.get('median_cm', float('nan')):.1f}cm" if tp else f"  CAPACITA': recall={recall:.3f}")
    print(f"  COPERTURA : {n_valid}/{tot} = {100*n_valid/max(tot,1):.1f}% rilevabili")
    print(f"\nFigure+metrics: {out_dir}/ | cumulativo: {rf} | tabella: {cf}")


if __name__ == '__main__':
    main()