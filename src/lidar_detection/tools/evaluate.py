"""
evaluate.py  --  Valutazione COMPLETA a partire dal dump di run_inference.py.

Posizione consigliata:  src/lidar_detection/tools/evaluate.py

Ogni metrica risponde a una domanda precisa:
  1. AP per classe e distanza  -> quanto bene rileva OGNI colore, e DOVE?
  2. AP class-agnostic         -> quanto bene LOCALIZZA (ignorando il colore)?
  3. Matrice di confusione     -> come si comporta sul COLORE? confonde le classi?
  4. Curve precision-recall    -> qual e' il compromesso a ogni soglia di confidenza?
  5. Errore di localizzazione  -> quanto sono PRECISI i centri (istogramma)?
  6. Sweep della tolleranza    -> quanto e' precisa la localizzazione a soglie strette?
  7. Recall vs punti minimi    -> il limite di detectabilita' per densita' di punti.

Il calcolo dell'AP viene da cone_eval (gia' testato); qui sopra aggiungiamo il resto.

USO
---
  python src/lidar_detection/tools/evaluate.py --dump val_predictions.pkl --out_dir eval_report
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import cone_eval as E   # AP core (match_frame, average_precision, evaluate, print_table)


# ------------------------------------------------------------------ #
#  2. AP class-agnostic: localizzazione a prescindere dal colore       #
# ------------------------------------------------------------------ #
def class_agnostic_ap(frames, thr):
    scores, tps, n_gt = [], [], 0
    for fr in frames:
        is_tp, ng = E.match_frame(fr['pred_xy'], fr['pred_scores'], fr['gt_xy'], thr)
        n_gt += ng
        scores.append(np.sort(fr['pred_scores'])[::-1] if len(fr['pred_scores']) else fr['pred_scores'])
        tps.append(is_tp)
    scores = np.concatenate(scores) if scores else np.zeros(0)
    tps = np.concatenate(tps) if tps else np.zeros(0, bool)
    return E.average_precision(scores, tps, n_gt)


# ------------------------------------------------------------------ #
#  3. Matrice di confusione (abbinamento a prescindere dal colore)     #
# ------------------------------------------------------------------ #
def confusion_matrix(frames, class_names, thr):
    K = len(class_names); idx = {c: i for i, c in enumerate(class_names)}
    cm = np.zeros((K, K), int); missed = np.zeros(K, int); false = np.zeros(K, int)
    for fr in frames:
        order = np.argsort(-fr['pred_scores'])
        p_xy = fr['pred_xy'][order]; p_lab = fr['pred_labels'][order]
        g_xy = fr['gt_xy']; g_lab = fr['gt_labels']
        taken = np.zeros(len(g_xy), bool)
        for p, pl in zip(p_xy, p_lab):
            if pl not in idx:
                continue
            if len(g_xy) == 0:
                false[idx[pl]] += 1; continue
            d = np.linalg.norm(g_xy - p, axis=1); d[taken] = np.inf
            j = int(np.argmin(d))
            if d[j] <= thr:
                taken[j] = True; cm[idx[g_lab[j]]][idx[pl]] += 1
            else:
                false[idx[pl]] += 1
        for j, gl in enumerate(g_lab):
            if not taken[j] and gl in idx:
                missed[idx[gl]] += 1
    return cm, missed, false


# ------------------------------------------------------------------ #
#  4. Curva precision-recall per classe                                #
# ------------------------------------------------------------------ #
def pr_curve(frames, cls, thr):
    scores, tps, n_gt = [], [], 0
    for fr in frames:
        pm = fr['pred_labels'] == cls; gm = fr['gt_labels'] == cls
        p_xy, p_sc, g_xy = fr['pred_xy'][pm], fr['pred_scores'][pm], fr['gt_xy'][gm]
        is_tp, ng = E.match_frame(p_xy, p_sc, g_xy, thr); n_gt += ng
        scores.append(np.sort(p_sc)[::-1] if len(p_sc) else p_sc); tps.append(is_tp)
    scores = np.concatenate(scores); tps = np.concatenate(tps)
    o = np.argsort(-scores); tps = tps[o]
    tp_c = np.cumsum(tps); fp_c = np.cumsum(~tps)
    recall = tp_c / max(n_gt, 1); precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    return recall, precision


# ------------------------------------------------------------------ #
#  5. Errore di localizzazione (distanza dei veri positivi dal GT)     #
# ------------------------------------------------------------------ #
def localization_errors(frames, thr):
    errs = []
    for fr in frames:
        if len(fr['pred_xy']) == 0 or len(fr['gt_xy']) == 0:
            continue
        order = np.argsort(-fr['pred_scores']); p_xy = fr['pred_xy'][order]
        g_xy = fr['gt_xy']; taken = np.zeros(len(g_xy), bool)
        for p in p_xy:
            d = np.linalg.norm(g_xy - p, axis=1); d[taken] = np.inf
            j = int(np.argmin(d))
            if d[j] <= thr:
                taken[j] = True; errs.append(d[j])
    return np.array(errs)


# ------------------------------------------------------------------ #
#  6. Sweep della tolleranza                                           #
# ------------------------------------------------------------------ #
def tolerance_sweep(frames, class_names, thrs):
    out = {c: [] for c in class_names}; out['class-agnostic'] = []
    for t in thrs:
        r = E.evaluate(frames, class_names, dist_thresh=t)
        for c in class_names:
            out[c].append(r[('all', c)])
        out['class-agnostic'].append(class_agnostic_ap(frames, t))
    return out


# ------------------------------------------------------------------ #
#  Stampa e grafici                                                    #
# ------------------------------------------------------------------ #
COLORS = {'blue': 'tab:blue', 'yellow': 'gold', 'orange': 'darkorange'}

def plot_confusion(cm, missed, false, class_names, path):
    K = len(class_names)
    tot = cm.sum(1, keepdims=True); rown = cm / np.maximum(tot, 1)
    fig, ax = plt.subplots(figsize=(1.6*K+1.5, 1.4*K+1))
    im = ax.imshow(rown, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels(class_names)
    ax.set_yticks(range(K)); ax.set_yticklabels(class_names)
    ax.set_xlabel('classe PREDETTA'); ax.set_ylabel('classe VERA')
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f'{cm[i,j]}\n{rown[i,j]*100:.0f}%', ha='center', va='center',
                    color='white' if rown[i,j] > .5 else 'black', fontsize=10, fontweight='bold')
    ax.set_title('Matrice di confusione (coni abbinati)', fontweight='bold')
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def plot_pr(frames, class_names, thr, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in class_names:
        r, p = pr_curve(frames, c, thr)
        ax.plot(r, p, label=c, color=COLORS.get(c, 'gray'), lw=2)
    ax.set_xlabel('recall'); ax.set_ylabel('precision'); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title(f'Curve Precision-Recall per classe (tol. {thr} m)', fontweight='bold')
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def plot_loc_err(errs, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(errs, bins=40, color='tab:blue', alpha=.8)
    med = np.median(errs)
    ax.axvline(med, color='red', ls='--', label=f'mediana {med*100:.1f} cm')
    ax.set_xlabel('errore di localizzazione [m]'); ax.set_ylabel('n. veri positivi')
    ax.set_title('Precisione dei centri predetti', fontweight='bold')
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def plot_tolerance(sweep, thrs, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for k, v in sweep.items():
        style = '--' if k == 'class-agnostic' else '-'
        ax.plot(thrs, v, style, marker='o', ms=3, label=k,
                color=COLORS.get(k, 'green' if k == 'class-agnostic' else 'gray'))
    ax.set_xlabel('tolleranza [m]'); ax.set_ylabel('AP'); ax.set_ylim(0, 1.02)
    ax.set_title('AP al variare della tolleranza', fontweight='bold')
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--out_dir', type=Path, default=Path('eval_report'))
    ap.add_argument('--dist_thresh', type=float, default=0.5)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = pickle.load(open(args.dump, 'rb'))
    frames = data['frames']; class_names = data['class_names']
    T = args.dist_thresh

    # conteggi di base
    gt_counts = {c: 0 for c in class_names}
    for fr in frames:
        for gl in fr['gt_labels']:
            if gl in gt_counts:
                gt_counts[gl] += 1
    print(f'\n=== {len(frames)} frame val | tolleranza {T} m ===')
    print('coni veri per classe:', gt_counts)

    print('\n[1] AP per classe e fascia di distanza')
    results = E.evaluate(frames, class_names, dist_thresh=T)
    E.print_table(results, class_names)

    print(f'\n[2] AP class-agnostic (localizzazione): {class_agnostic_ap(frames, T):.3f}')

    print('\n[3] Matrice di confusione:')
    cm, missed, false = confusion_matrix(frames, class_names, T)
    hdr = ' '*9 + ''.join(c.ljust(9) for c in class_names) + 'MISSED'
    print(hdr)
    for i, c in enumerate(class_names):
        print(c.ljust(9) + ''.join(str(cm[i][j]).ljust(9) for j in range(len(class_names))) + str(missed[i]))
    print('FALSE'.ljust(9) + ''.join(str(false[j]).ljust(9) for j in range(len(class_names))))

    errs = localization_errors(frames, T)
    if len(errs):
        print(f'\n[5] Errore di localizzazione: mediana {np.median(errs)*100:.1f} cm, '
              f'p90 {np.percentile(errs,90)*100:.1f} cm')

    # grafici
    plot_confusion(cm, missed, false, class_names, args.out_dir / 'confusion_matrix.png')
    plot_pr(frames, class_names, T, args.out_dir / 'precision_recall.png')
    if len(errs):
        plot_loc_err(errs, args.out_dir / 'localization_error.png')
    thrs = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5]
    plot_tolerance(tolerance_sweep(frames, class_names, thrs), thrs, args.out_dir / 'tolerance_sweep.png')
    print(f'\nGrafici salvati in {args.out_dir}/')


if __name__ == '__main__':
    main()