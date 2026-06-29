import numpy as np


def match_frame(pred_xy, pred_scores, gt_xy, dist_thresh):
    
    n_pred, n_gt = len(pred_xy), len(gt_xy)
    if n_pred == 0:
        return np.zeros(0, bool), n_gt
    order = np.argsort(-pred_scores)            
    pred_xy = pred_xy[order]
    is_tp = np.zeros(n_pred, bool)
    gt_taken = np.zeros(n_gt, bool)
    for i, p in enumerate(pred_xy):
        if n_gt == 0:
            break
        d = np.linalg.norm(gt_xy - p, axis=1)
        d[gt_taken] = np.inf                    
        j = np.argmin(d)
        if d[j] <= dist_thresh:
            is_tp[i] = True
            gt_taken[j] = True                 
        # altrimenti resta FP
    return is_tp, n_gt


def average_precision(scores, is_tp, n_gt_total):
    
    if n_gt_total == 0:
        return float('nan')                   
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(is_tp, bool)[order].astype(np.float64)
    fp = 1.0 - tp
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recall = tp_cum / n_gt_total
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    # envelope: precision resa monotona decrescente da destra
    mrec = np.concatenate([[0.0], recall, [recall[-1]]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]    
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


def _dist_from_origin(xy):
    return np.linalg.norm(xy, axis=1) if len(xy) else np.zeros(0)


def evaluate(frames, class_names, dist_thresh=0.5,
             dist_bands=((0, 15), (15, 30), (30, 50))):
    
    results = {}
    bands = [('all', None)] + [(f'{a}-{b}m', (a, b)) for a, b in dist_bands]

    for band_name, band in bands:
        for cls in class_names:
            scores_all, tp_all, n_gt_total = [], [], 0
            for fr in frames:
                pl = np.asarray(fr['pred_labels'])
                gl = np.asarray(fr['gt_labels'])
                pm = (pl == cls)
                gm = (gl == cls)
                p_xy = np.asarray(fr['pred_xy'], float).reshape(-1, 2)[pm]
                p_sc = np.asarray(fr['pred_scores'], float)[pm]
                g_xy = np.asarray(fr['gt_xy'], float).reshape(-1, 2)[gm]
                if band is not None:             
                    a, b = band
                    pk = (_dist_from_origin(p_xy) >= a) & (_dist_from_origin(p_xy) < b)
                    gk = (_dist_from_origin(g_xy) >= a) & (_dist_from_origin(g_xy) < b)
                    p_xy, p_sc, g_xy = p_xy[pk], p_sc[pk], g_xy[gk]
                is_tp, n_gt = match_frame(p_xy, p_sc, g_xy, dist_thresh)
                scores_all.append(p_sc[np.argsort(-p_sc)] if len(p_sc) else p_sc)
                tp_all.append(is_tp)
                n_gt_total += n_gt
            scores_all = np.concatenate(scores_all) if scores_all else np.zeros(0)
            tp_all = np.concatenate(tp_all) if tp_all else np.zeros(0, bool)
            results[(band_name, cls)] = average_precision(scores_all, tp_all, n_gt_total)
    return results


def print_table(results, class_names):
    bands = sorted({b for b, _ in results}, key=lambda s: (s != 'all', s))
    header = 'fascia'.ljust(10) + ''.join(c.ljust(10) for c in class_names) + 'mAP'
    print(header); print('-' * len(header))
    for b in bands:
        vals = [results[(b, c)] for c in class_names]
        finite = [v for v in vals if v == v]
        mAP = np.mean(finite) if finite else float('nan')
        row = b.ljust(10) + ''.join(
            (f'{v:.3f}' if v == v else '  -  ').ljust(10) for v in vals)
        print(row + f'{mAP:.3f}')