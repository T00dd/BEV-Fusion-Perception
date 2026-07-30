"""
robustness_probe.py -- misura la FRAGILITA' del detector a distribution shift, come
proxy del gap sim-to-real (senza dati reali).

Idea
----
Degrada le point cloud pulite di CARLA in modo controllato e crescente, rifa'
l'inferenza, e misura come cambiano recall/precisione/FP a SOGLIA OPERATIVA FISSA
(quella del pulito). Tre perturbazioni, ognuna stressa una metrica diversa:
  - dropout punti   -> ritorni mancanti / densita' reale   -> stressa la RECALL
  - jitter gaussiano-> rumore di misura oltre il tuo std    -> stressa la LOCALIZZAZIONE
  - clutter         -> erba/cordoli/detriti (cluster spuri) -> stressa la PRECISIONE

Le perturbazioni agiscono sull'INPUT (i punti); la GT resta pulita (i coni "veri"
restano quelli da trovare). Riusa la logica di matching di evaluate.py.

USO
---
  python robustness_probe.py --cfg cone_centerpoint_agnostic_test.yaml \
      --ckpt output/.../checkpoint_epoch_80.pth --out_dir robustness --max_frames 800
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import evaluate as EV   # riusa valid_mask, ap, pr_curve, collect_operating

DROP_LEVELS = [0.0, 0.1, 0.2, 0.4, 0.6]
JITTER_LEVELS = [0.0, 0.01, 0.03, 0.05, 0.10]      # metri
CLUTTER_LEVELS = [0, 5, 20, 50, 100]               # cluster spuri per frame


# --------------------------------------------------------------------------- #
# perturbazioni (agiscono su un array (N, nfeat))
# --------------------------------------------------------------------------- #
def perturb_drop(pts, p, rng):
    if p <= 0 or len(pts) == 0:
        return pts
    return pts[rng.random(len(pts)) >= p]


def perturb_jitter(pts, std, rng):
    if std <= 0 or len(pts) == 0:
        return pts
    q = pts.copy()
    q[:, :3] = q[:, :3] + rng.normal(0, std, size=(len(pts), 3)).astype(pts.dtype)
    return q


def perturb_clutter(pts, n, rng, pc_range, nfeat):
    if n <= 0:
        return pts
    xmin, ymin, zmin, xmax, ymax, zmax = pc_range
    out = [pts]
    for _ in range(n):
        cx = rng.uniform(xmin + 2, xmax); cy = rng.uniform(ymin, ymax)
        k = rng.randint(2, 5)                                  # cluster cono-like 2-4 punti
        c = np.zeros((k, nfeat), np.float32)
        c[:, 0] = cx + rng.uniform(-0.1, 0.1, k)
        c[:, 1] = cy + rng.uniform(-0.1, 0.1, k)
        c[:, 2] = rng.uniform(0.0, 0.32, k)                    # altezza tipo cono
        if nfeat > 3:
            c[:, 3] = rng.uniform(0.1, 0.9, k)
        out.append(c)
    return np.vstack(out)


# --------------------------------------------------------------------------- #
def run_under(model, dataset, perturb_fn, idxs, merge, score_thresh):
    """Inferenza con la perturbazione applicata a get_lidar. Ritorna 'frames'."""
    import torch
    from pcdet.models import load_data_to_gpu

    frames = []
    with torch.no_grad():
        for i in idxs:
            dataset._perturb = perturb_fn        # letto dal get_lidar patchato
            d = dataset[i]
            batch = dataset.collate_batch([d])
            load_data_to_gpu(batch)
            pred = model(batch)[0][0]
            pb = pred['pred_boxes'].cpu().numpy().reshape(-1, 7)
            ps = pred['pred_scores'].cpu().numpy()
            keep = ps >= score_thresh
            fid = batch['frame_id'][0]
            scene, fr = fid.rsplit('_', 2)[0], '_'.join(fid.rsplit('_', 2)[1:])
            gb, gn, gnp = dataset.get_label(scene, fr)
            if merge is not None and len(gn):
                gn = np.array([merge] * len(gn))
            frames.append({'frame_id': fid,
                           'pred_boxes': pb[keep], 'pred_scores': ps[keep],
                           'gt_boxes': gb.reshape(-1, 7), 'gt_npts': gnp.astype(int)})
    for fr in frames:
        fr['pred_boxes'] = np.asarray(fr['pred_boxes'], float).reshape(-1, 7)
        fr['gt_boxes'] = np.asarray(fr['gt_boxes'], float).reshape(-1, 7)
        fr['pred_scores'] = np.asarray(fr['pred_scores'], float)
        fr['gt_valid'] = EV.valid_mask(fr['gt_boxes'], fr['gt_npts'], 2, EV.PC_RANGE)
    return frames


def metrics_at(frames, op, T=0.5):
    R = EV.collect_operating(frames, T, op)
    tp, fp = R['tp'], R['fp']
    return {'recall': len(tp) / max(R['n_gt'], 1),
            'precision': len(tp) / max(len(tp) + len(fp), 1),
            'fp_per_frame': len(fp) / max(len(frames), 1),
            'ap': EV.ap(frames, T),
            'loc_median_cm': float(np.median([t['err'] for t in tp]) * 100) if tp else None}


# --------------------------------------------------------------------------- #
def plot_curve(levels, results, xlabel, title, path):
    fig, axL = plt.subplots(figsize=(6.8, 4.4)); axR = axL.twinx()
    rec = [r['recall'] for r in results]; prec = [r['precision'] for r in results]
    fp = [r['fp_per_frame'] for r in results]
    axL.plot(levels, rec, 'o-', color='tab:blue', label='recall')
    axL.plot(levels, prec, 's-', color='tab:green', label='precisione')
    axR.plot(levels, fp, '^--', color='tab:red', label='FP/frame')
    axL.set_xlabel(xlabel); axL.set_ylabel('recall / precisione'); axL.set_ylim(0, 1.02)
    axR.set_ylabel('FP/frame', color='tab:red'); axR.tick_params(axis='y', labelcolor='tab:red')
    axL.set_title(title, fontweight='bold'); axL.grid(alpha=.2)
    l1, la1 = axL.get_legend_handles_labels(); l2, la2 = axR.get_legend_handles_labels()
    axL.legend(l1 + l2, la1 + la2, loc='center left', fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out_dir', type=Path, default=Path('robustness'))
    ap.add_argument('--max_frames', type=int, default=800, help='sottocampiona per velocita\'')
    ap.add_argument('--score_thresh', type=float, default=0.01)
    ap.add_argument('--op', type=float, default=None, help='soglia operativa fissa (default: F1-max sul pulito)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import torch, functools
    torch.load = functools.partial(torch.load, weights_only=False)
    import lidar_detection.datasets   # noqa
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network
    from pcdet.utils import common_utils

    cfg_from_yaml_file(args.cfg, cfg)
    cfg.MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH = args.score_thresh
    logger = common_utils.create_logger()
    dataset, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                                     batch_size=1, dist=False, workers=0, logger=logger, training=False)
    merge = getattr(dataset, 'merge_to', None)
    pc_range = list(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    nfeat = dataset.num_point_features

    # patch di get_lidar: applica la perturbazione corrente (dataset._perturb)
    dataset._perturb = lambda pts: pts
    _orig = dataset.get_lidar
    dataset.get_lidar = lambda s, f: dataset._perturb(_orig(s, f))

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda(); model.eval()

    n = len(dataset)
    idxs = list(range(n)) if args.max_frames >= n else \
        sorted(np.random.RandomState(args.seed).choice(n, args.max_frames, replace=False).tolist())

    # 1) pulito -> soglia operativa fissa + metriche base
    clean = run_under(model, dataset, lambda p: p, idxs, merge, args.score_thresh)
    s, rec, prec, f1, _ = EV.pr_curve(clean, 0.5)
    op = args.op if args.op is not None else (float(s[int(np.argmax(f1))]) if len(f1) else 0.28)
    base = metrics_at(clean, op)
    print(f'op fissa = {op:.3f} | pulito: recall={base["recall"]:.3f} prec={base["precision"]:.3f} '
          f'FP/frame={base["fp_per_frame"]:.3f}')

    rng = np.random.RandomState(args.seed)
    all_results = {'op': op, 'n_frames': len(idxs)}

    for name, levels, mk in [
        ('dropout', DROP_LEVELS, lambda v: (lambda p: perturb_drop(p, v, rng))),
        ('jitter',  JITTER_LEVELS, lambda v: (lambda p: perturb_jitter(p, v, rng))),
        ('clutter', CLUTTER_LEVELS, lambda v: (lambda p: perturb_clutter(p, v, rng, pc_range, nfeat))),
    ]:
        res = []
        for v in levels:
            m = base if v == 0 else metrics_at(run_under(model, dataset, mk(v), idxs, merge, args.score_thresh), op)
            res.append(m)
            print(f'  {name:8s} sev={v:<5}: recall={m["recall"]:.3f} prec={m["precision"]:.3f} '
                  f'FP/frame={m["fp_per_frame"]:.3f} AP={m["ap"]:.3f}')
        all_results[name] = {'levels': levels, 'metrics': res}
        xl = {'dropout': 'frazione punti rimossi', 'jitter': 'std rumore [m]',
              'clutter': 'cluster spuri / frame'}[name]
        plot_curve(levels, res, xl, f'Robustezza a: {name}', args.out_dir / f'robustness_{name}.png')

    json.dump(EV.jsonable(all_results), open(args.out_dir / 'robustness.json', 'w'), indent=2)
    print(f'\nFigure + robustness.json in {args.out_dir}/')


if __name__ == '__main__':
    main()