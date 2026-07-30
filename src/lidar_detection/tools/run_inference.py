"""
run_inference.py -- dump completo delle predizioni per la valutazione geometrica.

Rispetto alla versione precedente salva i BOX 7D interi (non solo x,y) di predizioni
e GT: servono a evaluate.py per l'errore in z, l'errore sulle dimensioni e la
decomposizione radiale/laterale. Salva anche num_points_in_gt (detectabilita').

USO
---
  python run_inference.py --cfg cone_centerpoint_agnostic.yaml --ckpt ckpt.pth \
      --out val_predictions.pkl --score_thresh 0.01
"""
import argparse
import pickle

import numpy as np
import torch

import lidar_detection.datasets   # noqa: F401  (registry)

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

import functools
torch.load = functools.partial(torch.load, weights_only=False)


def parse_frame_id(fid):
    parts = fid.rsplit('_', 2)
    return parts[0], parts[1] + '_' + parts[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='val_predictions.pkl')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--score_thresh', type=float, default=0.01,
                    help='basso, per una curva PR completa')
    args = ap.parse_args()

    cfg_from_yaml_file(args.cfg, cfg)
    cfg.MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH = args.score_thresh
    logger = common_utils.create_logger()

    dataset, loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size, dist=False, workers=args.workers,
        logger=logger, training=False,
    )
    # se il dataset e' in modalita' class-agnostic, collassiamo anche i GT in eval
    merge = getattr(dataset, 'merge_to', None)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda(); model.eval()

    dump = []
    with torch.no_grad():
        for batch in loader:
            load_data_to_gpu(batch)
            pred_dicts, _ = model(batch)
            for i, pd in enumerate(pred_dicts):
                boxes = pd['pred_boxes'].cpu().numpy()            # (N,7)
                scores = pd['pred_scores'].cpu().numpy()          # (N,)
                labels = pd['pred_labels'].cpu().numpy()          # 1-indexed
                names = np.array([cfg.CLASS_NAMES[l - 1] for l in labels])

                scene, fr = parse_frame_id(batch['frame_id'][i])
                gt_boxes, gt_names, gt_npts = dataset.get_label(scene, fr)
                if merge is not None and len(gt_names):
                    gt_names = np.array([merge] * len(gt_names))

                dump.append({
                    'frame_id': batch['frame_id'][i],
                    'pred_boxes': boxes.astype(np.float32),       # (N,7) x,y,z,dx,dy,dz,heading
                    'pred_scores': scores.astype(np.float32),
                    'pred_labels': names,
                    'gt_boxes': gt_boxes.astype(np.float32),      # (M,7)
                    'gt_labels': gt_names,
                    'gt_npts': gt_npts.astype(np.int32),          # punti LiDAR per cono
                })

    with open(args.out, 'wb') as f:
        pickle.dump({'class_names': list(cfg.CLASS_NAMES), 'frames': dump}, f)
    print(f'Salvate le predizioni di {len(dump)} frame in {args.out}')


if __name__ == '__main__':
    main()