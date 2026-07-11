import argparse
import pickle

import numpy as np
import torch

import lidar_detection.datasets   # noqa: F401  (registry)

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

import torch
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
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda(); model.eval()

    dump = []
    with torch.no_grad():
        for batch in loader:
            load_data_to_gpu(batch)
            pred_dicts, _ = model(batch)
            for i, pd in enumerate(pred_dicts):
                boxes = pd['pred_boxes'].cpu().numpy()
                scores = pd['pred_scores'].cpu().numpy()
                labels = pd['pred_labels'].cpu().numpy()          # 1-indexed
                names = np.array([cfg.CLASS_NAMES[l - 1] for l in labels])

                scene, fr = parse_frame_id(batch['frame_id'][i])
                gt_boxes, gt_names, gt_npts = dataset.get_label(scene, fr)

                dump.append({
                    'frame_id': batch['frame_id'][i],
                    'pred_xy': boxes[:, :2].astype(np.float32),
                    'pred_scores': scores.astype(np.float32),
                    'pred_labels': names,
                    'gt_xy': gt_boxes[:, :2].astype(np.float32),
                    'gt_labels': gt_names,
                    'gt_npts': gt_npts.astype(np.int32),          # per detectabilita'/distanza
                })

    with open(args.out, 'wb') as f:
        pickle.dump({'class_names': list(cfg.CLASS_NAMES), 'frames': dump}, f)
    print(f'Salvate le predizioni di {len(dump)} frame in {args.out}')


if __name__ == '__main__':
    main()