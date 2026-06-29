import argparse

import numpy as np
import torch

import lidar_detection.datasets   # noqa: F401  (registry)

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

import cone_eval as E             # stesso folder tools/


def parse_frame_id(fid):
    # 'scene_0000_frame_000000' -> ('scene_0000', 'frame_000000')
    parts = fid.rsplit('_', 2)
    return parts[0], parts[1] + '_' + parts[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--dist_thresh', type=float, default=0.5, help='tolleranza in metri')
    ap.add_argument('--min_points', type=int, default=1,
                    help='valuta solo coni con >= N punti (detectabilita\')')
    ap.add_argument('--score_thresh', type=float, default=None,
                    help='abbassa la soglia in eval per una PR completa (es. 0.01)')
    args = ap.parse_args()

    cfg_from_yaml_file(args.cfg, cfg)
    if args.score_thresh is not None:
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

    frames = []
    with torch.no_grad():
        for batch in loader:
            load_data_to_gpu(batch)
            pred_dicts, _ = model(batch)        # eval: la testa decodifica i picchi
            for i, pd in enumerate(pred_dicts):
                boxes = pd['pred_boxes'].cpu().numpy()      # (N,7)
                scores = pd['pred_scores'].cpu().numpy()    # (N,)
                labels = pd['pred_labels'].cpu().numpy()    # (N,) 1-indexed
                names = np.array([cfg.CLASS_NAMES[l - 1] for l in labels])

                scene, fr = parse_frame_id(batch['frame_id'][i])
                gt_boxes, gt_names, gt_npts = dataset.get_label(scene, fr)
                keep = gt_npts >= args.min_points           # solo coni detectabili

                frames.append({
                    'pred_xy': boxes[:, :2],
                    'pred_scores': scores,
                    'pred_labels': names,
                    'gt_xy': gt_boxes[keep, :2],
                    'gt_labels': gt_names[keep],
                })

    print(f'\nValutati {len(frames)} frame | tolleranza {args.dist_thresh} m | '
          f'coni con >= {args.min_points} punti\n')
    results = E.evaluate(frames, cfg.CLASS_NAMES, dist_thresh=args.dist_thresh)
    E.print_table(results, cfg.CLASS_NAMES)

    import cone_diagnostic as D
    print('\nAP class-agnostic (localizzazione):', round(D.class_agnostic_ap(frames, args.dist_thresh), 3))
    cm, missed, false = D.confusion_matrix(frames, cfg.CLASS_NAMES, args.dist_thresh)
    D.print_confusion(cm, missed, false, cfg.CLASS_NAMES)


if __name__ == '__main__':
    main()