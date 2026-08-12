"""
wandb_log_pointclouds.py -- logga scene 3D INTERATTIVE su wandb (killer feature LiDAR).

Per ogni frame di validazione carica su wandb un visualizzatore 3D ruotabile con:
  * la nuvola di punti,
  * i box di GROUND TRUTH (verdi),
  * i box PREDETTI sopra soglia (rossi, etichettati con lo score).
Cosi' vedi a occhio QUALI coni la rete si perde e se sbaglia posizione/dimensioni.

Puoi lanciarlo su un checkpoint qualsiasi (fine training o intermedio):
  python wandb_log_pointclouds.py --cfg cone_centerpoint_agnostic.yaml \
      --ckpt output/.../checkpoint_epoch_80.pth --num_frames 12 --score_thresh 0.3

Nota: e' una run wandb separata (job_type=viz). Per accumulare piu' epoche nello stesso
pannello, passa --step <epoca> e riusa lo stesso --name.
"""
import argparse

import numpy as np
import torch

import lidar_detection.datasets   # noqa: F401  (registry)

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

import functools
torch.load = functools.partial(torch.load, weights_only=False)

import wandb


def parse_frame_id(fid):
    parts = fid.rsplit('_', 2)
    return parts[0], parts[1] + '_' + parts[2]


def box_corners(b):
    """8 vertici del box [x,y,z,dx,dy,dz,heading] nel formato atteso da wandb."""
    x, y, z, dx, dy, dz, yaw = [float(v) for v in b[:7]]
    xs, ys, zs = dx / 2, dy / 2, dz / 2
    c = np.array([[xs, ys, zs], [xs, -ys, zs], [-xs, -ys, zs], [-xs, ys, zs],
                  [xs, ys, -zs], [xs, -ys, -zs], [-xs, -ys, -zs], [-xs, ys, -zs]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    return ((c @ Rz.T) + np.array([x, y, z])).tolist()


def make_object3d(points, gt_boxes, pred_boxes, pred_scores, max_points):
    if len(points) > max_points:
        sel = np.random.RandomState(0).choice(len(points), max_points, replace=False)
        points = points[sel]
    boxes = []
    for g in gt_boxes:
        boxes.append({"corners": box_corners(g), "label": "gt", "color": [0, 255, 0]})
    for b, s in zip(pred_boxes, pred_scores):
        boxes.append({"corners": box_corners(b), "label": f"{s:.2f}", "color": [255, 0, 0]})
    return wandb.Object3D({
        "type": "lidar/beta",
        "points": points[:, :3].astype(np.float32),
        "boxes": np.array(boxes),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--num_frames', type=int, default=12)
    ap.add_argument('--score_thresh', type=float, default=0.3)
    ap.add_argument('--max_points', type=int, default=40000, help='sottocampiona per il browser')
    ap.add_argument('--project', default='thesis')
    ap.add_argument('--name', default='pointcloud_viz')
    ap.add_argument('--entity', default='andrewboa-universit-degli-studi-di-trento')
    ap.add_argument('--step', type=int, default=None, help='es. numero di epoca')
    args = ap.parse_args()

    cfg_from_yaml_file(args.cfg, cfg)
    cfg.MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH = min(args.score_thresh, 0.05)
    logger = common_utils.create_logger()

    dataset, loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=2, logger=logger, training=False)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda(); model.eval()

    wandb.init(entity=args.entity, project=args.project, name=args.name, job_type='viz')

    logged = 0
    with torch.no_grad():
        for batch in loader:
            load_data_to_gpu(batch)
            preds, _ = model(batch)
            pd = preds[0]
            boxes = pd['pred_boxes'].cpu().numpy()
            scores = pd['pred_scores'].cpu().numpy()
            keep = scores >= args.score_thresh

            scene, fr = parse_frame_id(batch['frame_id'][0])
            points = dataset.get_lidar(scene, fr)
            gt_boxes, _, _ = dataset.get_label(scene, fr)

            obj = make_object3d(points, gt_boxes, boxes[keep], scores[keep], args.max_points)
            payload = {f'scene/{batch["frame_id"][0]}': obj}
            if args.step is not None:
                wandb.log(payload, step=args.step)
            else:
                wandb.log(payload)

            logged += 1
            if logged >= args.num_frames:
                break

    wandb.finish()
    print(f'Loggate {logged} scene 3D su wandb ({args.project}/{args.name}).')


if __name__ == '__main__':
    main()