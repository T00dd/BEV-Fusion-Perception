import argparse
import pickle
from pathlib import Path

import numpy as np

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from lidar_detection.datasets.dataset_adapter import ConeDataset


def points_in_box(points, box):
    """Maschera dei punti dentro la box [x,y,z,dx,dy,dz,heading]. heading=0 -> assi-allineata."""
    cx, cy, cz, dx, dy, dz, yaw = box
    q = points[:, :3] - np.array([cx, cy, cz], dtype=np.float32)
    if abs(yaw) > 1e-6:                       # robustezza, ma per i coni yaw=0
        c, s = np.cos(-yaw), np.sin(-yaw)
        q = q @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32).T
    return ((np.abs(q[:, 0]) <= dx / 2) &
            (np.abs(q[:, 1]) <= dy / 2) &
            (np.abs(q[:, 2]) <= dz / 2))


def list_frames(root, split):
    split_file = root / 'splits' / f'{split}.txt'
    scenes = [ln.strip() for ln in open(split_file) if ln.strip()]
    frames = []
    for sc in scenes:
        ld = root / 'scenes' / sc / 'lidar'
        if ld.exists():
            frames += [(sc, p.stem) for p in sorted(ld.glob('*.bin'))]
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    args = ap.parse_args()

    cfg_from_yaml_file(args.cfg, cfg)
    root = Path(cfg.DATA_CONFIG.DATA_PATH)
    class_names = list(cfg.CLASS_NAMES)
    logger = common_utils.create_logger()

    # training=False -> NON costruisce il data_augmentor, quindi NON tenta di
    # caricare il database (che stiamo generando ora): niente dipendenza circolare.
    dataset = ConeDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=class_names,
                          training=False, root_path=root, logger=logger)
    nfeat = dataset.num_point_features
    print(f'-> feature per punto: {nfeat} | classi: {class_names}')

    db_dir = root / 'gt_database'
    db_dir.mkdir(exist_ok=True)
    db_infos = {c: [] for c in class_names}

    for split in ['train', 'val', 'test']:
        frames = list_frames(root, split)
        print(f'\n[{split}] {len(frames)} frame')
        infos = []

        for k, (scene, frame) in enumerate(frames):
            points = dataset.get_lidar(scene, frame)
            gt_boxes, gt_names, gt_npts = dataset.get_label(scene, frame)

            infos.append({
                'point_cloud': {'num_features': nfeat, 'lidar_idx': f'{scene}_{frame}'},
                'annos': {
                    'gt_boxes_lidar': gt_boxes,
                    'name': gt_names,
                    'num_points_in_gt': gt_npts,
                    'difficulty': np.zeros(len(gt_boxes), np.int32),
                },
            })

            # database SOLO dal train
            if split == 'train':
                for i, (box, name) in enumerate(zip(gt_boxes, gt_names)):
                    if name not in db_infos:
                        continue
                    mask = points_in_box(points, box)
                    obj = points[mask].copy()
                    if len(obj) == 0:                 # niente punti -> niente da incollare
                        continue
                    obj[:, :3] -= box[:3]             # centra sull'origine (convenzione OpenPCDet)

                    fname = f'{scene}_{frame}_{name}_{i}.bin'
                    obj.astype(np.float32).tofile(db_dir / fname)
                    db_infos[name].append({
                        'name': name,
                        'path': f'gt_database/{fname}',   # relativo a DATA_PATH
                        'gt_idx': i,
                        'box3d_lidar': box.astype(np.float32),
                        'num_points_in_gt': int(len(obj)),
                        'sample_idx': f'{scene}_{frame}',
                        'num_point_features': nfeat,
                        'difficulty': 0,
                    })

            if (k + 1) % 1000 == 0 or (k + 1) == len(frames):
                print(f'   {k + 1}/{len(frames)}')

        with open(root / f'cone_infos_{split}.pkl', 'wb') as f:
            pickle.dump(infos, f)
        print(f'-> salvato cone_infos_{split}.pkl')

    with open(root / 'cone_dbinfos_train.pkl', 'wb') as f:
        pickle.dump(db_infos, f)
    print('\n-> salvato cone_dbinfos_train.pkl')

    # riepilogo: quanti coni nel db, e quanti superano la soglia filter_by_min_points
    prepare = cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].PREPARE.filter_by_min_points
    thr = {}
    for rule in prepare:
        c, v = rule.rsplit(':', 1); thr[c] = int(v)
    print('\nRiepilogo database (soglia filter_by_min_points tra parentesi):')
    for c in class_names:
        n_pts = np.array([d['num_points_in_gt'] for d in db_infos[c]])
        n_pass = int((n_pts >= thr.get(c, 1)).sum()) if len(n_pts) else 0
        print(f'   {c:14s}: {len(db_infos[c]):6d} coni totali | '
              f'{n_pass:6d} superano soglia {thr.get(c, 1)}')


if __name__ == '__main__':
    main()