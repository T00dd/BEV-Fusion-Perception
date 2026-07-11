import os
import pickle
import numpy as np
import copy
from pathlib import Path
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
import lidar_detection.datasets

def generate_infos(config_path):
    cfg_from_yaml_file(config_path, cfg)
    dataset_cfg = cfg.DATA_CONFIG
    root_path = Path(dataset_cfg.DATA_PATH)
    
    print(f"-> Initializing database generation for path: {root_path}")
    
    for split in ['train', 'val']:
        current_dataset_cfg = copy.deepcopy(dataset_cfg)
        
        if hasattr(current_dataset_cfg, 'DATA_AUGMENTOR') and current_dataset_cfg.DATA_AUGMENTOR is not None:
            current_dataset_cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = ['gt_sampling', 'placeholder']
        
        from lidar_detection.datasets.dataset_adapter import ConeDataset
        dataset = ConeDataset(
            dataset_cfg=current_dataset_cfg,
            class_names=cfg.CLASS_NAMES,
            training=False,
            root_path=root_path,
            logger=common_utils.create_logger()
        )
        
        dataset.split = split
        dataset.sample_list = dataset._build_sample_list()
        
        print(f"\n[Processing split: {split}] Found {len(dataset)} frames.")
        infos_list = []
        
        db_infos = {}
        for c in cfg.CLASS_NAMES:
            db_infos[c] = []
            
        for idx in range(len(dataset)):
            scene, frame_name = dataset.sample_list[idx]
            points = dataset.get_lidar(scene, frame_name)
            gt_boxes, gt_names, num_pts = dataset.get_label(scene, frame_name)
            
            info = {
                'point_cloud': {
                    'num_features': dataset.num_point_features,
                    'lidar_idx': f"{scene}_{frame_name}"
                },
                'annos': {
                    'gt_boxes_lidar': gt_boxes,
                    'name': gt_names,
                    'difficulty': np.zeros(len(gt_boxes), dtype=np.int32)
                }
            }
            infos_list.append(info)
            
            if split == 'train' and len(gt_boxes) > 0:
                for i, (box, name, n_pts) in enumerate(zip(gt_boxes, gt_names, num_pts)):
                    min_pts_required = dataset_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].PREPARE.filter_by_min_points
                    limit = 1
                    for rule in min_pts_required:
                        if rule.startswith(name):
                            limit = int(rule.split(':')[1])
                    
                    if n_pts >= limit:
                        db_info = {
                            'name': name,
                            'box3d_lidar': box,
                            'num_points_in_gt': n_pts,
                            'sample_idx': f"{scene}_{frame_name}",
                            'path': f"scenes/{scene}/lidar/{frame_name}.bin"
                        }
                        db_infos[name].append(db_info)
            
            if (idx + 1) % 500 == 0 or (idx + 1) == len(dataset):
                print(f"   Progress: {idx + 1}/{len(dataset)} frames parsed...")

        out_pkl = root_path / f"cone_infos_{split}.pkl"
        with open(out_pkl, 'wb') as f:
            pickle.dump(infos_list, f)
        print(f"-> Info file saved successfully: {out_pkl}")
        
        if split == 'train':
            out_db_pkl = root_path / "cone_dbinfos_train.pkl"
            with open(out_db_pkl, 'wb') as f:
                pickle.dump(db_infos, f)
            print(f"-> Ground truth sampling database saved successfully: {out_db_pkl}")
            for c in cfg.CLASS_NAMES:
                print(f"   Class [{c}]: {len(db_infos[c])} valid objects injected into the database.")

if __name__ == "__main__":
    generate_infos("src/lidar_detection/configs/second_centerpoint_cones.yaml")