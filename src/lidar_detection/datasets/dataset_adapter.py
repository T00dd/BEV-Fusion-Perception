# adapter OpenPCDet for Cone Dataset
# serves as a first translation layer betweem OpenPCDet and disk stored data
# produces the standardized data format used by OpenPCDet

import json
from pathlib import Path
import numpy as np

from pcdet.datasets.dataset import DatasetTemplate

CLASS_MAP = {'blue': 'blue', 'yellow': 'yellow'}

class ConeDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        
        if self.root_path is None:
            self.root_path = Path(self.dataset_cfg.DATA_PATH)

        self.root_path = Path(self.root_path)
        self.scene_dir = self.root_path / 'scenes'
        self.split_dir = self.root_path / 'splits'

        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]

        # number of columns in .bin files, e.g. 4 for x,y,z,intensity
        self.num_point_features = len(self.dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)
        self.min_points_for_gt = self.dataset_cfg.get('MIN_POINTS_FOR_GT', 1)

        self.sample_list = self._build_sample_list()
        if self.logger is not None:
            self.logger.info(f'ConeDataset [{self.split}]: {len(self.sample_list)} frames')

    def _build_sample_list(self):

        # expand each scene in sample frames. Split remains per scene, but sample_list is per frame
        split_file = self.split_dir / f'{self.split}.txt'
        assert split_file.exists(), f'Manca lo split file: {split_file}'

        scenes = [ln.strip() for ln in open(split_file) if ln.strip()]
        self._num_scenes = len(scenes)

        samples = []
        for scene in scenes:
            lidar_dir = self.scene_dir / scene / 'lidar'
            if not lidar_dir.exists():
                if self.logger is not None:
                    self.logger.warning(f'scena senza cartella lidar, salto: {scene}')
                continue
            for bin_path in sorted(lidar_dir.glob('*.bin')):
                samples.append((scene, bin_path.stem))
        return samples
    
    def __len__(self):
        return len(self.sample_list)
    
    def get_lidar(self, scene, frame_name):
        path = self.scene_dir / scene / 'lidar' / f'{frame_name}.bin'
        pts = np.fromfile(str(path), dtype=np.float32)

        # check that .bin file has the expected number of features
        assert pts.size % self.num_point_features == 0, (f'Expected {self.num_point_features} features per point, but .bin file has {pts.size} values')
        return pts.reshape(-1, self.num_point_features)
    
    def get_label(self, scene, frame_name):
        path = self.scene_dir / scene / 'labels' / f'{frame_name}.json'
        if not path.exists():
            return (np.zeros((0, 7), np.float32),
                    np.zeros(0, dtype='<U16'),
                    np.zeros(0, np.int32))
 
        cones = json.load(open(path))['cones']
        boxes, names, npts = [], [], []
        for c in cones:
            raw = c['class']
            assert raw in CLASS_MAP, (
                f'{path}: classe sconosciuta {raw!r}. Aggiungila a CLASS_MAP.'
            )
            # field 'box' is already a list of 7 floats, in the order [x, y, z, dx, dy, dz, heading]
            boxes.append(c['box'])
            names.append(CLASS_MAP[raw])
            npts.append(c['num_lidar_points'])
 
        boxes = np.array(boxes, np.float32).reshape(-1, 7)
        names = np.array(names)
        npts = np.array(npts, np.int32)
        return boxes, names, npts
    
    def __getitem__(self, index):
        scene, frame_name = self.sample_list[index]
 
        points = self.get_lidar(scene, frame_name)
        gt_boxes, gt_names, num_pts = self.get_label(scene, frame_name)
 
        # apply filtering of ground truth boxes based on number of points, if in training mode
        if self.training and len(gt_boxes) > 0:
            keep = num_pts >= self.min_points_for_gt
            gt_boxes, gt_names = gt_boxes[keep], gt_names[keep]
 
        input_dict = {
            'points': points,
            'gt_boxes': gt_boxes,       # (M, 7)
            'gt_names': gt_names,       # (M,)  string names of classes
            'frame_id': f'{scene}_{frame_name}',   # unique frame identifier
        }

        # prepare data: apply data augmentation, feature encoding and voxelization
        # convert gt_names to class indices, and add other fields required by OpenPCDet
        return self.prepare_data(data_dict=input_dict)

