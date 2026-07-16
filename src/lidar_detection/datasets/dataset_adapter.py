# adapter OpenPCDet for Cone Dataset
# serves as a first translation layer betweem OpenPCDet and disk stored data
# produces the standardized data format used by OpenPCDet

import json
from pathlib import Path
import numpy as np

from pcdet.datasets.dataset import DatasetTemplate

CLASS_MAP = {'blue': 'blue', 'yellow': 'yellow', 'orange_small': 'orange_small'}


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
        self.min_points_for_gt = self.dataset_cfg.get('MIN_POINTS_FOR_GT', 3)

        # CLASS-AGNOSTIC MERGE (vedi collapse_names)
        self.merge_to = self.dataset_cfg.get('MERGE_CLASSES_TO', None)
        if self.merge_to is not None and self.logger is not None:
            self.logger.info(f'ConeDataset: merge di tutte le classi -> "{self.merge_to}"')

        # ------------------------------------------------------------------ #
        # DOMAIN RANDOMIZATION sul rumore dei punti (solo in training).
        # Indurisce contro il gap sim-to-real: jitter di posizione (il fallimento
        # critico del robustness probe), dropout (densita' reale variabile) e
        # clutter cono-like non etichettato (insegna a RIFIUTARE i sosia -> precisione).
        # Parametri da NOISE_AUG nel dataset yaml.
        # ------------------------------------------------------------------ #
        self.noise_cfg = self.dataset_cfg.get('NOISE_AUG', None)
        if self.noise_cfg is not None and self.noise_cfg.get('ENABLED', False) and self.logger is not None:
            self.logger.info(f'ConeDataset: NOISE_AUG attivo (training) -> {dict(self.noise_cfg)}')

        self.sample_list = self._build_sample_list()
        if self.logger is not None:
            self.logger.info(f'ConeDataset [{self.split}]: {len(self.sample_list)} frames')

    def _build_sample_list(self):
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

    def collapse_names(self, names):
        if self.merge_to is None or len(names) == 0:
            return names
        return np.array([self.merge_to] * len(names))

    def get_lidar(self, scene, frame_name):
        path = self.scene_dir / scene / 'lidar' / f'{frame_name}.bin'
        pts = np.fromfile(str(path), dtype=np.float32)
        assert pts.size % self.num_point_features == 0, (f'Expected {self.num_point_features} features per point, but .bin file has {pts.size} values')
        return pts.reshape(-1, self.num_point_features)

    def get_label(self, scene, frame_name, merge=False):
        path = self.scene_dir / scene / 'labels' / f'{frame_name}.json'
        if not path.exists():
            return (np.zeros((0, 7), np.float32),
                    np.zeros(0, dtype='<U16'),
                    np.zeros(0, np.int32))

        cones = json.load(open(path))['cones']
        boxes, names, npts = [], [], []
        for c in cones:
            raw = c['class']
            assert raw in CLASS_MAP, (f'{path}: classe sconosciuta {raw!r}. Aggiungila a CLASS_MAP.')
            boxes.append(c['box'])
            names.append(CLASS_MAP[raw])
            npts.append(c['num_lidar_points'])

        boxes = np.array(boxes, np.float32).reshape(-1, 7)
        names = np.array(names)
        npts = np.array(npts, np.int32)
        if merge:
            names = self.collapse_names(names)
        return boxes, names, npts

    # ---------------------------------------------------------------------- #
    # rumore per-punto (domain randomization) - solo in training
    # ---------------------------------------------------------------------- #
    def _add_clutter(self, points, n):
        pcr = self.dataset_cfg.POINT_CLOUD_RANGE   # [xmin,ymin,zmin,xmax,ymax,zmax]
        rng = np.random
        chunks = [points]
        for _ in range(n):
            cx = rng.uniform(pcr[0] + 2.0, pcr[3])
            cy = rng.uniform(pcr[1], pcr[4])
            k = rng.randint(2, 5)                          # cluster cono-like: 2-4 punti
            c = np.zeros((k, self.num_point_features), np.float32)
            c[:, 0] = cx + rng.uniform(-0.1, 0.1, k)
            c[:, 1] = cy + rng.uniform(-0.1, 0.1, k)
            c[:, 2] = rng.uniform(0.0, 0.32, k)            # altezza tipo cono
            if self.num_point_features > 3:
                c[:, 3] = rng.uniform(0.85, 0.97, k)       # intensita' plausibile
            chunks.append(c)
        return np.vstack(chunks)

    def _augment_points(self, points):
        cfg = self.noise_cfg
        if cfg is None or not cfg.get('ENABLED', False) or not self.training or len(points) == 0:
            return points
        rng = np.random
        # dropout: frazione per-frame ~ U(0, DROP_MAX)
        dmax = float(cfg.get('DROP_MAX', 0.0))
        if dmax > 0:
            p = rng.uniform(0.0, dmax)
            if p > 0:
                points = points[rng.random(len(points)) >= p]
        # jitter: sigma per-frame ~ U(0, JITTER_STD_MAX), gaussiano per-punto su x,y,z
        jmax = float(cfg.get('JITTER_STD_MAX', 0.0))
        if jmax > 0 and len(points):
            sigma = rng.uniform(0.0, jmax)
            points = points.copy()
            points[:, :3] = points[:, :3] + rng.normal(0.0, sigma, size=(len(points), 3)).astype(points.dtype)
        # clutter: n cluster cono-like per-frame ~ U(0, CLUTTER_MAX), NON etichettati
        cmax = int(cfg.get('CLUTTER_MAX', 0))
        if cmax > 0:
            k = rng.randint(0, cmax + 1)
            if k > 0:
                points = self._add_clutter(points, k)
        return points

    def __getitem__(self, index):
        scene, frame_name = self.sample_list[index]

        points = self.get_lidar(scene, frame_name)
        points = self._augment_points(points)          # rumore per-punto (solo training)
        gt_boxes, gt_names, num_pts = self.get_label(scene, frame_name)

        if self.training and len(gt_boxes) > 0:
            keep = num_pts >= self.min_points_for_gt
            gt_boxes, gt_names = gt_boxes[keep], gt_names[keep]

        gt_names = self.collapse_names(gt_names)        # class-agnostic merge

        input_dict = {
            'points': points,
            'gt_boxes': gt_boxes,
            'gt_names': gt_names,
            'frame_id': f'{scene}_{frame_name}',
        }
        return self.prepare_data(data_dict=input_dict)