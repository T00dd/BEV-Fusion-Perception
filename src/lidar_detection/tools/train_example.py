import os
import sys
import runpy

# repo root = .../  (questo file: <repo>/src/lidar_detection/tools/train_cones.py)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PCDET_TOOLS = os.path.join(REPO, 'lib', 'OpenPCDet', 'tools')

sys.path.insert(0, os.path.join(REPO, 'src'))   # per importare lidar_detection
sys.path.insert(0, PCDET_TOOLS)                  # per i 'train_utils' di OpenPCDet

import lidar_detection.datasets   # noqa: F401  -> registra ConeDataset

train_py = os.path.join(PCDET_TOOLS, 'train.py')
sys.argv[0] = train_py
runpy.run_path(train_py, run_name='__main__')