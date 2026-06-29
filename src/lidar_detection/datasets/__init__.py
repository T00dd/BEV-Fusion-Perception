from pcdet.datasets import __all__ as PCDET_DATASET_REGISTRY
from .dataset_adapter import ConeDataset

PCDET_DATASET_REGISTRY['ConeDataset'] = ConeDataset

__all__ = ['ConeDataset']