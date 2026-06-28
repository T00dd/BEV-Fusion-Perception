from pcdet.datasets import __all__ as PCDET_DATASET_REGISTRY
from .cone_dataset import ConeDataset

PCDET_DATASET_REGISTRY.append['ConeDataset'] = ConeDataset

__all__ = ['ConeDataset']