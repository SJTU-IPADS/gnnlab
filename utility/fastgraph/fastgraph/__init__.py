import os

from .dataset_loader import DatasetLoader


def dataset(name, root_path):
    assert(name in ['papers100M', 'com-friendster', 'reddit', 'products'])
    dataset_path = os.path.join(root_path, name)
    dataset_loader = DatasetLoader(dataset_path)
    return dataset_loader


__all__ = ['dataset']
