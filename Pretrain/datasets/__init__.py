# Copyright (c) Facebook, Inc. and its affiliates.
# from .scannet import ScannetDetectionDataset, ScannetDatasetConfig
from .sunrgbd import SunrgbdDetectionDataset, SunrgbdDatasetConfig, ClassificationDataset
from .sunrgbd import *

DATASET_FUNCTIONS = {
    "sunrgbd": [SunrgbdDetectionDataset, SunrgbdDatasetConfig],
}


def build_dataset(config):
    dataset_builder = DATASET_FUNCTIONS['sunrgbd'][0]
    dataset_config = DATASET_FUNCTIONS['sunrgbd'][1]()
    
    dataset_dict = {
        "train": dataset_builder(dataset_config, split_set="train", root_dir=None, augment=config.train_aug),
        "test": dataset_builder(dataset_config, split_set="val", root_dir=None, augment=False),
    }
    return dataset_dict, dataset_config
