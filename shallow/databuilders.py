from pathlib import Path
from functools import partial

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataset import ConcatDataset as TorchConcatDataset

#from shallow.data import *
from shallow import data, data_shard
from shallow.utils.common import check_field_is_none


def build_transform_fact_example():
    # TODO fix rename transform for augs
    #def train_trans_get(*args, **kwargs): return augs.get_aug(*args, **kwargs)
    def train_trans_get(*args, **kwargs): return None

    transform_factory = {
            'TRAIN':{'factory':TransformDataset, 'transform_getter':train_trans_get},
            'VALID':{'factory':TransformDataset, 'transform_getter':train_trans_get},
            'TEST':{'factory':TransformDataset, 'transform_getter':train_trans_get},
        }
    return transform_factory


def create_datasets(cfg, all_datasets, dataset_types):
    """
        Joins lists of datasets with TRAIN, VALID, ... types into concated datasets:
            {
                "TRAIN": ConcatDataset1, | or in splits mode : "TRAIN": {'S0':[ds0], 'S1':[ds1,ds2,...]}
                "VALID": ConcatDataset2,
                ...
            }
    """
    converted_datasets = {}
    for dataset_type in dataset_types:
        data_field = cfg.DATA[dataset_type]
        datasets = [all_datasets[ds] for ds in data_field.DATASETS]
        ds = TorchConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
        converted_datasets[dataset_type] = ds
    return converted_datasets


def build_datasets(cfg, predefined_datasets, dataset_types=['TRAIN', 'VALID', 'TEST']):
    """
        Creates dictionary :
        {
            'TRAIN': <122254afsf9a>.obj.dataset,
            'VALID': <ascas924ja>.obj.dataset,
            'TEST': <das92hjasd>.obj.dataset,
        }

        train_dataset = build_datasets(cfg)['TRAIN']
        preprocessed_image, preprocessed_mask = train_dataset[0]

        All additional operations like preloading into memory, augmentations, etc
        is just another Dataset over existing one.
            preload_dataset = PreloadingDataset(MyDataset)
            augmented_dataset = AugDataset(preload_dataset)
    """
    datasets = create_datasets(cfg, predefined_datasets, dataset_types)
    return datasets
