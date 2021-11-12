from pathlib import Path
from functools import partial

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataset import ConcatDataset as TorchConcatDataset

#from shallow.data import *
from shallow import data, data_shard
from shallow.utils.common import check_field_is_none



def init_datasets_example(cfg):
    """
        DATASETS dictionary:
            keys are custom names to use in unet.yaml
            values are actual Datasets on desired folders
    """
    DATA_DIR = Path(cfg.INPUTS).absolute()
    if not DATA_DIR.exists(): raise Exception(DATA_DIR)
    mult = cfg['TRAIN']['HARD_MULT']
    weights = cfg['TRAIN']['WEIGHTS']

    AuxDataset = partial(ImageDataset, pattern='*.png')

    DATASETS = {
        "grid_1": AuxDataset(DATA_DIR/'grid1'),
        "grid_2": AuxDataset(DATA_DIR/'grid2'),
        "grid_3": AuxDataset(DATA_DIR/'grid3'),
        "grid_4": AuxDataset(DATA_DIR/'grid4'),

        "train_1": AuxDataset(DATA_DIR/'train1'),
        "train_2": AuxDataset(DATA_DIR/'train2'),
        "train_3": AuxDataset(DATA_DIR/'train3'),
        "train_4": AuxDataset(DATA_DIR/'train4'),

        "val_1": AuxDataset(DATA_DIR/'val1'),
        "val_2": AuxDataset(DATA_DIR/'val2'),
        "val_3": AuxDataset(DATA_DIR/'val3'),
        "val_4": AuxDataset(DATA_DIR/'val4'),

    }
    return DATASETS


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


def create_datasets(cfg, all_datasets, dataset_types, current_split_id=None):
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
        if 'DATASETS' in data_field and not check_field_is_none(data_field, 'DATASETS'):
            assert check_field_is_none(data_field, "SPLITS")
            datasets = [all_datasets[ds] for ds in data_field.DATASETS]
            ds = TorchConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
            converted_datasets[dataset_type] = ds

        elif 'SPLITS' in data_field and not check_field_is_none(data_field, "SPLITS"):
            assert check_field_is_none(data_field, "DATASETS")
            datasets = {}
            for split_id, split_datasets_ids in data_field.SPLITS.items():
                if current_split_id is not None and  current_split_id != split_id: continue
                if split_datasets_ids == (0,): continue
                split_datasets = [all_datasets[split_dataset_id] for split_dataset_id in split_datasets_ids]
                ds = TorchConcatDataset(split_datasets) if len(split_datasets) > 1 else split_datasets[0]
                datasets[split_id] = ds

            converted_datasets[dataset_type] = datasets
        else:
            pass
            #raise Exception('INVALID DATASET/SPLITS cfg')

    return converted_datasets


def build_datasets(cfg, aug_factory, predefined_datasets, dataset_types=['TRAIN', 'VALID', 'TEST'], num_proc=4, split_id=None, ext_last=False):
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

    extend_factories = {
        'PRELOAD':partial(data.PreloadingDataset, num_proc=num_proc, progress=tqdm),
        'PRELOAD_SHARDED':partial(data_shard.ShardedPreloadingDataset,
                                  num_proc=num_proc,
                                  progress=tqdm,
                                  seed=cfg.TRAIN.SEED,
                                  rank=cfg.PARALLEL.LOCAL_RANK,
                                  num_replicas=cfg.PARALLEL.WORLD_SIZE,
                                  to_tensor=cfg.FEATURES.TENSOR_DATASET),
        'MULTIPLY':data.MultiplyDataset,
        'CACHING':data.CachingDataset,
    }
    datasets = create_datasets(cfg, predefined_datasets, dataset_types, current_split_id=split_id)

    if split_id is not None:
        # split mode, split_id from cfg, S0, S1, ...
        assert isinstance(datasets['TRAIN'], dict) # split mode, train->splits->{s0:[], s1:[], s2:[]}
        datasets['TRAIN'] = datasets['TRAIN'][split_id]
        datasets['VALID'] = datasets['VALID'][split_id]

    if not ext_last: datasets = data.create_extensions(cfg, datasets, extend_factories)
    augs = data.create_augmentators(cfg, aug_factory, dataset_types)
    datasets = data.apply_augs_datasets(datasets, augs)
    if ext_last: datasets = data.create_extensions(cfg, datasets, extend_factories)
    return datasets


def build_dataloaders(cfg, datasets, **kwargs):
    dls = {}
    for kind, dataset in datasets.items():
        dls[kind] = build_dataloader(cfg, dataset, kind, **kwargs)
    return dls

def build_dataloader(cfg, dataset, mode, **kwargs):
    sampler = None 

    if cfg.PARALLEL.DDP:
        if sampler is None:
            sampler = DistributedSampler(dataset, num_replicas=cfg.PARALLEL.WORLD_SIZE, rank=cfg.PARALLEL.LOCAL_RANK, shuffle=True)

    MAX_PROC = 16
    num_workers = cfg.TRAIN.NUM_WORKERS #if mode=='TRAIN' else MAX_PROC
    shuffle = sampler is None

    dl = DataLoader(
        dataset,
        batch_size=cfg[mode]['BATCH_SIZE'],
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        **kwargs)
    return dl


