
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/data.ipynb

from pathlib import Path
from functools import lru_cache, partial

# from tqdm.auto import tqdm
from tqdm.notebook import tqdm

import os
import cv2
import yaml
from PIL import Image
import numpy as np
import albumentations as albu
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset as ConcatDataset

from shallow import nb_utils


class Dataset:
    def __init__(self, root, pattern, filter_fn=nb_utils.noop):
        self.pattern = pattern
        self.root = Path(root)
        files = list(self.root.glob(self.pattern))
        assert len(files) > 0, 'There is no matching files!'
        files = sorted(files)
        self.files = filter_fn(files)
        assert len(self.files) > 0, 'Filtered out all files!'
        self.files_map = {f.with_suffix('').name:i for i,f in enumerate(self.files)}

    def load_item(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        item = self.load_item(idx)
        return item

    def __len__(self):
        return len(self.files)

#     def __add__(self, other):
#         return ConcatDataset([self, other])


class ImageDataset(Dataset):
    def load_item(self, idx):
        img_path = self.files[idx]
        #img = cv2.imread(str(img_path))
        #img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        img = Image.open(str(img_path))
        return img

class PairDataset:
    def __init__(self, ds1, ds2):
        self.ds1, self.ds2 = ds1, ds2
        self.check_len()

    def check_len(self):
        assert len(self.ds1) == len(self.ds2)

    def __getitem__(self, idx):
        return self.ds1.__getitem__(idx), self.ds2.__getitem__(idx)

    def __len__(self):
        return len(self.ds1)

class TransformDataset:
    def __init__(self, dataset, transforms, is_masked=False):
        self.dataset = dataset
        self.transforms = albu.Compose([]) if transforms is None else transforms
        self.is_masked = is_masked

    def __getitem__(self, idx):
        item = self.dataset.__getitem__(idx)
        if self.is_masked:
            img, mask = item
            augmented = self.transforms(image=img, mask=mask)
            return augmented["image"], augmented["mask"]
        else:
            return self.transforms(image=item[0], mask=None)['image']

    def __len__(self):
        return len(self.dataset)

class MultiplyDataset:
    def __init__(self, dataset, rate):
        _dataset = ConcatDataset([dataset])
        for i in range(rate-1):
            _dataset += ConcatDataset([dataset])
        self.dataset = _dataset

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def __len__(self):
        return len(self.dataset)

class CachingDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def __len__(self):
        return len(self.dataset)


class PreloadingDataset:
    def __init__(self, dataset, num_proc=False):
        self.dataset = dataset
        self.num_proc = num_proc
        if self.num_proc:
            self.data = nb_utils.mp_func_gen(self.preload_data, range(len(self.dataset)) , self.num_proc)
        else:
            self.data = self.preload_data((self.dataset, range(len(self.dataset))))

    def preload_data(self, args):
        idxs = args
        data = []
        for i in idxs:
            r = self.dataset.__getitem__(i)
            data.append(r)
        return data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class GpuPreloadingDataset:
    def __init__(self, dataset, devices):
        self.dataset = dataset
        self.devices = devices
        self.data = self.preload_data()

    def preload_data(self):
        data = []
        for i in range(len(self.dataset)):
            item, idx = self.dataset.__getitem__(i)
            item = item.to(self.devices[0])
            data.append((item, idx))
        return data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.dataset)

class DatasetCatalog():
    DATA_DIR = "/tmp/"
    DATA_DIR_MNT = "/mnt/tmp"

    DATASETS = {
        "default": {
            'factory':'default',
            "root": "def_root",
        }
    }
    @staticmethod
    def create_factory_dict(data_dir, dataset_attrs):
        #{factory:Dataset, args:args}
        raise NotImplementedError

    @classmethod
    def get(cls, name):
        try:
            attrs = cls.DATASETS[name]
        except:
            print(cls.DATASETS)
            raise RuntimeError("Dataset not available: {}".format(name))

        if os.path.exists(cls.DATA_DIR):
            data_dir = cls.DATA_DIR
        elif os.path.exists(cls.DATA_DIR_MNT):
            data_dir = cls.DATA_DIR_MNT

        return cls.create_factory_dict(data_dir, attrs)



# dataset_factories = {'termit':TermitDataset}
# transform_factories = {'TRAIN':{'factory':TransformDataset_Partial_HARD, 'transform_getter':get_aug}}
# extend_factories = {'GPU_PRELOAD':GpuPreloadingDataset_Partial_GPU0}
# dataset_types = ['TRAIN', 'VALID', 'TEST']
# datasets = {'TRAIN': dataset1, 'VALID': ...}

def extend_dataset(ds, data_field, extend_factories):
    for k, factory in extend_factories.items():
        field_val = data_field.get(k, None)
        if field_val:
            args = {}
            if isinstance(field_val, dict): args.update(field_val)
            ds = factory(ds, **args)
    return ds

def create_extensions(cfg, datasets, extend_factories):
    extended_datasets = {}
    for kind, ds in datasets.items():
        extended_datasets[kind] = extend_dataset(ds, cfg.DATA[kind], extend_factories)
    return extended_datasets


def create_datasets(cfg,
                   catalog,
                   dataset_factories,
                   dataset_types=['TRAIN', 'VALID', 'TEST']):

    def _create_dataset_fact(ds):
        dataset_attrs = catalog.get(ds)
        factory = dataset_factories[dataset_attrs['factory']]
        return factory(**dataset_attrs['args'])

    converted_datasets = {}
    for dataset_type in dataset_types:
        data_field = cfg.DATA[dataset_type]
        datasets_strings = data_field.DATASETS

        if datasets_strings:
            datasets = [_create_dataset_fact(ds) for ds in datasets_strings]
            ds = ConcatDataset(datasets) if len(datasets)>1 else datasets[0]
            converted_datasets[dataset_type] = ds
    return converted_datasets


def create_transforms(cfg,
                      transform_factories,
                      dataset_types=['TRAIN', 'VALID', 'TEST']):
    transformers = {}
    for dataset_type in dataset_types:
        aug_type = cfg.TRANSFORMERS[dataset_type]['AUG']
        args={
            'aug_type':aug_type,
            'size':cfg.TRANSFORMERS.CROP_SIZE
        }
        transform_getter = transform_factories[dataset_type]['transform_getter'](**args)
        transformer = partial(transform_factories[dataset_type]['factory'], transforms=transform_getter)
        transformers[dataset_type] = transformer
    return transformers

def apply_transforms_datasets(datasets, transforms):
    return [transforms[dataset_type](dataset) for dataset_type, dataset in datasets.items()]

class DatasetBuilder:
    def __init__(self, cfg,
                       catalog,
                       dataset_factories,
                       transform_factory,
                       dataset_types=['TRAIN', 'VALID', 'TEST']):
        nb_utils.store_attr(self, locals())

    def build_datasets(self):
        transformers = self._build_transformers()
        converted_datasets = {}
        for dataset_type in self.dataset_types:
            data_field = self.cfg.DATA[dataset_type]
            datasets_strings = data_field.DATASETS

            if datasets_strings:
                datasets = [self._create_dataset_fact(ds) for ds in datasets_strings]
                ds = ConcatDataset(datasets) if len(datasets)>1 else datasets[0]
                ds = transformers[dataset_type](ds)
                converted_datasets[dataset_type] = ds
        return converted_datasets

    def _create_dataset_fact(self, ds):
        dataset_attrs = self.catalog.get(ds)
        factory = self.dataset_factories[dataset_attrs['factory']]
        return factory(**dataset_attrs['args'])

    def _build_transformers(self):
        transformers = {}
        for dataset_type in self.dataset_types:
            aug_type = self.cfg.TRANSFORMERS[dataset_type]['AUG']
            args={
                'aug_type':aug_type,
                'size':self.cfg.TRANSFORMERS.CROP_SIZE
            }
            transform_getter = self.transform_factory[dataset_type]['transform_getter'](**args)
            transformer = partial(self.transform_factory[dataset_type]['factory'], transforms=transform_getter)
            transformers[dataset_type] = transformer
        return transformers

def build_dataloaders(datasets, samplers=None, batch_size=1, num_workers=0, drop_last=False, pin=False):
    dls = {}
    for kind, dataset in datasets.items():
        sampler = samplers[kind] if samplers is not None else None
        dls[kind] = create_dataloader(dataset, sampler, batch_size, num_workers, drop_last, pin)
    return dls

def create_dataloader(dataset, sampler=None, batch_size=1, num_workers=0, drop_last=False, pin=False):
    collate_fn=None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=drop_last,
        collate_fn=collate_fn,
        sampler=sampler,
    )
    return data_loader