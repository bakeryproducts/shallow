from pathlib import Path
from joblib import Parallel, delayed
from functools import lru_cache, partial

import torch
from PIL import Image
import albumentations as albu
from torch.utils.data.dataset import ConcatDataset as ConcatDataset



class Dataset:
    def __init__(self, root, pattern, recursive=False):
        self.root = Path(root)
        self.pattern = pattern
        _files = self.root.rglob(self.pattern) if recursive else self.root.glob(self.pattern)
        self.files = sorted(list(_files))
        self._is_empty('There is no matching files!')

    def apply_filter(self, filter_fn):
        self.files = filter_fn(self.files)
        self._is_empty()

    def _is_empty(self, msg='There is no item in dataset!'): assert len(self.files) > 0
    def __len__(self): return len(self.files)
    def __getitem__(self, idx): return self.process_item(self.load_item(idx))
    def load_item(self, idx): raise NotImplementedError
    def process_item(self, item): return item
#     def __add__(self, other):
#         return ConcatDataset([self, other])


class ImageDataset(Dataset):
    def load_item(self, idx):
        img_path = self.files[idx]
        img = Image.open(str(img_path))
        return img


class PairDataset:
    def __init__(self, ds1, ds2):
        self.ds1, self.ds2 = ds1, ds2
        self.check_len()

    def __len__(self): return len(self.ds1)
    def check_len(self): assert len(self.ds1) == len(self.ds2)

    def __getitem__(self, idx):
        return self.ds1.__getitem__(idx), self.ds2.__getitem__(idx)


class FoldDataset:
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self): return len(self.idxs)
    def __getitem__(self, idx): return self.dataset[self.idxs[idx]]


class AugDataset:
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
            return self.transforms(image=item[0])['image']

    def __len__(self):
        return len(self.dataset)


class MultiplyDataset:
    def __init__(self, dataset, rate):
        if rate > 1:
            _dataset = ConcatDataset([dataset])
            for i in range(rate-1):
                _dataset += ConcatDataset([dataset])
            self.dataset = _dataset
        else:
            self.dataset = dataset

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
    def __init__(self, dataset, num_proc=1, progress=None):
        self.dataset = dataset
        self.num_proc = num_proc
        self.progress = progress
        # self.data = self.preload_data_torch()
        self.data = self.preload_data_joblib()

    def get_some(self, dataset, sub_idxs):
        return [dataset[i] for i in sub_idxs]

    def preload_data_joblib(self):
        l = len(self.dataset)
        idxs = range(l)
        n = l // self.num_proc
        splits = [idxs[i:i + n] for i in range(0, len(idxs), n)]
        jobs = []
        for s in splits:
            jobs.append(delayed(self.get_some)(self.dataset,s))
        data = Parallel(n_jobs=self.num_proc)(jobs)
        data = [i for d in data for i in d]
        return data

    def preload_data_torch(self):
        dl = torch.utils.data.DataLoader(self.dataset, batch_size=64, drop_last=False, num_workers=self.num_proc, prefetch_factor=1)
        data = []
        if self.progress is not None and not self.num_proc: dl = self.progress(dl)

        for item in dl:
            if isinstance(item, tuple) or isinstance(item, list):
                xb, yb = item
                for x,y in zip(xb.numpy(), yb.numpy()):
                    data.append([x,y])
            elif isinstance(item, dict):
                ds = [{} for _ in range(len(list(item.values())[0]))]

                for k, vs in item.items():
                    for i, v in enumerate(vs):
                        try:
                            ds[i][k] = v.numpy()
                        except:
                            ds[i][k] = v
                for d in ds:
                    data.append(d)

            else:
                # print(item)
                for x in item.numpy():
                    data.append(x)

        del dl
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


def extend_dataset(ds, data_field, extend_factories):
    for k, factory in extend_factories.items():
        field_val = data_field.get(k, None)
        if field_val:
            args = {}
            try: args.update(field_val)
            except:pass
            ds = factory(ds, **args)
    return ds


def create_extensions(cfg, datasets, extend_factories):
    extended_datasets = {}
    for kind, ds in datasets.items():
        extended_datasets[kind] = extend_dataset(ds, cfg.DATA[kind], extend_factories)
    return extended_datasets


def create_augmentators(cfg, aug_factories, dataset_types=['TRAIN', 'VALID', 'TEST']):
    augmentators = {}
    for dataset_type in dataset_types:
        aug_type = cfg.AUGS[dataset_type]['AUG']
        args = {
            'aug_type':aug_type,
            'aug_cfg':cfg.AUGS
        }
        if aug_factories[dataset_type]['factory'] is not None:
            aug_getter = aug_factories[dataset_type]['aug_getter'](**args)
            augmentator = partial(aug_factories[dataset_type]['factory'], transforms=aug_getter)
        else:
            augmentator = lambda x: x
        augmentators[dataset_type] = augmentator
    return augmentators


def apply_augs_datasets(datasets, augmentators):
    return {dataset_type:augmentators[dataset_type](dataset) for dataset_type, dataset in datasets.items()}


def count_folds(cfg):
    n = 0
    if 'SPLITS' in cfg:
        for sid, dataset_idxs in cfg.SPLITS.items():
            if dataset_idxs != (0,): n+=1
    return n
