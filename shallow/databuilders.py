def check_field_is_none(field):
    if isinstance(field, dict):
        for _, val in field.items():
            if val != (0,): return False 
    elif isinstance(field, list):
        if field != (0,): return False 

    return True


def create_datasets(cfg, all_datasets, dataset_types):
    """
        Joins lists of datasets with TRAIN, VALID, ... types into concated datasets:
            {
                "TRAIN": ConcatDataset1, | or in folds mode : "TRAIN": [FoldDs1_1, FOlds1_2, ...]
                "VALID": ConcatDataset2,
                ...
            }
    """
    converted_datasets = {}
    for dataset_type in dataset_types:
        data_field = cfg.DATA[dataset_type]
        if data_field.DATASETS != (0,):
            assert check_field_is_none(data_field.FOLDS)
            datasets = [all_datasets[ds] for ds in data_field.DATASETS]
            ds = TorchConcatDataset(datasets) if len(datasets)>1 else datasets[0] 
            converted_datasets[dataset_type] = ds

        elif not check_field_is_none(data_field.FOLDS)
            assert check_field_is_none(data_field.DATASETS)
            datasets = {}
            for fold_id, fold_datasets_ids in data_field.FOLDS.items():
                fold_datasets = [all_datasets[fold_dataset_id] for fold_dataset_id in fold_datasets_ids]
                ds = TorchConcatDataset(fold_datasets) if len(fold_datasets)>1 else fold_datasets[0] 
                datasets[fold_id] = ds

            converted_datasets[dataset_type] = datasets
        else:
            raise Exception('INVALID DATASET/FOLDS cfg')

    return converted_datasets


def build_datasets(cfg, mode_train=True, num_proc=4, dataset_types=['TRAIN', 'VALID', 'TEST'], fold_id=None):
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
            augmented_dataset = TransformDataset(preload_dataset)
    """


    def train_trans_get(*args, **kwargs): return augs.get_aug(*args, **kwargs)

    transform_factory = {
            'TRAIN':{'factory':TransformDataset, 'transform_getter':train_trans_get},
            'VALID':{'factory':TransformDataset, 'transform_getter':train_trans_get},
            'VALID2':{'factory':TransformDataset, 'transform_getter':train_trans_get},
            'TEST':{'factory':TransformDataset, 'transform_getter':train_trans_get},
        }

    extend_factories = {
             'PRELOAD':partial(PreloadingDataset, num_proc=num_proc, progress=tqdm),
             'MULTIPLY':MultiplyDataset,
             'CACHING':CachingDataset,
    }
 
    # TODO create_datasets should take fold_id as an input 
    datasets = create_datasets(cfg, init_datasets(cfg), dataset_types)

    if fold_id is not None:
        # fold mode, fold_id from cfg, F0, F1, ...
        assert isinstance(datasets['TRAIN'], dict) # fold mode, train->folds->{f0:[], f1:[], f2:[]}
        datasets['TRAIN'] = datasets['TRAIN'][fold_id] 
        datasets['VALID'] = datasets['VALID'][fold_id] 

    if cfg.TRANSFORMERS.STD == (0,) and cfg.TRANSFORMERS.MEAN == (0,):
        mean, std = mean_std_dataset(datasets['TRAIN'])
        update_mean_std(cfg, mean, std)

    datasets = create_extensions(cfg, datasets, extend_factories)
    transforms = create_transforms(cfg, transform_factory, dataset_types)
    datasets = apply_transforms_datasets(datasets, transforms)
    return datasets


def create_transforms(cfg, transform_factories, dataset_types=['TRAIN', 'VALID', 'TEST']):
    transformers = {}
    for dataset_type in dataset_types:
        aug_type = cfg.TRANSFORMERS[dataset_type]['AUG']
        args={
            'aug_type':aug_type,
            'transforms_cfg':cfg.TRANSFORMERS
        }
        if transform_factories[dataset_type]['factory'] is not None:
            transform_getter = transform_factories[dataset_type]['transform_getter'](**args)
            transformer = partial(transform_factories[dataset_type]['factory'], transforms=transform_getter)
        else:
            transformer = lambda x: x
        transformers[dataset_type] = transformer
    return transformers    

def apply_transforms_datasets(datasets, transforms):
    return {dataset_type:transforms[dataset_type](dataset) for dataset_type, dataset in datasets.items()}


def build_dataloaders(cfg, datasets, selective=False):
    dls = {}
    for kind, dataset in datasets.items():
        dls[kind] = build_dataloader(cfg, dataset, kind, selective=selective)
    return dls

def build_dataloader(cfg, dataset, mode, selective):
    drop_last = True
    sampler = None 

    if cfg.PARALLEL.DDP and (mode == 'TRAIN' or mode == 'SSL'):
        if sampler is None:
            sampler = DistributedSampler(dataset, num_replicas=cfg.PARALLEL.WORLD_SIZE, rank=cfg.PARALLEL.LOCAL_RANK, shuffle=True)

    num_workers = cfg.TRAIN.NUM_WORKERS 
    shuffle = sampler is None

    dl = DataLoader(
        dataset,
        batch_size=cfg[mode]['BATCH_SIZE'],
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=None,
        sampler=sampler,)
    return dl


