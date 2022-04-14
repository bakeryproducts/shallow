from functools import partial

import torch
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from loguru import logger

from shallow.utils.nn import load_model_state, load_optim_state


def model_select_example(model_id):
    MODELS = {
        'timm-unet-regnety_016-scse':
            partial(SomeUnet,
                    encoder_name='timm-regnety_016',
                    decoder_attention_type='scse'),
        'timm-unet-regnetx_032-scse':
            partial(SomeUnet,
                    encoder_name='timm-regnetx_032',
                    decoder_attention_type='scse')
    }
    model = MODELS[model_id]
    return model


def build_model(model_name, model_select, model_weights=None, weight_init_fn=load_model_state):
    """
        TODO: doc
    """
    model = model_select(model_name)()
    if model_weights:
        logger.log('DEBUG', f'Initializing weights from: {model_weights}')
        weight_init_fn(model, model_weights)
    model = model.cuda()
    model.train()
    model.name = model_name
    return model


def get_optim(cfg, model):
    lr = 1e-4 if not cfg.PARALLEL.DDP else scale_lr(base_lr, cfg) 
    opt = optim.AdamW
    opt_kwargs = {'amsgrad':True, 'weight_decay':1e-3}
    optimizer = opt(tencent_trick(model), lr=lr, **opt_kwargs)
    if cfg.TRAIN.INIT_MODEL and cfg.TRAIN.INIT_OPT: 
        load_optim_state(optimizer, cfg.TRAIN.INIT_MODEL)
    return optimizer


def wrap_ddp(cfg,
             model,
             sync_bn=False,
             broadcast_buffers=True,
             find_unused_parameters=True):
    if sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    #logger.log('DEBUG', f'DDP wrapper, {model, cfg}')
    if cfg.PARALLEL.DDP:
        model = DistributedDataParallel(
            model,
            device_ids=[cfg.PARALLEL.LOCAL_RANK],
            find_unused_parameters=find_unused_parameters,
            broadcast_buffers=broadcast_buffers)
    return model
