from pathlib import Path
from functools import partial
from collections import OrderedDict
from logger import logger

import torch
import torch.nn as nn
from torch import optim
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from mutils import *


def model_select(model_id):
    MODELS = {
            'unet-regnety_016-scse': partial(smp.Unet, encoder_name='timm-regnety_016', decoder_attention_type='scse')
            'unet-regnetx_032-scse': partial(smp.Unet, encoder_name='timm-regnetx_032', decoder_attention_type='scse')
            }
    model = MODELS[model_id]
    return model

def build_model(cfg):
    model = model_select(cfg.TRAIN.MODEL)()
    if cfg.TRAIN.INIT_MODEL: 
        logger.log('DEBUG', f'Init model: {cfg.TRAIN.INIT_MODEL}') 
        model = _load_model_state(model, cfg.TRAIN.INIT_MODEL)
    elif cfg.TRAIN.INIT_ENCODER != (0,): 
        if cfg.TRAIN.FOLD_IDX == -1: enc_weights_name = cfg.TRAIN.INIT_ENCODER[0]
        else: enc_weights_name = cfg.TRAIN.INIT_ENCODER[cfg.TRAIN.FOLD_IDX]
        _init_encoder(model, enc_weights_name)
    else: pass

    model = model.cuda()
    model.train()
    return model 

def get_optim(cfg, model):
    base_lr = 1e-4# should be overriden in LR scheduler anyway
    lr = base_lr if not cfg.PARALLEL.DDP else scale_lr(base_lr, cfg) 
    
    opt = optim.AdamW
    opt_kwargs = {'amsgrad':True, 'weight_decay':1e-3}
    optimizer = opt(tencent_trick(model), lr=lr, **opt_kwargs)
    if cfg.TRAIN.INIT_MODEL: 
        st =  _load_opt_state(model, cfg.TRAIN.INIT_MODEL)
        optimizer.load_state_dict(st)
    return optimizer

def wrap_ddp(cfg, model, sync_bn=False, broadcast_buffers=True, find_unused_parameters=True):
    if sync_bn: model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if cfg.PARALLEL.DDP: 
        model = DistributedDataParallel(model, 
                                    device_ids=[cfg.PARALLEL.LOCAL_RANK],
                                    find_unused_parameters=find_unused_parameters,
                                    broadcast_buffers=broadcast_buffers)
    return model


