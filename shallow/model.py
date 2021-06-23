from functools import partial

import torch
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from loguru import logger

from shallow.utils.nn import _init_encoder, load_model_state, load_optim_state, _load_state
from shallow.utils.common import check_field_is_none


def model_select_example(model_id):
    MODELS = {
            'unet-regnety_016-scse': partial(smp.Unet, encoder_name='timm-regnety_016', decoder_attention_type='scse'),
            'unet-regnetx_032-scse': partial(smp.Unet, encoder_name='timm-regnetx_032', decoder_attention_type='scse')
            }
    model = MODELS[model_id]
    return model

def build_model(cfg, mod_select, MODEL=None):
    if MODEL is None : MODEL = cfg.TRAIN.MODEL
    model = mod_select(MODEL)()
    if cfg.TRAIN.INIT_MODEL: 
        logger.log('DEBUG', f'Init model: {cfg.TRAIN.INIT_MODEL}') 
        load_model_state(model, cfg.TRAIN.INIT_MODEL)

    elif not check_field_is_none(cfg.TRAIN.INIT_ENCODER):
        if cfg.TRAIN.FOLD_ID == '': enc_weights_name = cfg.TRAIN.INIT_ENCODER[0]
        else: enc_weights_name = cfg.TRAIN.INIT_ENCODER[cfg.TRAIN.FOLD_ID]
        _init_encoder(model, enc_weights_name)
    else: pass

    model = model.cuda()
    model.train()
    if cfg.TRAIN.TORCHSCRIPT: model = torch.jit.script(model)
    return model 

def get_optim(cfg, model):
    lr = 1e-4 if not cfg.PARALLEL.DDP else scale_lr(base_lr, cfg) 
    
    opt = optim.AdamW
    opt_kwargs = {'amsgrad':True, 'weight_decay':1e-3}
    optimizer = opt(tencent_trick(model), lr=lr, **opt_kwargs)
    if cfg.TRAIN.INIT_MODEL and cfg.TRAIN.INIT_OPT: 
        load_optim_state(optimizer, cfg.TRAIN.INIT_MODEL)
    return optimizer

def wrap_ddp(cfg, model, sync_bn=False, broadcast_buffers=True, find_unused_parameters=True):
    if sync_bn: 
        assert not cfg.TRAIN.TORCHSCRIPT
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    #logger.log('DEBUG', f'DDP wrapper, {model, cfg}') 
    if cfg.PARALLEL.DDP: 
        model = DistributedDataParallel(model, 
                                    device_ids=[cfg.PARALLEL.LOCAL_RANK],
                                    find_unused_parameters=find_unused_parameters,
                                    broadcast_buffers=broadcast_buffers)
    return model


