from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import optim
from torch.nn import init
import torch.nn.functional as F


def scale_lr(lr, cfg): return lr * float(cfg.TRAIN.BATCH_SIZE * cfg.PARALLEL.WORLD_SIZE)/256.

def load_model(cfg, model_folder_path, eval_mode=True):
    print(model_folder_path)
    model = model_select(cfg.TRAIN.MODEL)
    model = _load_model_state(model, model_folder_path)
    if eval_mode: model.eval()
    return model

def _load_state(path, key):
    path = Path(path)
    if path.suffix != '.pth': path = get_last_model_name(path)
    state = torch.load(path, map_location='cpu').get(key, None)
    return state

def load_model_state(model, path): model.load_state_dict(_load_state(path, 'model_state'))
def load_optim_state(optim, path): optim.load_state_dict(_load_state(path, 'optim_state'))
def load_scaler_state(scaler, path): scaler.load_state_dict(_load_state(path, 'scaler_state'))


def _init_encoder(model, src):
    enc_state = torch.load(src)['model_state']
    if "head.fc.weight" not in enc_state: 
        enc_state["head.fc.weight"] = None
        enc_state["head.fc.bias"] = None
    model.encoder.load_state_dict(enc_state)

class ModelUnwrap(nn.Module):
    def __init__(self, model, read_pred):
        super(ModelUnwrap, self).__init__()
        self.model = model
        self.read_pred = read_pred
        
    def forward(self, x): return self.read_pred(self.model(x))

class FoldModel(nn.Module):
    def __init__(self, models, read_pred=lambda x:x, write_pred=lambda x:x):
        super(FoldModel, self).__init__()
        self.ms = models
        self.read_pred = read_pred
        self.write_pred = write_pred
        
    def forward(self, x):
        res = torch.stack([self.read_pred(m(x)) for m in self.ms])
        return self.write_pred(res.mean(0))

def tencent_trick(model):
    """
    Divide parameters into 2 groups.
    First group is BNs and all biases.
    Second group is the remaining model's parameters.
    Weight decay will be disabled in first group (aka tencent trick).
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0},
            {'params': decay}]

def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if m.in_channels == 3: nn.init.normal_(m.weight, 0, 0.21)
            if isinstance(m, nn.Conv2d) : nn.init.kaiming_uniform_(m.weight, a=.1, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.ConvTranspose2d): nn.init.kaiming_uniform_(m.weight, a=.1, mode='fan_in', nonlinearity='leaky_relu')#nn.init.xavier_uniform_(m.weight, 0.1)
            if m.bias is not None: m.bias.data.zero_()
    if hasattr(model, '_layer_init'):
        model._layer_init()

def replace_relu_to_silu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.SiLU(inplace=True))
        else:
            replace_relu_to_silu(child)

def parse_model_path(p):
    name = str(p.name)
    epoch = name.split('_')[0]
    return int(epoch[1:])

def get_last_model_name(src, after_epoch=False):
    # assumes that model name is of type e500_blabla.pth, sorted by epoch #500
    model_names = list(Path(src).glob('*.pth'))
    assert model_names != [], 'No valid models at init path'

    res = []
    for i, m in enumerate(model_names):
        if not after_epoch and 'after_epoch' in str(m): continue
        epoch = parse_model_path(m)
        res.append([i,epoch])
    idx = sorted(res, key=lambda x: -x[1])[0][0]
    return model_names[idx]
