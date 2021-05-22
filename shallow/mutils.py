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

def _load_opt_state(model, path):
    path = Path(path)
    if path.suffix != '.pth': path = get_last_model_name(path)
    opt_state = torch.load(path)['opt_state']
    return opt_state

def _load_model_state(model, path):
    path = Path(path)
    if path.suffix != '.pth': path = get_last_model_name(path)
    state_dict = torch.load(path)['model_state']
    model.load_state_dict(state_dict)
    return model


def _init_encoder(model, src):
    enc_state = torch.load(src)['model_state']
    if "head.fc.weight" not in enc_state: 
        enc_state["head.fc.weight"] = None
        enc_state["head.fc.bias"] = None
    model.encoder.load_state_dict(enc_state)

class FoldModel(nn.Module):
    def __init__(self, models):
        super(FoldModel, self).__init__()
        self.ms = models
        
    def forward(self, x):
        res = torch.stack([m(x) for m in self.ms])
        return res.mean(0)

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

def get_last_model_name(src):
    # assumes that model name is of type e500_blabla.pth, sorted by epoch #500
    model_names = list(Path(src).glob('*.pth'))
    assert model_names != [], 'No valid models at init path'

    res = []
    for i, m in enumerate(model_names):
        epoch = parse_model_path(m)
        res.append([i,epoch])
    idx = sorted(res, key=lambda x: -x[1])[0][0]
    return model_names[idx]
