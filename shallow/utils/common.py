import os
import functools

import torch
import numpy as np
from torchvision.transforms import ToPILImage


def noop(x=None, *args, **kwargs): return x
def noops(self, x=None, *args, **kwargs): return x
def tpi(i): return ToPILImage()(i)
def in_docker(): return os.path.exists('/.dockerenv')
def to_cuda(l): return [i.cuda() for i in l]
def upscale(tensor, size): return torch.nn.functional.interpolate(tensor, size=size)
def denorm(images, mean=(0.46454108, 0.43718538, 0.39618185), std=(0.23577851, 0.23005974, 0.23109385), num_ch=3):
    mean = torch.tensor(mean).view((1,num_ch,1,1))
    std = torch.tensor(std).view((1,num_ch,1,1))
    images = images * std + mean
    return images


def st(t, tol=3):
    """
        shape, dtype, min, max, mean, std
    """
    stats = t.min(), t.max(), t.mean(), t.std()
    return list(t.shape), t.dtype, [round(i.tolist(), tol) for i in stats]


def set_cuda_devices(gpu_idx):
    if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
        gpus = ','.join([str(g) for g in gpu_idx])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    else:
        print(f'WARNING, GPU OS AND CFG CONFLICT: ', cfg.TRAIN.GPUS, os.environ.get('CUDA_VISIBLE_DEVICES'))
        print('USING ', os.environ.get('CUDA_VISIBLE_DEVICES'))


class TorchBuffer:
    # TODO convert to torch.Tensor extension
    def __init__(self, shape=(1,), device=torch.device('cpu'), max_len=200):
        self.shape = shape
        self.count = 0
        self.max_len = max_len
        self.enlarge_factor = 2
        self.device = device
        self.buffer = torch.zeros((self.max_len, *self.shape)).to(self.device)

    def enlarge_buffer(self):
        self.max_len = int(self.max_len * self.enlarge_factor)
        #print(f'BUFFER GROWS TO {self.max_len}')
        self.new_buffer = torch.zeros((self.max_len, *self.shape)).to(self.device)
        self.new_buffer[:self.count] = self.buffer[:self.count]
        self.buffer = self.new_buffer

    def push(self, t):
        if self.count > .9 * self.max_len: self.enlarge_buffer()
        self.buffer[self.count,...] = t
        self.count += 1

    @property
    def data(self): return self.buffer[:self.count]

    def reset(self):
        self.count = 0
        self.buffer.zero_()


def fig_to_array(fig):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def check_field_is_none(base, name):
    field = base.get(name, None)
    if not field: return True

    if isinstance(field, dict):
        for _, val in field.items():
            if val == (0,) or val == '': return True
    elif isinstance(field, list):
        if field == (0,) and val == '': return True

    return False


def return_first_call(func):
    value = None

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        nonlocal value
        if value is None:
            value = func(*args, **kwargs)
            func.value = value
            func._clear = _clear
        return value

    def _clear():
        nonlocal value
        value = None

    return wrapper_decorator


def run_once(func):
    def _clear():
        nonlocal callers
        callers = {}

    callers = {}
    func._clear = _clear

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        nonlocal callers
        caller = args[0]

        if caller not in callers:
            value = func(*args, **kwargs)
            callers[caller] = value
        return callers[caller]

    return wrapper_decorator

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
