import os
import datetime
import multiprocessing as mp
from contextlib import contextmanager
from collections.abc import Iterable
from functools import partial, reduce

import torch
import numpy as np
from torchvision.transforms import ToPILImage
from tqdm.auto import tqdm


@contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    
def mp_func(foo, args, n):
    args_chunks = [args[i:i + n] for i in range(0, len(args), n)]
    with poolcontext(processes=n) as pool:
        res = pool.map(foo, args_chunks)
    return [ri for r in res for ri in r]


def mp_func_gen(foo, args, n, progress=None):
    args_chunks = [args[i:i + n] for i in range(0, len(args), n)]
    results = []
    with poolcontext(processes=n) as pool:
        gen = pool.imap(foo, args_chunks)
        if progress is not None: gen = progress(gen, total=len(args_chunks))
        for r in gen:
            results.extend(r)
    return results


def noop (x=None, *args, **kwargs): return x

def noops(self, x=None, *args, **kwargs): return x

def compose2(f, g):return lambda *a, **kw: f(g(*a, **kw))

def compose(*fs):return reduce(compose2, fs)

def tpi(i): return ToPILImage()(i)

def in_docker(): return os.path.exists('/.dockerenv')

def timestamp(): return '{:%Y_%b_%d_%H_%M_%S}'.format(datetime.datetime.now())

def st(t): return t.shape, t.dtype, t.min(), t.max(), t.mean(), t.std()

def to_cuda(l): return [i.cuda() for i in l]

def upscale(tensor, size): return torch.nn.functional.interpolate(tensor, size=size)

def denorm(images, mean=(0.46454108, 0.43718538, 0.39618185), std=(0.23577851, 0.23005974, 0.23109385)):
    mean = torch.tensor(mean).view((1,3,1,1))
    std = torch.tensor(std).view((1,3,1,1))
    images = images * std + mean
    return images

def get_cb_by_instance(cbs, cls):
    for cb in cbs:
        if isinstance(cb, cls): return cb
    return None

def unwrap_model(model): return model.module if hasattr(model, 'module') else model
def get_state_dict(model, unwrap_fn=unwrap_model): return unwrap_fn(model).state_dict()


def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]

def setify(o): return o if isinstance(o,set) else set(listify(o))

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def store_attr(self, ll):
    self.__dict__.update(ll)
    del self.__dict__['self']

def custom_dir(c, add:list):
    "Implement custom `__dir__`, adding `add` to `cls`"
    return dir(type(c)) + list(c.__dict__.keys()) + add

class GetAttr:
    "Inherit from this to have all attr accesses in `self._xtra` passed down to `self.default`"
    _default='default'
    def _component_attr_filter(self,k):
        if k.startswith('__') or k in ('_xtra',self._default): return False
        xtra = getattr(self,'_xtra',None)
        return xtra is None or k in xtra
    def _dir(self): return [k for k in dir(getattr(self,self._default)) if self._component_attr_filter(k)]
    def __getattr__(self,k):
        if self._component_attr_filter(k):
            attr = getattr(self,self._default,None)
            if attr is not None: return getattr(attr,k)
        raise AttributeError(k)
    def __dir__(self): return custom_dir(self,self._dir())
#     def __getstate__(self): return self.__dict__
    def __setstate__(self,data): self.__dict__.update(data)
    


class ListContainer():
    def __init__(self, items): self.items = listify(items)
    def __getitem__(self, idx):
        if isinstance(idx, (int,slice)): return self.items[idx]
        if isinstance(idx[0],bool):
            assert len(idx)==len(self) # bool mask
            return [o for m,o in zip(idx,self.items) if m]
        return [self.items[i] for i in idx]
    def __len__(self): return len(self.items)
    def __iter__(self): return iter(self.items)
    def __setitem__(self, i, o): self.items[i] = o
    def __delitem__(self, i): del(self.items[i])
    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
        if len(self)>10: res = res[:-1]+ '...]'
        return res



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
        self.count=0
        self.buffer.zero_()
        
def fig_to_array(fig):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1]+ (3,))
    return data

def on_master(f):
    def wrapper(*args):
        if args[0].kwargs['cfg'].PARALLEL.IS_MASTER:
            return f(*args)
    return wrapper

def on_epoch_step(f):
    def wrapper(*args):
        if (args[0].n_epoch % args[0].step) == 0:
            return f(*args)
    return wrapper

def on_train(f):
    def wrapper(*args):
        if args[0].model.training:
            return f(*args)
    return wrapper

def on_validation(f):
    def wrapper(*args):
        if not args[0].model.training:
            return f(*args)
    return wrapper

