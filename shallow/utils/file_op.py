import os
import json
import datetime
from pathlib import Path
from collections.abc import Iterable
from functools import partial, reduce

import numpy as np
from tqdm.auto import tqdm



def timestamp(): return '{:%Y_%b_%d_%H_%M_%S}'.format(datetime.datetime.now())

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

def jread(path):
    with open(str(path), 'r') as f:
        data = json.load(f)
    return data

def jdump(data, path):
    with open(str(path), 'w') as f:
        json.dump(data, f, indent=4)

def filter_ban_str_in_name(s, bans): return any([(b in str(s)) for b in bans])

def get_filenames(path, pattern, filter_out_func=lambda x: False):
    """
    pattern : "*.json"
    filter_out : function that return True if file name is acceptable
    """

    filenames = list(Path(path).glob(pattern))
    assert (filenames), f'There is no matching filenames for {path}, {pattern}'
    filenames = [fn for fn in filenames if not filter_out_func(fn)]
    assert (filenames), f'There is no matching filenames for {filter_out_func}'
    return filenames
