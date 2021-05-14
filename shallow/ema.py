from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn


class ModelEmaV2(nn.Module):
    """ Model Exponential Moving Average V2

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.

    WONT WORK WITH SYNCBN AT THE MOMENT

    """
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for (ema_k, ema_v), (model_k, model_v) in zip(self.module.state_dict().items(), model.state_dict().items()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)

                #if 'bn' in ema_k: 
                #    ema_v.copy_(model_v) # SYNCBN
                #    continue
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model, decay=None):
        if decay is None: decay = self.decay
        self._update(model, update_fn=lambda e, m: decay * e + (1. - decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

