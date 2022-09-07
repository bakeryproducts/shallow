from pathlib import Path

import torch
import torch.nn as nn


def unwrap_model(model): return model.module if hasattr(model, 'module') else model
def get_state_dict(model, unwrap_fn=unwrap_model): return unwrap_fn(model).state_dict()
def get_model_name(model): return unwrap_model(model).name if hasattr(unwrap_model(model), 'name') else None
def scale_lr(lr, cfg): return lr * float(cfg.TRAIN.BATCH_SIZE * cfg.PARALLEL.WORLD_SIZE) / 256.


def _load_state(path, key):
    path = Path(path)
    if path.suffix != '.pth': path = get_last_model_name(path)
    state = torch.load(path, map_location='cpu')
    if isinstance(key, list):
        # multilevel dict : {'state':{'enc':s1, 'dec':s2, ...}} key = ['state', 'dec']
        for k in key:
            state = state.get(k, None)
    else:
        state = state.get(key, None)
    return state


def load_state(m, path, k, filt_fn=lambda x:x):
    st = _load_state(path, k)
    st = filt_fn(st)
    # m.load_state_dict(st, strict=False)
    m.load_state_dict(st)

def load_model_state(model, path): model.load_state_dict(_load_state(path, 'model_state'))
def load_optim_state(optim, path): optim.load_state_dict(_load_state(path, 'optim_state'))
def load_scaler_state(scaler, path): scaler.load_state_dict(_load_state(path, 'scaler_state'))
def unwrap_model(model): return model.module if hasattr(model, 'module') else model


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

    def forward(self, *args, **kwargs): return self.read_pred(self.model(*args, **kwargs))


class FoldModel(nn.Module):
    def __init__(self, models, read_pred=lambda x:x, write_pred=lambda x:x):
        super(FoldModel, self).__init__()
        self.ms = models
        self.read_pred = read_pred
        self.write_pred = write_pred

    def forward(self, x):
        res = torch.stack([self.read_pred(m(x)) for m in self.ms])
        preds = []
        for m in self.ms:
            pred = m(x)
            preds.append(pred)

        res = {}
        for k in preds[0]:
            p = [p[k] for p in preds]
            p = torch.stack([p])
            p = p.mean(0)
            res[k] = p

        #res = torch.stack([self.read_pred(m(x)) for m in self.ms])
        return res


def configure_optim_groups(model):
    """
        https://github.com/karpathy/minGPT/commit/bbbdac74fa9b2e55574d70056163ffbae42310c1
    """
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
    # special case the position embedding parameter in the root GPT module as not decayed
    [no_decay.add(i) for i in ('module.encoder.model.cls_token', 'module.encoder.model.patch_embed.proj.weight', 'module.encoder.model.pos_embed')]


    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )
    return [
        {"params": [param_dict[pn] for pn in sorted(list(decay))]},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0}
    ]


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


def get_last_model_name(src, after_epoch=False, keyword=''):
    # assumes that model name is of type e500_blabla.pth, sorted by epoch #500
    model_names = list(Path(src).glob('*.pth'))
    if keyword: model_names = [m for m in model_names if keyword in str(m.name)]
    assert model_names != [], f'No valid models at init path {src}'

    res = []
    for i, m in enumerate(model_names):
        if not after_epoch and 'after_epoch' in str(m): continue
        epoch = parse_model_path(m)
        res.append([i,epoch])
    idx = sorted(res, key=lambda x: -x[1])[0][0]
    return model_names[idx]




def avg_sq_ch_mean(model, input, output):
    "calculate average channel square mean of output activations"
    return torch.mean(output.mean(axis=[0,2,3])**2).item()


def avg_ch_var(model, input, output):
    "calculate average channel variance of output activations"
    return torch.mean(output.var(axis=[0,2,3])).item()


def avg_ch_var_residual(model, input, output):
    "calculate average channel variance of output activations"
    return torch.mean(output.var(axis=[0,2,3])).item()


class ActivationStatsHook:
    """Iterates through each of `model`'s modules and matches modules using unix pattern 
    matching based on `hook_fn_locs` and registers `hook_fn` to the module if there is 
    a match. 

    Arguments:
        model (nn.Module): model from which we will extract the activation stats
        hook_fn_locs (List[str]): List of `hook_fn` locations based on Unix type string 
            matching with the name of model's modules. 
        hook_fns (List[Callable]): List of hook functions to be registered at every
            module in `layer_names`.
    
    Inspiration from https://docs.fast.ai/callback.hook.html.

    Refer to https://gist.github.com/amaarora/6e56942fcb46e67ba203f3009b30d950 for an example 
    on how to plot Signal Propogation Plots using `ActivationStatsHook`.
    """

    def __init__(self, model, hook_fn_locs, hook_fns):
        self.model = model
        self.hook_fn_locs = hook_fn_locs
        self.hook_fns = hook_fns
        if len(hook_fn_locs) != len(hook_fns):
            raise ValueError("Please provide `hook_fns` for each `hook_fn_locs`, \
                their lengths are different.")
        self.stats = dict((hook_fn.__name__, []) for hook_fn in hook_fns)
        for hook_fn_loc, hook_fn in zip(hook_fn_locs, hook_fns): 
            self.register_hook(hook_fn_loc, hook_fn)

    def _create_hook(self, hook_fn):
        def append_activation_stats(module, input, output):
            out = hook_fn(module, input, output)
            self.stats[hook_fn.__name__].append(out)
        return append_activation_stats
        
    def register_hook(self, hook_fn_loc, hook_fn):
        for name, module in self.model.named_modules():
            if not fnmatch.fnmatch(name, hook_fn_loc):
                continue
            module.register_forward_hook(self._create_hook(hook_fn))


def extract_spp_stats(model,
                      hook_fn_locs,
                      hook_fns,
                      input_shape=[8, 3, 224, 224]):
    """Extract average square channel mean and variance of activations during
    forward pass to plot Signal Propogation Plots (SPP).

    Paper: https://arxiv.org/abs/2101.08692

    Example Usage: https://gist.github.com/amaarora/6e56942fcb46e67ba203f3009b30d950
    """
    x = torch.normal(0., 1., input_shape)
    hook = ActivationStatsHook(model, hook_fn_locs=hook_fn_locs, hook_fns=hook_fns)
    _ = model(x)
    return hook.stats
