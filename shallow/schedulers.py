import numpy as np
from functools import partial

def annealer(f):
    def _inner(start, end): return partial(f, start, end)
    return _inner

@annealer
def sched_lin(start, end, pos): return start + pos*(end-start)
@annealer
def sched_cos(start, end, pos): return start + (1 + np.cos(np.pi*(1-pos))) * (end-start) / 2
@annealer
def sched_const(start, end, pos):  return start
@annealer
def sched_exp(start, end, pos): return start * (end/start) ** pos

def combine_scheds(scheds):
    pcts, fscheds = [], []
    for s in scheds: pcts.append(s[0]); fscheds.append(s[1])

    assert sum(pcts) == 1.
    pcts = np.array([0] + pcts)
    assert (pcts >= 0).all()
    pcts = np.cumsum(pcts, 0)
    def _inner(pos):
        idx = (pos >= pcts).nonzero()[0].max() #[0] for 0-th axis, pcts is 1d
        if idx == len(pcts)-1: idx -= 1
        actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])
        return fscheds[idx](actual_pos)
    return _inner
