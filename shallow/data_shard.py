import sys

import torch
from joblib import Parallel, delayed


class EmptyBatchSelection(Exception): pass


def generate_fix_permuted_inidices(n, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.arange(n)
    _sample = torch.randperm(n, generator=g)
    return indices[_sample]


class ShardedPreloadingDatasetCPU:
    def __init__(self, dataset, num_proc=False, progress=None, seed=0, rank=0, num_replicas=1, joblib=False, prepr_fn=None):
        """
            Preloading data into processes with respect to process idxs. Can load into one huge tensor, or python list
            WITH TO_TENSOR LAST UNFUL BATCH WILL BE DROPPED
        """
        self.dataset = dataset
        self.bs = 8 # preloading bs
        self.num_replicas = num_replicas
        self.num_proc = num_proc

        all_idxs = generate_fix_permuted_inidices(len(dataset), seed)
        if num_replicas > 1:
            total_size = len(dataset) - len(dataset) % num_replicas # much pythonic
            self.chosen_idxs = all_idxs[rank:total_size:num_replicas]
        else:
            self.chosen_idxs = all_idxs
        # print(rank, self.chosen_idxs[:8])
        # print(rank, sum(self.chosen_idxs))

        self.prepr_fn = prepr_fn if prepr_fn is not None else lambda x:x
        preloader = self.preload_data_joblib if joblib else self.simple_read
        self.data = preloader(progress)
        print('SHARDS LOADED')

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

    def simple_read(self, progress):
        data = []
        for i in progress(self.chosen_idxs):
            data.append(self.dataset[i])
        return data

    def get_some(self, dataset, sub_idxs):
        return [dataset[i] for i in sub_idxs]

    def preload_data_joblib(self, progress):
        idxs = self.chosen_idxs
        l = len(self.dataset)
        n = 200#l // self.num_proc
        splits = [idxs[i:i + n] for i in range(0, len(idxs), n)]
        jobs = []
        for s in splits:
            jobs.append(delayed(self.get_some)(self.dataset,s))

        data = Parallel(n_jobs=self.num_proc, verbose=5)(jobs)
        data = [i for d in data for i in d]
        return data



class ShardedPreloadingDatasetGPU:
    def __init__(self, dataset, num_proc=False, progress=None, seed=0, rank=0, num_replicas=1, to_tensor=False, prepr_fn=None):
        """
            Preloading data into processes with respect to process idxs. Can load into one huge tensor, or python list
            WITH TO_TENSOR LAST UNFUL BATCH WILL BE DROPPED
        """
        self.dataset = dataset
        self.bs = 8 # preloading bs
        self.num_replicas = num_replicas
        self.num_proc = num_proc
        drop_last = to_tensor
        # dl = torch.utils.data.DataLoader(dataset, batch_size=self.bs, shuffle=False, drop_last=drop_last, num_workers=num_proc)
        # if progress is not None: dl = progress(dl)

        if num_replicas > 1:
            total_size = len(dataset) - len(dataset) % num_replicas # much pythonic
            all_idxs = generate_fix_permuted_inidices(len(dataset), seed)
            self.chosen_idxs = all_idxs[rank:total_size:num_replicas]
        else:
            self.chosen_idxs = None

        self.prepr_fn = prepr_fn if prepr_fn is not None else lambda x:x
        preloader = self.simple_read_tensor if to_tensor else self.simple_read
        # preloader = self.preload_data_joblib
        # SPLIT idxs ? out of shmem
        # print(len(dataset), len(self.chosen_idxs))
        self.data = preloader(progress)
        # print(len(self.data), len(self.chosen_idxs), len(dataset))
        # print(self.data[0].keys())
        print('SHARDS LOADED')
        # print([d.shape for d in self.data])
        # print(sys.getsizeof(self.data[1].storage()))
        # print(self.data[0]['ann_data'].shape, 'PRELOADING SHARDS')


    def simple_read_tensor(self, progress):
        data = None
        n = len(self.chosen_idxs)
        for i, idx in enumerate(progress(self.chosen_idxs)):
            item = self.dataset[idx]
            tensors = self.prepr_fn(item)
            if data is None:
                data = [torch.zeros(n, *t.shape, dtype=t.dtype) for t in tensors]
            for j,t in enumerate(tensors):
                data[j][i] = t
        return data

    def __getitem__(self, idx): return [d[idx] for d in self.data]
    def __len__(self): return len(self.data[0])

    # def __len__(self): return len(self.data)
    # def __getitem__(self, idx): return self.data[idx]

    def simple_read(self, progress):
        data = []
        for i in progress(self.chosen_idxs):
            data.append(self.dataset[i])
        return data

    def get_some(self, dataset, sub_idxs):
        return [dataset[i] for i in sub_idxs]

    def preload_data_joblib(self, idxs):
        l = len(self.dataset)
        n = l // self.num_proc
        splits = [idxs[i:i + n] for i in range(0, len(idxs), n)]
        jobs = []
        for s in splits:
            jobs.append(delayed(self.get_some)(self.dataset,s))
        data = Parallel(n_jobs=self.num_proc)(jobs)
        data = [i for d in data for i in d]
        return data


    def _select_idxs(self, item, batch_idx):
        if self.chosen_idxs is None: return item
        idxs = range(batch_idx * self.bs, (batch_idx + 1) * self.bs)

        # TODO read_batch?
        xb, yb = item
        xx, yy = [], []
        for x,y,i in zip(xb,yb,idxs):
            if i in self.chosen_idxs:
                tx = x.clone()
                tx = self.prepr_fn(tx)
                xx.append(tx)
                yy.append(y.clone())
        if not xx: raise EmptyBatchSelection
        return (xx,yy)

    def select_idxs(self, item, batch_idx):
        if self.chosen_idxs is None: return item
        idxs = range(batch_idx * self.bs, (batch_idx + 1) * self.bs)

        # TODO read_batch?
        res = []
        for r,i in zip(item,idxs):
            if i in self.chosen_idxs:
                res.append(r)
        if not res: raise EmptyBatchSelection
        return res

    def preload_data(self, dl):
        R = []
        for batch_idx, item in enumerate(dl):
            try: rb = self.select_idxs(item, batch_idx)
            except EmptyBatchSelection: continue
            R.extend(rb)
        del dl
        return R

    def _preload_data(self, dl):
        X, Y = [], []
        for batch_idx, item in enumerate(dl):
            try: xb, yb = self.select_idxs(item, batch_idx)
            except EmptyBatchSelection: continue
            X.extend(xb)
            Y.extend(yb)
        del dl
        return X, Y

    def preload_data_torch_tensor(self, dl):
        #X, Y = None, None
        X,Y = [],[]
        for batch_idx, item in enumerate(dl):
            try:
                xb, yb = self.select_idxs(item, batch_idx)
                #xb = torch.stack(xb)
                #yb = torch.stack(yb)
                    #X = torch.zeros((len(dl) * self.bs, *xb[0].shape), dtype=xb[0].dtype)
                    #Y = torch.zeros((len(dl) * self.bs, *yb[0].shape), dtype=yb[0].dtype)

            except EmptyBatchSelection: continue
            X.extend(xb)
            Y.extend(yb)
            #if X is None:
            #    X = xb
            #    Y = yb
            #else:
            #    X = torch.vstack([X,xb])
            #    Y = torch.hstack([Y,yb]) # yb 1dim, TODO fix

            #X[batch_idx * self.bs:(batch_idx + 1) * self.bs] = torch.stack(xb)
            #Y[batch_idx * self.bs:(batch_idx + 1) * self.bs] = torch.stack(yb)
        X = torch.stack(X)
        Y = torch.stack(Y)
        del dl
        return X, Y

