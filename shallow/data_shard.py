import torch


class EmptyBatchSelection(Exception): pass


def generate_fix_permuted_inidices(n, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.arange(n)
    _sample = torch.randperm(n, generator=g)
    return indices[_sample]


class ShardedPreloadingDataset:
    def __init__(self, dataset, num_proc=False, progress=None, seed=0, rank=0, num_replicas=1, to_tensor=False, prepr_fn=None):
        """
            Preloading data into processes with respect to process idxs. Can load into one huge tensor, or python list
            WITH TO_TENSOR LAST UNFUL BATCH WILL BE DROPPED
        """
        self.dataset = dataset
        self.bs = 64 # preloading bs
        self.num_replicas = num_replicas
        drop_last = to_tensor
        dl = torch.utils.data.DataLoader(dataset, batch_size=self.bs, shuffle=False, drop_last=drop_last, num_workers=num_proc)
        if progress is not None: dl = progress(dl)

        if num_replicas > 1:
            total_size = len(dataset) - len(dataset) % num_replicas # much pythonic
            all_idxs = generate_fix_permuted_inidices(len(dataset), seed)
            self.chosen_idxs = all_idxs[rank:total_size:num_replicas]
        else:
            self.chosen_idxs = None

        self.prepr_fn = prepr_fn if prepr_fn is not None else lambda x:x
        preloader = self.preload_data_torch_tensor if to_tensor else self.preload_data
        self.data, self.labels = preloader(dl)

    def select_idxs(self, item, batch_idx):
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

    def preload_data(self, dl):
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

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]
