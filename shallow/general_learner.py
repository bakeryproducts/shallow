from shallow import utils


class CancelFitException(Exception): pass
class CancelEpochException(Exception): pass


class Learner:
    def __init__(self, model, opt, dls, loss_func, lr, cbs, batch_bar, epoch_bar, **kwargs):
        utils.file_op.store_attr(self, locals())
        for cb in self.cbs: cb.L = self
        self.modes = list(dls.keys()) # dls {'TRAIN': dl, 'VAL':dl, ...}

    def one_batch(self):
        self('before_batch')
        self('step')
        self('after_batch')

    def one_epoch(self, mode):
        if 'train' in mode.lower(): self.model.training = True
        else: self.model.training = False

        self.dl = self.dls[mode]
        self('before_epoch')
        try:
            for self.n_batch, self.batch in enumerate(self.batch_bar(self.dl)):
                self.np_batch = self.n_batch / len(self.dl)
                self.one_batch()
        except CancelEpochException: pass
        self('after_epoch')

    def fit(self, total_epochs):
        self('before_fit')
        self.total_epochs = total_epochs
        try:
            for self.n_epoch in self.epoch_bar:
                self.np_epoch = self.n_epoch / self.total_epochs
                for self.mode in self.modes:
                    self.one_epoch(self.mode)
        except CancelFitException: pass
        self('after_fit')

    def __call__(self, name):
        for cb in self.cbs: getattr(cb, name, utils.common.noop)()
