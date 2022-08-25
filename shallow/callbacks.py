from functools import partial

import torch

from shallow import utils, meters


class CancelFitException(Exception): pass


class Callback:
    _default = 'L'
    logger = None
    def log_debug(self, *m): [self.logger.debug(i) for i in m] if self.logger is not None else False
    def log_warning(self, *m): [self.logger.warning(i) for i in m] if self.logger is not None else False
    def log_info(self, *m): [self.logger.info(i) for i in m] if self.logger is not None else False
    def log_error(self, *m): [self.logger.error(i) for i in m] if self.logger is not None else False
    def log_critical(self, *m): [self.logger.critical(i) for i in m] if self.logger is not None else False


class ParamSchedulerCB(Callback):
    def __init__(self, phase, pname, sched_func): # before_fit, lr, lin_sch(start, end, pos)
        self.pname, self.sched_func = pname, sched_func
        setattr(self, phase, self.set_param) # i.e. def before_epoch

    def set_param(self): setattr(self.L, self.pname, self.sched_func(self.L.np_epoch))


def batch_transform_cuda(b):
    xb,yb = b
    return {'xb': xb.cuda(), 'yb': yb.cuda()}


class SetupLearnerCB(Callback):
    def __init__(self, batch_transform=batch_transform_cuda): utils.file_op.store_attr(self, locals())

    def before_fit(self):
        self.cfg = self.L.kwargs['cfg']
        self.L.model.cuda()

    def before_batch(self): self.L.batch = self.batch_transform(self.L.batch)


class LRFinderCB(Callback):
    def before_fit(self):
        self.losses, self.lrs = [], []
        self.learner.lr = 1e-6

    def before_batch(self):
        if not self.model.training: return
        self.learner.lr *= 1.2
        print(self.lr)

    def after_batch(self):
        if not self.model.training: return
        if self.lr > 1 or torch.isnan(self.loss): raise CancelFitException
        self.losses.append(self.loss.item())
        self.lrs.append(self.lr)


class TimerCB(Callback):
    def __init__(self, Timer=None, mode_train=False, logger=None):
        self.logger = logger
        self.mode_train = mode_train
        self.perc = 90
        if Timer is None: Timer = meters.StopwatchMeter
        self.batch_timer = Timer()
        self.epoch_timer = Timer()

    def _before_batch(self):
        if self.L.model.training == self.mode_train: self.batch_timer.start()

    def before_epoch(self):
        if self.L.model.training == self.mode_train: self.epoch_timer.start()

    def _after_batch(self):
        if self.L.model.training == self.mode_train: self.batch_timer.stop()

    def after_epoch(self):
        if self.L.model.training == self.mode_train:
            self.epoch_timer.stop()
            bs, es = self.L.dl.batch_size, len(self.L.dl)
            if bs is None:
                try:
                    bs = self.L.kwargs['cfg'].TRAIN.BATCH_SIZE
                except:
                    bs = -1

            self.log_info(f'\t[E {self.L.n_epoch} / {self.L.total_epochs}]: {self.epoch_timer.last: .3f} s,' + f'{bs * es/self.epoch_timer.last: .3f} im/s; ')
            #f'batch {self.batch_timer.avg: .3f} s'   )
            self.batch_timer.reset()

    def after_fit(self):
        if self.L.model.training == self.mode_train:
            et = self.epoch_timer
            em = et.avg
            estd = ((et.p(self.perc) - em) + (em - et.p(1-self.perc))) / 2
            self.log_info(f'\tEpoch average time: {em: .3f} +- {estd: .3f} s')


class MemChLastCB(Callback):
    def __init__(self, batch_read=lambda x: x, logger=None, step=1):
        utils.file_op.store_attr(self, locals())

    def before_fit(self):
        self.log_debug('USING MEM CHANNELS LAST CALLBACK')
        self.cfg = self.L.kwargs['cfg']
        self.L.model = self.L.model.to(memory_format=torch.channels_last)
        #if self.cfg.TRAIN.EMA > 0: self.L.model_ema = self.L.model_ema.to(memory_format=torch.channels_last)

    def before_batch(self):
        #self.L.batch = [i.to(memory_format=torch.channels_last) for i in self.batch_read(self.L.batch)]
        self.L.batch['xb'].to(memory_format=torch.channels_last)


class FrozenEncoderCB(Callback):
    def __init__(self, np_epoch, encoder, logger=None, leave_head=False):
        utils.file_op.store_attr(self, locals())
        self.enc_frozen = True

    def before_fit(self):
        # utils.nn.unwrap_model(self.L.model).encoder.requires_grad_(False)
        self.encoder.requires_grad_(False)

    def before_epoch(self):
        if self.enc_frozen and self.L.np_epoch > self.np_epoch:
            self.enc_frozen = False
            self.log_warning(f'UNFREEZING ENCODER at {self.L.np_epoch}')
            self.encoder.requires_grad_(True)


class TBMetricCB(Callback):
    def __init__(self, writer, track_cb, train_metrics=None, validation_metrics=None, logger=None):
        ''' train_metrics = {'losses':['train_loss', 'val_loss']}
            val_metrics = {'metrics':['localization_f1']}
        '''
        utils.file_op.store_attr(self, locals())

    def before_fit(self): self.cfg = self.L.kwargs['cfg']

    def parse_metrics(self, metric_collection, training=True):
        mode = ''#'train_' if training else 'valid_'
        if metric_collection is None: return
        for category, metrics in metric_collection.items():
            for metric_name in metrics:
                metric_value = getattr(self.track_cb, metric_name, None)
                if metric_value is not None:
                    self.log_debug(f"{category + '/' + metric_name, metric_value, self.L.n_epoch}")
                    self.writer.add_scalar(category + '/' + mode + metric_name, metric_value, self.L.n_epoch)

    def after_epoch_train(self):
        #self.log_debug('tb metric after train epoch')
        self.parse_metrics(self.train_metrics, training=True)

    def after_epoch_valid(self):
        #self.log_debug('tb metric after validation')
        self.parse_metrics(self.validation_metrics, training=False)

    def after_epoch(self):
        if self.L.model.training: self.after_epoch_train()
        else: self.after_epoch_valid()
        self.writer.flush()


class TBPredictionsCB(Callback):
    def __init__(self, writer, batch_read=lambda x: x, denorm=utils.common.denorm, upscale=utils.common.upscale, logger=None, step=1):
        utils.file_op.store_attr(self, locals())
        self.num_images, self.wh = 5, (256, 256)

    def before_fit(self):
        self.mean, self.std = self.L.kwargs['cfg'].TRANSFORMERS.MEAN, self.L.kwargs['cfg'].TRANSFORMERS.STD

    def process_batch(self):
        batch = self.batch_read(self.L.batch)
        xb, yb = batch['xb'], batch['yb']
        preds = self.L.preds
        num_channels = 1

        xb = xb[:self.num_images]
        yb = yb[:self.num_images].repeat(1,3,1,1)
        preds = torch.sigmoid(preds[:self.num_images, ...])

        xb = self.denorm(xb, self.mean, self.std)
        xb = self.upscale(xb, self.wh)

        yb = self.upscale(yb, self.wh)

        preds = preds.max(1, keepdim=True)[0].repeat(1,3,1,1)
        preds = self.upscale(preds, self.wh)

        return xb, yb, preds
    
    def process_write_predictions(self):
        #self.log_debug('tb predictions')
        xb, yb, preds = self.process_batch() # takes last batch that been used
        #self.log_debug(f"{xb.shape}, {yb.shape}, {preds.shape}")
        summary_image = torch.cat([xb,yb,preds])
        #self.log_debug(f"{summary_image.shape}")
        grid = torchvision.utils.make_grid(summary_image, nrow=self.num_images, pad_value=4)
        label = 'train predictions' if self.L.model.training else 'val_predictions'
        self.writer.add_image(label, grid, self.L.n_epoch)
        self.writer.flush()

    def after_epoch(self):
        if not self.L.model.training or self.L.n_epoch % self.step == 0:
            self.process_write_predictions()


class TrainCB(Callback):
    def __init__(self, batch_read=lambda x: x, logger=None):
        utils.file_op.store_attr(self, locals())

    @utils.call.on_train
    def after_epoch(self): pass

    def before_fit(self):
        self.cfg = self.L.kwargs['cfg']
        self.amp = self.cfg.TRAIN.AMP
        if self.amp: self.amp_scaler = torch.cuda.amp.GradScaler()

    @utils.call.on_train
    def before_epoch(self):
        if self.cfg.PARALLEL.DDP: self.L.dl.sampler.set_epoch(self.L.n_epoch)
        for i in range(len(self.L.opt.param_groups)):
            self.L.opt.param_groups[i]['lr'] = self.L.lr 

    def train_step(self):
        batch = self.batch_read(self.L.batch)
        xb, yb = batch['xb'], batch['yb']
        preds = self.L.model(xb)
        self.L.preds = preds

        with torch.cuda.amp.autocast(enabled=self.amp):
            loss = self.L.loss_func(self.L.preds['output'], yb)

        if self.amp:
            l = self.amp_scaler.scale(loss)
            l.backward()
            self.amp_scaler.step(self.L.opt)
            self.amp_scaler.update()
        else:
            self.L.loss = loss
            self.L.loss.backward()
            self.L.opt.step()

        self.L.opt.zero_grad(set_to_none=True)
        if self.cfg.TRAIN.EMA: self.L.model_ema.update(self.L.model)


class CheckpointCB(Callback):
    # TODO simplier, **kwargs into save_dict, do we really need to do that
    def __init__(self, save_path, ema=False, save_step=None):
        utils.file_op.store_attr(self, locals())
        self.pct_counter = None if isinstance(self.save_step, int) else self.save_step

    def _save_dict_example(self, **kwargs):
        m = self.L.model_ema if kwargs.get('save_ema') else self.L.model
        name = m.name if hasattr(m, 'name') else None
        model_state_dict = utils.nn.get_state_dict(m)
        amp_scaler = self.L.amp_scaler if hasattr(self.L, 'amp_scaler') else None
        scaler_state = amp_scaler.state_dict() if amp_scaler is not None else None
        sd = {
            'epoch': self.L.n_epoch,
            'loss': self.L.loss,
            'lr': self.L.lr,
            'model_state': model_state_dict,
            'optim_state': self.L.opt.state_dict(),
            'scaler_state': scaler_state,
            'model_name': name,
        }
        return sd

    def _save_dict(self, **kwargs): raise NotImplementedError

    def do_saving(self, val='', **kwargs):
        torch.save(self._save_dict(**kwargs), str(self.save_path / f'e{self.L.n_epoch}_t{self.L.total_epochs}_{val}.pth'))

    def after_epoch(self):
        save = False
        if self.L.n_epoch == self.L.total_epochs - 1: save = False
        elif isinstance(self.save_step, int): save = self.save_step % self.L.n_epoch == 0
        else:
            if self.L.np_epoch > self.pct_counter:
                save = True
                self.pct_counter += self.save_step

        save_ema = self.L.kwargs['cfg'].TRAIN.EMA > 0.
        if save: self.do_saving('_after_epoch', save_ema=save_ema)


class HooksCB(Callback):
    def __init__(self, func, layers, perc_start=.5, step=1, logger=None):
        utils.file_op.store_attr(self, locals())
        self.hooks = Hooks(self.layers, self.func)
        self.do_once = True

    @utils.call.on_epoch_step
    def before_batch(self):
        if self.do_once and self.np_batch > self.perc_start:
            self.log_debug(f'Gathering activations at batch {self.np_batch}')
            self.do_once = False
            self.hooks.attach()

    @utils.call.on_epoch_step
    def after_batch(self):
        if self.hooks.is_attached(): self.hooks.detach()

    @utils.call.on_epoch_step
    def after_epoch(self): self.do_once = True


class Hook():
    def __init__(self, m, f):
        self.m, self.f = m, f

    def attach(self):
        self.hook = self.m.register_forward_hook(partial(self.f, self))

    def detach(self):
        if hasattr(self, 'hook'): self.hook.remove()

    def __del__(self):
        self.detach()


class Hooks(utils.file_op.ListContainer):
    def __init__(self, ms, f):
        super().__init__([Hook(m, f) for m in ms])

    def __enter__(self, *args):
        self.attach()
        return self

    def __exit__(self, *args):
        self.detach()

    def __del__(self):
        self.detach()

    def __delitem__(self, i):
        self[i].detach()
        super().__delitem__(i)

    def attach(self):
        for h in self: h.attach()

    def detach(self):
        for h in self: h.detach()

    def is_attached(self): return hasattr(self[0], 'hook')


def get_layers(model, conv=False, convtrans=False, lrelu=False, relu=False, bn=False, verbose=False):
    layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            if conv: layers.append(m)
        if isinstance(m, torch.nn.ConvTranspose2d):
            if convtrans: layers.append(m)
        elif isinstance(m, torch.nn.LeakyReLU):
            if lrelu: layers.append(m)
        elif isinstance(m, torch.nn.ReLU):
            if relu: layers.append(m)
        elif isinstance(m, torch.nn.BatchNorm2d):
            if bn: layers.append(m)
        else:
            if verbose: print(m)
    return layers


def append_stats(hook, mod, inp, outp, bins=100, vmin=0, vmax=0):
    if not hasattr(hook,'stats'): hook.stats = ([],[],[])
    means,stds,hists = hook.stats
    means.append(outp.data.mean().cpu())
    stds .append(outp.data.std().cpu())
    hists.append(outp.data.cpu().histc(bins,vmin,vmax))


def append_stats_buffered(hook, mod, inp, outp, device=torch.device('cpu'), bins=100, vmin=0, vmax=0):
    if not hasattr(hook,'stats'): hook.stats = (utils.common.TorchBuffer(shape=(1,), device=device),
                                                utils.common.TorchBuffer(shape=(1,), device=device),
                                                utils.common.TorchBuffer(shape=(bins,), device=device)
                                               )
    means,stds,hists = hook.stats
    means.push(outp.data.mean())
    stds .push(outp.data.std())
    hists.push(outp.data.float().histc(bins,vmin,vmax))


