
def on_master(f):
    def wrapper(*args):
        if args[0].L.kwargs['cfg'].PARALLEL.IS_MASTER:
            return f(*args)
    return wrapper


def on_epoch_step(f):
    def wrapper(*args):
        if (args[0].L.n_epoch % args[0].L.step) == 0:
            return f(*args)
    return wrapper


def on_train(f):
    def wrapper(*args):
        if args[0].L.model.training:
            return f(*args)
    return wrapper


def on_validation(f):
    def wrapper(*args):
        if not args[0].L.model.training:
            return f(*args)
    return wrapper


def on_mode(mode):
    def decorator(function):
        def wrapper(*args, **kwargs):
            if args[0].L.mode == mode:
                return function(*args, **kwargs)
        return wrapper
    return decorator


def get_cb_by_instance(cbs, cls):
    for cb in cbs:
        if isinstance(cb, cls): return cb
    return None
