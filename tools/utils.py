import os
import glob
import torch
import torch.nn as nn
from copy import deepcopy
from pathlib import Path
import re
import logging
import math
import copy
from .flops_counter import get_model_complexity_info
# from utils.receptive_field import receptive_field

logger = logging.getLogger(__name__)


def rank_filter(func):
    def func_filter(local_rank=-1, *args, **kwargs):
        if local_rank < 1:
            return func(*args, **kwargs)
        else:
            pass
    return func_filter


@rank_filter
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_file(file):
    # Search for file if not found
    if os.path.isfile(file):
        return file
    # elif file == '' or file is None
    else: # file is fold
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found in Path: "%s"' % file  # assert file was found
        assert len(files) == 1, "Multiple files match '%s', specify exact path: %s" % (file, files)  # assert unique
        return files[0]  # return file


def create_workspace(cfg, resume=False):
    if resume:
        if 'weights_path' not in cfg:
            weights_dir = get_latest_run(cfg.save_path)
            assert weights_dir != '', \
                'last weights not exist in current resume path: {}'.format(os.path.abspath(cfg.save_path))
            log_dir = Path(weights_dir).parent.parent.as_posix()
        else:
            assert cfg.weights_path != '' and cfg.weights_path is not None, 'The Key "weights_path" is illegal in .yaml File'
            weights_dir = check_file(cfg.weights_path)
            log_dir = increment_path(Path(cfg.save_path) / '{}_exp'.format(cfg.proj_name), exist_ok=False)
    else:
        # mkdir(rank, work_fold)
        log_dir = increment_path(Path(cfg.save_path) / '{}_exp'.format(cfg.proj_name), exist_ok=False)
        weights_dir = (Path(log_dir) / 'weights/last.pt').as_posix()
        (Path(log_dir) / 'weights').mkdir(parents=True)
    return log_dir, weights_dir


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = f'Using torch {torch.__version__} '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            logger.info("%sCUDA:%g (%s, %dMB)" % (s, i, x[i].name, x[i].total_memory / c))
    else:
        logger.info(f'Using torch {torch.__version__} CPU')

    # logger.info('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


def get_latest_run(search_dir='.'):

    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/*last.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def flops_info(model, input_shape=(3, 320, 320)):
    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')


def consine_decay(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


class ModelEMA(object):
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


class WarmUpScheduler(object):
    def __init__(self, optimizer, max_steps, max_lr, ratio, warm_type='linear'):
        assert warm_type in ['linear', 'constant', 'exp'], 'Unsupport warm up type!'
        self.optim = optimizer
        self.max_steps = max_steps
        self.lr = max_lr
        self.punish_ratio = ratio
        self.warm_type = warm_type

    def get_lr(self, step):
        if self.warm_type == 'linear':
            lr = (1 - (1 - step / self.max_steps) * (1 - self.punish_ratio)) * self.lr
        elif self.warm_type == 'constant':
            lr = self.lr * self.punish_ratio
        elif self.warm_type == 'exp':
            lr = math.pow(self.punish_ratio, (1 - step / self.max_steps)) * self.lr
        else:
            raise NotImplementedError
        return lr

    def step(self, step):
        lr = self.get_lr(step)
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr



# class CosineDecayScheduler(object):
#     def __init__(self, optimizer, epochs, iters_epoch, init_lr, base_lr):
#         """
#         Cosine Decay Scheduler with Warm up.
#         :param optimizer: ex. optim.SGD.
#         :param epochs: maximum training iterations.
#         :param iters_epoch: warmup iterations.
#         :param init_lr:
#         :param base_lr:
#         """
#         self.optim = optimizer
#         self.epochs = epochs
#         self.iters_epoch = iters_epoch
#         self.init_lr = init_lr
#         self.base_lr = base_lr
#
#     def cosine_lr(self, iters):
#         pass
#
#
#     def step(self, iters):
#         pass

