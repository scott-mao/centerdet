import os
import logging
import torch
import numpy as np
import time
from termcolor import colored
from .utils import mkdir


class Logger(object):
    def __init__(self, local_rank, save_dir='./', use_tensorboard=True):
        mkdir(local_rank, save_dir)
        self.rank = local_rank
        fmt = colored('[%(name)s]', 'magenta', attrs=['bold']) + colored('[%(asctime)s]', 'blue') + \
              colored('%(levelname)s:', 'green') + colored('%(message)s', 'white')

        txt_path = os.path.join(save_dir, 'log_{}.txt'.format(int(time.time())))

        logging.basicConfig(level=logging.INFO,
                            filename=txt_path,
                            filemode='w')
        self.log_dir = os.path.join(save_dir, 'logs')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt, datefmt="%m-%d %H:%M:%S")
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)
        if use_tensorboard:
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')
            if self.rank < 1:
                logging.info('Using Tensorboard, logs will be saved in {}'.format(self.log_dir))
                logging.info('Check it with Command "tensorboard --logdir ./{}" in Terminal, '
                             'view at http://localhost:6006/'.format(self.log_dir))
                self.writer = SummaryWriter(log_dir=self.log_dir)

    def log(self, string):
        if self.rank < 1:
            logging.info(string)

    def scalar_summary(self, tag, value, step):
        if self.rank < 1:
            self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)


class MovingAverage(object):
    def __init__(self, val, window_size=50):
        self.window_size = window_size
        # self.queue = list()
        self.reset()
        self.push(val)

    def reset(self):
        self.queue = []

    def push(self, val):
        self.queue.append(val)
        if len(self.queue) > self.window_size:
            self.queue.pop(0)

    def avg(self):
        return np.mean(self.queue)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, val):
        self.reset()
        self.update(val)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
