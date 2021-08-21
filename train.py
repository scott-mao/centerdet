import torch
from torch.utils import data
import argparse
from pathlib import Path
import numpy as np

from tools import (cfg, load_config, Logger, create_workspace, select_device)
from data.collate import collate_center, collate_ttf
from data.dataset import build_dataset
from model.arch import build_model
from evaluator import build_evaluator
from trainer import build_trainer

from icecream import ic

# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/PAFNet_lite.yaml',
                        help='train config file path')
    parser.add_argument('--resume', nargs='?', const=True, default=True,
                        help='resume most recent training')
    parser.add_argument('--device', type=str, default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')
    args = parser.parse_args()
    return args


def init_seeds(seed=0):
    """
    manually set a random seed for numpy, torch and cuda
    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(opt):
    torch.backends.cudnn.benchmark = True

    # load project configuration
    load_config(cfg, opt.config)
    local_rank = int(opt.local_rank)
    device = select_device(opt.device)
    # create workspace environment
    log_path, weigth_path = create_workspace(cfg.work, opt.resume)

    cfg.update({'log_dir': log_path})
    cfg.update({'weight_dir': weigth_path})
    cfg.update({'device': device})

    # create logger
    logger = Logger(local_rank, log_path, use_tensorboard=True)

    if opt.seed is not None:
        logger.log('Set random seed to {}'.format(opt.seed))
        init_seeds(opt.seed)

    logger.log('Creating model...')
    model = build_model(cfg.model).to(device)

    logger.log('Setting up data...')

    train_loader_cfg = cfg.data.train.pop('loader')
    val_loader_cfg = cfg.data.val.pop('loader')

    train_dataset = build_dataset(cfg.data.train, 'train')
    val_dataset = build_dataset(cfg.data.val, 'val')


    # TODO add 'build_loader' adaptation, do not expose such code to high-level api
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=train_loader_cfg.batch_size,
                                   num_workers=train_loader_cfg.num_workers,
                                   shuffle=train_loader_cfg.shuffle,
                                   pin_memory=train_loader_cfg.pin_memory,
                                   collate_fn=collate_ttf)

    val_loader = data.DataLoader(val_dataset,
                                 batch_size=val_loader_cfg.batch_size,
                                 num_workers=val_loader_cfg.num_workers,
                                 shuffle=val_loader_cfg.shuffle,
                                 pin_memory=val_loader_cfg.pin_memory,
                                 collate_fn=collate_ttf)

    evaluator = build_evaluator(cfg, val_dataset)
    trainer = build_trainer(local_rank, cfg, model, logger)


    if opt.resume:
        trainer.resume(weigth_path)

    # trainer.test()
    # exit(1)


    trainer.run(train_loader, val_loader, evaluator)

    # for i in train_loader:
    #     ic(i['img'].shape)
    #     ic(i['ht_map'].shape)
    #     ic(i['reg_box'].shape)
    #     ic(i['weight'].shape)
    #     # x = trainer.run_step(model, meta=i, mode='test')
    #
    #     # out = model.forward_eval(i)
    #     # print(out)
    #     # for ix in m.head.get_boxes_single_img(out[0], out[1]):
    #     #     print(ix.shape)
    #     # print(m.head.compute_loss(out[0], out[1], i['ht_map'], i['reg_box'], i['weight']))
    #
    #     ic(i['img_info'])
    #     ic(i['warp_matrix'])
    #     break


if __name__ == '__main__':
    args_ = parse_args()
    ic(args_)
    main(args_)
