import torch
from torch.utils import data
from tools import cfg, load_config, load_model_weight, Logger, plot_results, plot_bboxes, mkdir, select_device
from data.collate import collate_center, collate_ttf
import os
import datetime
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import logging
from icecream import ic


from data.dataset import build_dataset
from model.arch import build_model
from evaluator import build_evaluator
from trainer import build_trainer



# logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='val', help='task to run, test or val')
    parser.add_argument('--config', type=str, default='config/centerdet_csp.yaml',
                        help='model config file(.yml) path')
    parser.add_argument('--weights', type=str, default='workspace/centerdet_csp_exp/model_best/model_best.pt',
                        help='model weight file(.pth) path')
    parser.add_argument('--iou_thr', type=float, default=0.3,
                        help='model weight file(.pth) path')
    parser.add_argument('--nms_thr', type=float, default=0.4,
                        help='model weight file(.pth) path')
    parser.add_argument('--save_result', action='store_true', default=True,
                        help='save val results to txt')
    parser.add_argument('--device', type=str, default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    return args


def test(opt):
    load_config(cfg, opt.config)
    local_rank = -1
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = select_device(opt.device)
    # cfg.defrost()
    # timestr = datetime.datetime.now().__format__('%Y%m%d%H%M%S')
    # cfg.save_dir = os.path.join(cfg.save_dir, timestr)
    # cfg.freeze()
    # mkdir(local_rank, cfg.save_dir)
    cfg.update({'device': device})

    logger = Logger(local_rank, use_tensorboard=False)

    model = build_model(cfg.model).to(device)
    val_dataset = build_dataset(cfg.data.val, opt.task)
    val_loader = data.DataLoader(val_dataset,
                                 batch_size=64,
                                 num_workers=6,
                                 shuffle=False,
                                 pin_memory=False,
                                 collate_fn=collate_ttf)

    trainer = build_trainer(local_rank, cfg, model, logger)
    trainer.load_model(opt.weights, device)
    evaluator = build_evaluator(cfg, val_dataset)

    with torch.no_grad():
        results, val_loss_dict = trainer.run_epoch(0, val_loader, mode='val')
    eval_results = evaluator.evaluate(results, './', 0, logger, rank=local_rank)
    print(eval_results)



if __name__ == '__main__':
    args = parse_args()
    test(args)









