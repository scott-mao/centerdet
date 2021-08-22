import torch
from model.arch import build_model
from data.transform import Pipeline
from tools import cfg, load_config, load_model_weight, Logger, plot_results, plot_bboxes
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import logging
from icecream import ic

# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/centerdet_csp.yaml',
                        help='train config file path')
    parser.add_argument('--model', type=str, default='workspace/centerdet_csp_exp/model_best/model_best.pt',
                        help='model file path')
    parser.add_argument('--img-path', type=str, default='test_imgs/2007_001311.jpg',#'coco/val2017/000000001000.jpg',#mchar/mchar_train/001012.png',
                        help='image file path')
    parser.add_argument('--device', type=str, default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    return args


class Detector(object):
    def __init__(self, args, logger):
        load_config(cfg, args.config)
        self.cfg = cfg
        self.device = args.device
        model = build_model(cfg.model)
        # load weights
        ckpt = torch.load(args.model, map_location=self.device)
        load_model_weight(model, ckpt, logger)
        self.model = model.to(self.device)
        self.model.eval()

    def preprocess(self, img_path):
        img_info = dict()
        pipline_fn = Pipeline(self.cfg.data.val.pipeline,
                              keep_ratio=True,
                              is_eval=True)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # plt.imshow(img)
        # plt.show()
        # img = (img / 255).astype(np.float32)
        img_info['height'], img_info['width'] = img.shape[:2]
        meta = {'img_info': img_info, 'img': img, 'raw_img': img}
        meta = pipline_fn(meta, self.cfg.data.val.input_size)
        meta['img'] = torch.from_numpy(np.expand_dims(np.transpose(meta['img'], (2, 0, 1)), axis=0))
        return meta

    def inference(self, img_path):
        meta = self.preprocess(img_path)
        dets = self.model.inference(meta)
        # ic(dets)

        # img = plot_bboxes(meta['raw_img'], dets, self.cfg.class_names)
        img = plot_results(meta['raw_img'], dets, self.cfg.class_names, vis_thr=0.3, out_size=(320, 320))
        # cv2.imshow('cv', img)
        # cv2.waitKey(0)
        plt.imshow(img)
        plt.show()


def main():
    args = parse_args()
    # logger = Logger(-1, use_tensorboard=False)
    d = Detector(args, logger)
    d.inference(args.img_path)
    # print(cfg)


if __name__ == '__main__':
    main()


