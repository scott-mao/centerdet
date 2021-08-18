from .warp import warp_and_resize
from .color import color_aug_and_norm
from .gridmask import Gridmask
from ..gt_builder import center_targets, ttf_targets
import functools

import matplotlib.pyplot as plt



class Pipeline(object):
    def __init__(self, cfg, keep_ratio, is_eval=False):
        self.warp = functools.partial(warp_and_resize,
                                      warp_kwargs=cfg,
                                      keep_ratio=keep_ratio)
        self.color = functools.partial(color_aug_and_norm,
                                       kwargs=cfg)
        # self.gridmask = Gridmask(use_h=True, use_w=True)

        self.is_eval = is_eval
        if not is_eval:
            cls_num = cfg.pop('class_num')
            gt_fn = cfg.pop('gt_type')
            if gt_fn == 'centernet':
                self.gt = functools.partial(center_targets, num_cls=cls_num)
            elif gt_fn == 'ttfnet':
                self.gt = functools.partial(ttf_targets, num_cls=cls_num)
            else:
                raise NotImplementedError('Unknown ground_truth type function name!')

    def __call__(self, meta, dst_shape):
        meta = self.warp(meta=meta, dst_shape=dst_shape)
        meta = self.color(meta=meta)

        # meta = self.gridmask(meta)

        # plt.imshow(meta['img'])
        # plt.show()
        # exit(11)
        if not self.is_eval:
            meta = self.gt(meta=meta)
        return meta
