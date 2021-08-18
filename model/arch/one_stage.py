import time
import torch
import torch.nn as nn
from ..backbone import build_backbone
from ..neck import build_neck
from ..head import build_head


class OneStage(nn.Module):
    def __init__(self,
                 backbone_cfg,
                 neck_cfg=None,
                 head_cfg=None,):
        super(OneStage, self).__init__()
        self.backbone = build_backbone(backbone_cfg)
        if neck_cfg is not None:
            neck_cfg.update({'in_channels': self.backbone.out_channels})
            self.neck = build_neck(neck_cfg)
        if head_cfg is not None:
            head_cfg.update({'in_channels': self.neck.out_channels})
            self.head = build_head(head_cfg)

    def forward(self, x):
        x = self.backbone(x)
        if hasattr(self, 'neck') and self.neck is not None:
            x = self.neck(x)
        if hasattr(self, 'head'):
            if isinstance(x, tuple):
                out = []
                for xx in x:
                    out.append(self.head(xx))
                x = tuple(out)
            else:
                x = self.head(x)
        return x

    # def inference(self, meta):
    #     with torch.no_grad():
    #         torch.cuda.synchronize()
    #         time1 = time.time()
    #         preds = self(meta['img'])
    #         torch.cuda.synchronize()
    #         time2 = time.time()
    #         print('forward time: {:.3f}s'.format((time2 - time1)), end=' | ')
    #         results = self.head.post_process(preds, meta)
    #         torch.cuda.synchronize()
    #         print('decode time: {:.3f}s'.format((time.time() - time2)), end=' | ')
    #     return results

    def forward_train(self, gt_meta):
        preds = self(gt_meta['img'])
        # pred_hm, pred_bbox = preds
        loss_hm, loss_reg = self.head.compute_loss(*preds,
                                                   gt_meta['ht_map'],
                                                   gt_meta['reg_box'],
                                                   gt_meta['weight'],
                                                   gt_meta['pos_weight'])
        return preds, loss_hm, loss_reg
        # return preds, loss, loss_states

    def forward_eval(self, gt_meta):
        with torch.no_grad():
            pred_hm, pred_bbox = self(gt_meta['img'])
            det_boxes = self.head.post_process(pred_hm, pred_bbox, gt_meta)

        return det_boxes

    def inference(self, meta):
        with torch.no_grad():
            pred_hm, pred_bbox = self(meta['img'])
            det_boxes = self.head.post_process_single_image(pred_hm, pred_bbox, meta)
        return det_boxes

