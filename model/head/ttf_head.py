import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from model.module.layers import ConvNormAct, LiteConv, ASGModule
from model.loss.iou_loss import IoULoss, giou_loss, ciou_loss
from model.loss.focal_loss import FocalLoss
from data.transform.warp import warp_boxes
from model.module.nms import non_max_suppression
from model.module.init_weights import normal_init, xavier_init

from icecream import ic

class HeadModule(nn.Module):
    def __init__(self, in_channels, out_channels, has_ext=True):
        super(HeadModule, self).__init__()
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.has_ext = has_ext
        if has_ext:
            self.ext = nn.Sequential(
                ConvNormAct(in_channels, in_channels, k=3, g=in_channels),
                ConvNormAct(in_channels, in_channels, k=1),
                # ConvNormAct(in_channels, in_channels, k=3, g=in_channels),
                # ConvNormAct(in_channels, in_channels, k=1),
            )

    def init_normal(self, std, bias):
        nn.init.normal_(self.head.weight, std=std)
        nn.init.constant_(self.head.bias, bias)
        if self.has_ext:
            for m in self.ext.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)

    def forward(self, x):
        if self.has_ext:
            x = self.ext(x)
        return self.head(x)


class HeadModuleLite(nn.Module):
    def __init__(self, ch_in, ch_out=128, planes_out=80, conv_num=2):
        super(HeadModuleLite, self).__init__()
        self.feat = nn.ModuleList()
        for i in range(conv_num):
            self.feat.append(LiteConv(in_channels=ch_in if i == 0 else ch_out, out_channels=ch_out))
        self.head = nn.Conv2d(in_channels=ch_out, out_channels=planes_out, kernel_size=(1, 1))

    def init_normal(self, std, bias):
        nn.init.normal_(self.head.weight, std=std)
        nn.init.constant_(self.head.bias, bias)
        for m in self.feat.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    def forward(self, x):
        for sub_module in self.feat:
            x = sub_module(x)
        return self.head(x)


class TTFHead(nn.Module):
    def __init__(self,
                 in_channels,
                 hm_head_planes,
                 wh_head_planes,
                 num_classes,
                 loc_weight,
                 reg_weight,
                 conv_num,
                 score_thr=0.02,
                 nms_thr=0.4,
                 topk=100,
                 use_asg=False,
                 wh_offset_base=16):
        super(TTFHead, self).__init__()
        self.topk = topk
        self.wh_offset_base = wh_offset_base
        self.num_classes = num_classes
        self.stride = 4
        self.base_loc = None
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.use_asg = use_asg
        # self.giou_loss = IoULoss()
        self.focal_loss = FocalLoss()
        self.loc_weight = loc_weight
        self.reg_weight = reg_weight

        if self.use_asg:
            self.asg_block = ASGModule(lam=0.5)

        self.hm_head = HeadModuleLite(in_channels, ch_out=hm_head_planes, planes_out=num_classes, conv_num=conv_num)
        self.tlrb_head = HeadModuleLite(in_channels, ch_out=wh_head_planes, planes_out=4, conv_num=conv_num)

        self.init_weights()

    def init_weights(self):
        # Set the initial probability to avoid overflow at the beginning
        prob = 0.01
        d = -np.log((1 - prob) / prob)  # -2.19
        self.hm_head.init_normal(0.001, d)
        self.tlrb_head.init_normal(0.001, 0)

    def forward(self, x):
        heatmap = self.hm_head(x)
        reg_box = F.relu(self.tlrb_head(x)) * self.wh_offset_base
        return heatmap, reg_box

    @staticmethod
    def _topk(scores, topk):
        batch, cat, height, width = scores.size()

        # both are (batch, cls_num, topk)
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), topk)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        # both are (batch, topk). select topk from 80 * topk
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), topk)
        topk_clses = (topk_ind / topk).int()
        topk_ind = topk_ind.unsqueeze(2)
        topk_inds = topk_inds.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
        topk_ys = topk_ys.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
        topk_xs = topk_xs.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    @staticmethod
    def _topk_simplify(scores, topk):
        # scores -> [b, cls_num, h, w]
        batch, cat, height, width = scores.size()
        # merge maximum on to a single heatmap
        # max_scores, max_labels: -> [b, h, w]
        max_scores, max_labels = torch.max(scores, dim=1, keepdim=False)
        # topk_scores -> [b, topk]
        topk_scores, topk_inds = torch.topk(max_scores.view(batch, -1), topk)
        topk_clses = max_labels.view(batch, -1).gather(1, topk_inds)

        # topk_ys -> [b, topk]
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

    @staticmethod
    def _nms(hm):
        hm_pool = F.max_pool2d(hm, 3, 1, 1)
        hm = ((hm_pool == hm).float() * hm)
        return hm

    def get_boxes(self, pred_hm, pred_wh):
        """
        Decode predction to bbox.
        :param pred_hm: shape -> [b, cls_num, feat_h, feat_w]
        :param pred_wh: shape -> [b, 4, feat_h, feat_w]
        :return: result_list which is a list of detection boxes: [det] * b
                    det: shape -> [x, y, x, y, conf, cls]
        """
        batch, cat, height, width = pred_hm.size()
        # print(cat)
        pred_hm = torch.sigmoid(pred_hm.detach())

        wh = pred_wh.detach()
        heat = self._nms(pred_hm)
        scores, inds, clses, ys, xs = self._topk(heat, topk=self.topk)

        xs = xs.view(batch, self.topk, 1) * self.stride
        ys = ys.view(batch, self.topk, 1) * self.stride

        wh = wh.permute(0, 2, 3, 1).contiguous()
        wh = wh.view(wh.size(0), -1, wh.size(3))
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), wh.size(2))
        wh = wh.gather(1, inds)

        wh = wh.view(batch, self.topk, 4)
        clses = clses.view(batch, self.topk, 1).float()
        scores = scores.view(batch, self.topk, 1)

        bboxes = torch.cat((xs - wh[..., [0]], ys - wh[..., [1]],
                            xs + wh[..., [2]], ys + wh[..., [3]]), dim=2)

        result_list = []
        for batch_i in range(bboxes.shape[0]):
            scores_per_img = scores[batch_i]
            scores_keep = (scores_per_img > self.score_thr).squeeze(-1)

            scores_per_img = scores_per_img[scores_keep]
            bboxes_per_img = bboxes[batch_i][scores_keep]
            labels_per_img = clses[batch_i][scores_keep]

            ret = torch.cat([bboxes_per_img, scores_per_img, labels_per_img], dim=1)
            # ret = torch.cat([bboxes_per_img, labels_per_img], dim=-1)  # shape=[num_boxes, 6] 6==>x1,y1,x2,y2,score,label
            result_list.append(ret)

        result_list = non_max_suppression(result_list, conf_thres=0.1)
        return result_list

    def get_boxes_single_img(self, pred_hm, pred_wh):
        # default batch is 1conf_thres
        batch, cat, height, width = pred_hm.size()
        pred_hm = torch.sigmoid(pred_hm.detach())

        wh = pred_wh.detach()
        heat = self._nms(pred_hm)

        # scores: shape -> [b, topk]
        scores, inds, clses, ys, xs = self._topk_simplify(heat, topk=self.topk)
        wh = wh.permute(0, 2, 3, 1).contiguous().view(batch, -1, 4)
        # inds from shape:[b, topk] to [b, topk, 4]
        inds = inds.unsqueeze(2).expand(batch, self.topk, 4)
        wh = wh.gather(1, inds).view(batch, self.topk, 4)

        ys = ys.view(batch, self.topk, 1) * self.stride
        xs = xs.view(batch, self.topk, 1) * self.stride

        bboxes = torch.cat((xs - wh[..., [0]],
                            ys - wh[..., [1]],
                            xs + wh[..., [2]],
                            ys + wh[..., [3]]), dim=-1)

        # scores = scores.view(batch, self.topk, 1)
        keep_ind = scores > self.score_thr
        keep_score = scores[keep_ind].view(batch, -1, 1)
        bboxes = bboxes[keep_ind].view(batch, -1, 4)
        clses = clses[keep_ind].view(batch, -1, 1)

        results = torch.cat((bboxes, keep_score, clses), dim=-1)
        if batch == 1:
            results = results.squeeze(0)
        return results

    def post_process(self, pred_hm, pred_box, meta):
        det_result = self.get_boxes(pred_hm, pred_box)
        # assert len(det_result) == len(meta['img_info'])
        warp_matrix = meta['warp_matrix']
        img_height = meta['img_info']['height']
        img_width = meta['img_info']['width']
        img_id = meta['img_info']['id']
        pred_imgs = dict()
        for idx, det_bboxes in enumerate(det_result):
            det_bboxes = det_bboxes.cpu().numpy()
            det_bboxes[:, :4] = warp_boxes(det_bboxes[:, :4],
                                           np.linalg.inv(warp_matrix[idx]),
                                           img_width[idx],
                                           img_height[idx])
            labels = det_bboxes[:, -1].astype(np.long)
            preds = dict()
            for i in range(self.num_classes):
                inds = (i == labels)
                preds[i] = det_bboxes[inds].tolist()
            pred_imgs[img_id[idx]] = preds

        return pred_imgs

    def post_process_single_image(self, pred_hm, pred_box, meta):
        det_result = self.get_boxes(pred_hm, pred_box)

        det_bboxes = det_result[0].cpu().numpy()
        if 'warp_matrix' in meta:
            warp_matrix = meta['warp_matrix']
            img_height = meta['img_info']['height']
            img_width = meta['img_info']['width']

            det_bboxes[:, :4] = warp_boxes(det_bboxes[:, :4],
                                           np.linalg.inv(warp_matrix),
                                           img_width,
                                           img_height)
        return det_bboxes


    def compute_loss(self,
                     pred_hm,
                     pred_wh,
                     gt_hm,
                     gt_box,
                     weight,
                     pos_weight):
        """
        Compute the loss of heatmap and regression bbox.
        :param pred_hm: shape -> [b, cls_num, feat_h, feat_w]
        :param pred_wh: shape -> [b, 4, feat_h, feat_w]
        :param gt_hm:   shape -> [b, cls_num, feat_h, feat_w]
        :param gt_box:  shape -> [b, 4, feat_h, feat_w]
        :param weight:  shape -> [b, 1, feat_h, feat_w]
        :param pos_weight
        :return: loss: scalar, loss_results: Tensor: [hm_loss, reg_loss, total_loss]
        """
        device = pred_hm.device
        gt_hm = gt_hm.to(device)
        gt_box = gt_box.to(device)
        weight = weight.to(device)
        pos_weight = pos_weight.to(device)

        feat_h, feat_w = pred_hm.shape[2:]
        pred_hm = torch.clamp(torch.sigmoid(pred_hm), min=1e-4, max=1 - 1e-4)

        if self.base_loc is None or feat_h != self.base_loc.shape[1] or \
                feat_w != self.base_loc.shape[2]:
            x_shift = torch.arange(0,
                                   (feat_w - 1) * self.stride + 1,
                                   self.stride,
                                   dtype=torch.float32,
                                   device=pred_hm.device)
            y_shift = torch.arange(0,
                                   (feat_h - 1) * self.stride + 1,
                                   self.stride,
                                   dtype=torch.float32,
                                   device=pred_hm.device)
            y_shift, x_shift = torch.meshgrid(y_shift, x_shift)
            self.base_loc = torch.stack((x_shift, y_shift), dim=0)

        # pred_box -> [b, 4, h, w]
        pred_box = torch.cat((self.base_loc - pred_wh[:, [0, 1]],
                              self.base_loc + pred_wh[:, [2, 3]]), dim=1).permute(0, 2, 3, 1)
        boxes = gt_box.permute(0, 2, 3, 1)
        # mask -> [b, h, w]
        mask = weight.view(-1, feat_h, feat_w)
        avg_factor = weight.sum() + 1e-4


        hm_loss = self.focal_loss(pred_hm, gt_hm, pos_weight) * self.loc_weight

        # ASG Module
        if self.use_asg:
            asg_weight = self.asg_block(pred_hm) # shape -> [b, h, w]
            reg_loss = ciou_loss(pred_box,
                                 boxes,
                                 mask,
                                 avg_factor=avg_factor,
                                 asg_weight=asg_weight) * self.reg_weight
        else:
            reg_loss = ciou_loss(pred_box, boxes, mask, avg_factor=avg_factor) * self.reg_weight

        return hm_loss, reg_loss

        # loss = hm_loss + reg_loss

        # metrics = {
        #     'hm': hm_loss.item(),
        #     'reg': reg_loss.item(),
        #     'total': loss.item()
        # }
        # torch.stack((hm_loss, reg_loss, loss)).detach()
        # return loss, metrics




# if __name__ == '__main__':
#     m = TTFNetHead(16, 32)
#     m.compute_loss()









