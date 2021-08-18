import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt, pos_weights, keep_mask=None):
        # pos_inds = gt.eq(1).float()

        # positiva objects
        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_weights
        # negative objects
        neg_inds = gt.lt(1).float()
        neg_weights = torch.pow(1 - gt, self.beta)
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        if keep_mask is not None:
            pos_loss = (pos_loss * keep_mask).sum()
            neg_loss = (neg_loss * keep_mask).sum()
        else:
            pos_loss = pos_loss.sum()
            neg_loss = neg_loss.sum()
        return -(pos_loss + neg_loss)


class GIoULoss(nn.Module):
    def __init__(self):
        super(GIoULoss, self).__init__()
        self.shift = None

    def forward(self, preds, targets, weight):
        # pred is   b, 4, h, w
        # gt is     b, 4, h, w
        # mask is   b, 1, h, w
        # 4 channel is x, y, r, b - cx
        h, w = preds.shape[2:]
        weight = weight.view(-1, h, w)
        mask = weight > 0
        weight = weight[mask]
        avg_factor = torch.sum(weight)

        if avg_factor == 0:
            print("avg is zero")
            return torch.tensor(0.0)

        if self.shift is None:
            x = torch.arange(0, w, device=preds.device)
            y = torch.arange(0, h, device=preds.device)
            shift_y, shift_x = torch.meshgrid(y, x)
            self.shift = torch.stack((shift_x, shift_y), dim=0).float()  # 2, h, w

        pred_boxes = torch.cat((
            self.shift - preds[:, [0, 1]],
            self.shift + preds[:, [2, 3]]
        ), dim=1).permute(0, 2, 3, 1)  # b, 4, h, w   to   b, h, w, 4

        # gt_boxes = torch.cat((
        #     self.shift + gt[:, [0, 1]],
        #     self.shift + gt[:, [2, 3]]
        # ), dim=1).permute(0, 2, 3, 1)  # b, 4, h, w   to   b, h, w, 4
        gt_boxes = targets.permute(0, 2, 3, 1)

        pred_boxes = pred_boxes[mask].view(-1, 4)
        gt_boxes = gt_boxes[mask].view(-1, 4)

        # max x, max y
        lt = torch.max(pred_boxes[:, :2], gt_boxes[:, :2])

        # min r, min b
        rb = torch.min(pred_boxes[:, 2:], gt_boxes[:, 2:])
        wh = (rb - lt + 1).clamp(0)  # n, 2

        enclose_lt = torch.min(pred_boxes[:, :2], gt_boxes[:, :2])
        enclose_rb = torch.max(pred_boxes[:, 2:], gt_boxes[:, 2:])
        enclose_wh = (enclose_rb - enclose_lt + 1).clamp(0)  # n, 2
        enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]
        overlap = wh[:, 0] * wh[:, 1]

        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0] + 1) * (pred_boxes[:, 3] - pred_boxes[:, 1] + 1)
        gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
        ious = overlap / (pred_area + gt_area - overlap)

        u = pred_area + gt_area - overlap
        gious = ious - (enclose_area - u) / enclose_area
        iou_distance = 1 - gious
        return torch.sum(iou_distance * weight) / avg_factor