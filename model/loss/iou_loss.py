import math
import torch
import torch.nn as nn

from icecream import ic


Iou_Mode = {
    'GIoU': False,
    'DIoU': False,
    'CIoU': False,
}


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):

    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    # box2 = box2.T
    # ic(box2.shape)
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


class IoULoss(nn.Module):
    """
    IoU loss: Computing the IoU loss between a set of predicted bboxes and target bboxes.
    """
    def __init__(self, mode='GIoU'):
        super(IoULoss, self).__init__()
        assert mode in Iou_Mode.keys(), 'wrong parameter value of iou mode!'
        # assert reduction in ('mean', 'sum'), 'wrong parameter value of reduction!'
        self.iou_mode = Iou_Mode
        self.iou_mode.update({mode: True})
        # self.reduction = reduction

    def forward(self, preds, gt_boxes, weight):
        # preds -> [b, h, w, 4]
        # gt_boxes -> [b, h, w, 4]
        # weight -> [b, 1, h, w]
        pos_mask = weight > 0
        weight = weight[pos_mask].float()
        avg_factor = torch.sum(weight).float().item() + 1e-6
        pred_bboxes = preds[pos_mask].view(-1, 4)
        target_bboxes = gt_boxes[pos_mask].view(-1, 4)
        # calculate giou between gt and predcton

        gious = bbox_iou(pred_bboxes, target_bboxes, **self.iou_mode)

        iou_distances = 1 - gious
        return torch.sum(iou_distances * weight) / avg_factor


# class GIoULoss(nn.Module):
#     def __init__(self):
#         super(GIoULoss, self).__init__()
#
#     def forward(self, x):
#         pass



def giou_loss(pred,
              target,
              weight,
              avg_factor=None):
    """GIoU loss.
    Computing the GIoU loss between a set of predicted bboxes and target bboxes.
    """
    pos_mask = weight > 0

    weight = weight[pos_mask].float()
    if avg_factor is None:
        avg_factor = torch.sum(pos_mask).float().item() + 1e-6
    bboxes1 = pred[pos_mask].view(-1, 4)
    bboxes2 = target[pos_mask].view(-1, 4)

    lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
    wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
    enclose_x1y1 = torch.min(bboxes1[:, :2], bboxes2[:, :2])
    enclose_x2y2 = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1 + 1).clamp(min=0)

    overlap = wh[:, 0] * wh[:, 1]
    ap = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    ag = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    ious = overlap / (ap + ag - overlap)

    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]  # i.e. C in paper
    u = ap + ag - overlap
    gious = ious - (enclose_area - u) / enclose_area
    iou_distances = 1 - gious
    return torch.sum(iou_distances * weight) / avg_factor


def ciou_loss(pred,
              target,
              weight,
              avg_factor=None, asg_weight=None):
    """
    CIoU loss.
    Computing the CIoU loss between a set of predicted bboxes and target bboxes.
    """
    pos_mask = weight > 0

    weight = weight[pos_mask].float()
    if avg_factor is None:
        avg_factor = torch.sum(pos_mask).float().item() + 1e-6

    bboxes1 = pred[pos_mask].view(-1, 4)
    bboxes2 = target[pos_mask].view(-1, 4)

    pred_wh = bboxes1[:, [2, 3]] - bboxes1[:, [0, 1]] + 1
    gt_wh = bboxes2[:, [2, 3]] - bboxes2[:, [0, 1]] + 1
    # ------------------- inter ----------------- #
    # max x, max y
    lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    # min r, min b
    rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    wh = (rb - lt + 1).clamp(0)  # n, 2
    overlap = wh[:, 0] * wh[:, 1]
    # ------------------- enclose ----------------- #
    enclose_lt = torch.min(bboxes1[:, :2], bboxes2[:, :2])
    enclose_rb = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    enclose_wh = (enclose_rb - enclose_lt + 1).clamp(0)  # n, 2
    # enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]
    # ------------------ distance ------------------- #
    c2 = enclose_wh[:, 0] ** 2 + enclose_wh[:, 1] ** 2
    rho2 = ((bboxes1[:, 0] + bboxes1[:, 2] - bboxes2[:, 0] - bboxes2[:, 2]) ** 2 +
            (bboxes1[:, 1] + bboxes1[:, 3] - bboxes2[:, 1] - bboxes2[:, 3]) ** 2) / 4
    v = (4 / math.pi ** 2) * torch.pow(torch.atan(pred_wh[:, 0] / pred_wh[:, 1])
                                       - torch.atan(gt_wh[:, 0] / gt_wh[:, 1]), 2)

    pred_area = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    gt_area = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    ious = overlap / (pred_area + gt_area - overlap)

    with torch.no_grad():
        alpha = v / ((1 + 1e-9) - ious + v)
    ciou = ious - (rho2 / c2 + v * alpha)  # CIoU

    # u = pred_area + gt_area - overlap
    # gious = ious - (enclose_area - u) / enclose_area
    # iou_distance = 1 - ciou
    # if ags_weight is not None:

    if asg_weight is not None:
        asg = asg_weight[pos_mask]
        ciou = ciou * asg

    return torch.sum((1 - ciou) * weight) / avg_factor





# if __name__ == '__main__':
    # x = {
    #     'a' :1111, 'b' :2334
    # }
