import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, preds, targets):
        pos_inds = targets.eq(1).float()
        neg_inds = targets.lt(1).float()

        neg_weights = torch.pow(1 - targets, self.beta)

        pred = torch.clamp(torch.sigmoid(preds), min=1e-4, max=1 - 1e-4)
        pos_loss = torch.pow(1 - pred, self.alpha) * torch.log(pred) * pos_inds
        neg_loss = torch.pow(pred, self.alpha) * torch.log(1 - pred) * neg_inds * neg_weights

        pos_num = pos_inds.sum()
        # if pos_num := pos_inds.sum() > 0:
        if pos_num > 0:
            loss = -(pos_loss.sum() + neg_loss.sum()) / pos_num
        else:
            loss = -(pos_loss.sum() + neg_loss.sum())

        return loss


class L1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(L1Loss, self).__init__()
        self.reduct = reduction

    def forward(self, preds, targets, mask):
        # pred -> [2, w, h]
        # terget -> [c, 2]
        # mask -> [c, 2]
        pred = torch.transpose(preds[:, mask[:, 0], mask[:, 1]], 0, 1)
        return F.l1_loss(pred, targets, reduction=self.reduct)


def compute_loss(preds, targets):
    # pred_heatmap -> [b, cls, w, h]
    # pred_wh -> [b, 2, w, h]
    # pred_offset -> [b, 2, w, h]
    pred_heatmap, pred_wh, pred_offset = preds
    # gt_heatmap -> [b, cls, w, h]
    # gt_wh -> [n, 2]
    # gt_offset -> [n, 2]
    # gt_mask -> [n, 3]
    gt_heatmap, gt_wh, gt_offset, gt_mask = targets

    focal_loss = FocalLoss()
    l1_loss = L1Loss()
    bs = pred_heatmap.shape[0]
    device = pred_heatmap.device
    wh = torch.zeros(1, device=device)
    offset = torch.zeros(1, device=device)

    heatmap = focal_loss(gt_heatmap, pred_heatmap)
    for idx in range(bs):
        inds = gt_mask[:, 0] == idx
        wh += l1_loss(pred_wh[idx], gt_wh[inds], gt_mask[inds][:, 1:])
        offset += l1_loss(pred_offset[idx], gt_offset[inds], gt_mask[inds])
    offset /= bs
    wh /= bs

    loss = heatmap + offset + wh * 0.1
    loss_metric = {
        'heatmap': heatmap.item(),
        'wh': wh.item(),
        'offset': offset.item()
    }

    return loss, loss_metric


# if __name__ == '__main__':
#     p = torch.randn((2, 20, 20))
#     t = torch.randn((4, 2))
#     mask_ = torch.randint(size=(4, 2), low=0, high=15)
#     l1 = L1Loss()
#     print(l1(p, t, mask_))