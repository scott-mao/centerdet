import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, preds, gt, pos_weight=None):
        if pos_weight is None:
            pos_inds = gt.eq(1).float()
        else:
            pos_inds = pos_weight

        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, self.beta)  # reduce punishment
        pos_loss = -torch.log(preds) * torch.pow(1 - preds, self.alpha) * pos_inds
        neg_loss = -torch.log(1 - preds) * torch.pow(preds, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            return neg_loss
        return (pos_loss + neg_loss) / num_pos