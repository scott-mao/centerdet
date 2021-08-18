import torch
import torch.nn as nn
import numpy as np
from model.module.layers import ConvNormAct

from icecream import ic

class ContextModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContextModule, self).__init__()
        block_wide = out_channels // 2
        self.inconv = ConvNormAct(in_channels, out_channels, k=3)
        self.upconv = ConvNormAct(out_channels, block_wide, k=3, g=block_wide)
        self.downconv = ConvNormAct(out_channels, block_wide, k=5, g=block_wide)

    def forward(self, x):
        x = self.inconv(x)
        up = self.upconv(x)
        down = self.downconv(x)
        return torch.cat([up, down], dim=1)


class DetectModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DetectModule, self).__init__()

        self.upconv = ConvNormAct(in_channels, out_channels // 2, 3, 1)
        self.context = ContextModule(in_channels, out_channels // 2)

    def forward(self, x):
        up = self.upconv(x)
        down = self.context(x)
        return torch.cat([up, down], dim=1)


class HeadModule(nn.Module):
    def __init__(self, in_channels, out_channels, has_ext=False):
        super(HeadModule, self).__init__()
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.has_ext = has_ext
        if has_ext:
            self.ext = ConvNormAct(in_channels, in_channels, k=5, g=in_channels)

    def init_normal(self, std, bias):
        nn.init.normal_(self.head.weight, std=std)
        nn.init.constant_(self.head.bias, bias)

    def forward(self, x):
        if self.has_ext:
            x = self.ext(x)
        return self.head(x)


class CenterHead(nn.Module):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_classes=80,
                 use_ext=True):
        super(CenterHead, self).__init__()

        self.det = DetectModule(in_channels, feat_channels)
        # TODO three-in-one head
        self.hm_head = HeadModule(feat_channels, num_classes, has_ext=use_ext)
        self.reg_head = HeadModule(feat_channels, 2, has_ext=use_ext)
        self.offset_head = HeadModule(feat_channels, 2, has_ext=use_ext)
        self.init_weights()

    def init_weights(self):
        # Set the initial probability to avoid overflow at the beginning
        prob = 0.01
        d = -np.log((1 - prob) / prob)  # -2.19
        # Load backbone weights from ImageNet
        # self.backbone.load_pretrain()
        self.hm_head.init_normal(0.001, d)
        self.reg_head.init_normal(0.001, 0)
        # if self.has_landmark:
        self.offset_head.init_normal(0.001, 0)

    def forward(self, x):
        x = self.det(x)
        hm_feat = self.hm_head(x)
        reg_feat = self.reg_head(x)
        offset_feat = self.offset_head(x)
        return hm_feat, reg_feat, offset_feat


if __name__ == '__main__':
    inp = torch.randn((1, 96, 80, 80))
    m = CenterHead(96, 64)
    for i in m(inp):
        print(i.shape)