import torch
import torch.nn as nn
import torch.nn.functional as F
# from ..module.init_weights import xavier_init
from model.module.layers import ConvNormAct

from icecream import ic

class DeconvDepthwiseMoudule(nn.Module):
    def __init__(self, inch, outch, k, d=3, act=False):
        super(DeconvDepthwiseMoudule, self).__init__()
        p = d + 1
        o_p = 1 if d == 2 else 0
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=inch, out_channels=outch, kernel_size=(k, k),
                               padding=(p, p), output_padding=(o_p, o_p),
                               stride=(2, 2), dilation=d, groups=inch, bias=False),
            nn.BatchNorm2d(outch),
            nn.ReLU(inplace=True) if act else nn.Identity()
        )

    def forward(self, x):
        return self.deconv(x)


class PyramidUpsampleModule(nn.Module):
    """
    Pyramid Interpolation Module
    """
    def __init__(self, inch, outch, d_k=3):
        super(PyramidUpsampleModule, self).__init__()  # 3 -> 5  2 -> 3  4-> 6  5 -> 8
        # padding = (3 * (d_k - 1) + 4) // 2
        # self.d_k = d_k
        self.up_pool1 = nn.Sequential(
            ConvNormAct(inch, outch, k=1, s=1, act=None),
            DeconvDepthwiseMoudule(outch, outch, 4, d=d_k),
            ConvNormAct(outch, outch, k=1, s=1),
        )
        self.up_pool2 = nn.Sequential(
            ConvNormAct(inch, outch, k=1, s=1, act=None),
            ConvNormAct(outch, outch, k=3, s=1, g=outch),
            ConvNormAct(outch, outch, k=1, s=1),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inch, outch, bias=False),
            nn.BatchNorm1d(outch)
        )

    def forward(self, x):
        # if self.d_k % 2 != 0:
        #     pool1 = self.up_pool1(x)[:, :, 1: -1, 1: -1]
        # else:
        pool1 = self.up_pool1(x)
        pool2 = self.up_pool2(x)
        pool3 = self.fc(self.avg_pool(x).view(x.shape[0], -1))
        return F.relu(pool1 + pool2 + pool3[:, :, None, None], inplace=True)


class MultiFeatFusionModule(nn.Module):
    def __init__(self, chA, chB, d):
        super(MultiFeatFusionModule, self).__init__()
        outch = chB // 2
        # outch = 64
        # features from deeper layer with larger stride
        self.upsample_feat = PyramidUpsampleModule(chA, outch, d)
        # shallow features
        self.shallow_feat = nn.Sequential(
            # TODO [need optimize normalization and activation]
            ConvNormAct(chB, outch, k=1),
            ConvNormAct(outch, outch, k=5, g=outch, act=None),
            ConvNormAct(outch, outch, k=1)
        )

    def forward(self, x0, x1):
        return torch.cat([self.upsample_feat(x0), self.shallow_feat(x1)], dim=1)


class MultiFeatFusionModule2(nn.Module):
    def __init__(self, chA, chB, d):
        super(MultiFeatFusionModule2, self).__init__()
        if chA == 96:
            outch = 32
        elif chA == 64:
            outch = 32
        elif chA == 116:
            outch = 32
        else:
            outch = chB // 2
        # outch = 64
        # features from deeper layer with larger stride
        self.upsample_feat = nn.Sequential(
            ConvNormAct(chA, outch, k=1, s=1),
            ConvNormAct(outch, outch, k=3, s=1, g=outch, act=None),
            ConvNormAct(outch, outch, k=1, s=1),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        # self.upsample_feat = nn.UpsamplingBilinear2d(scale_factor=2)
        # shallow features
        self.shallow_feat = nn.Sequential(
            # TODO [need optimize normalization and activation]
            ConvNormAct(chB, outch, k=1),
            ConvNormAct(outch, outch, k=3, g=outch, act=None),
            ConvNormAct(outch, outch, k=1)
        )

    def forward(self, x0, x1):
        return torch.cat([self.upsample_feat(x0), self.shallow_feat(x1)], dim=1)


class UFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(UFPN, self).__init__()
        assert isinstance(in_channels, list) and len(in_channels) == 4
        self.in_channels = in_channels
        ch_s4, ch_s8, ch_s16, ch_s32 = self.in_channels
        self.fuse_to_s16 = MultiFeatFusionModule(ch_s32, ch_s16, d=3)
        self.fuse_to_s8 = MultiFeatFusionModule(ch_s16, ch_s8, d=3)
        self.fuse_to_s4 = MultiFeatFusionModule2(ch_s8, ch_s4, d=3)
        self.ext = nn.Sequential(
            ConvNormAct(64, out_channels, k=1),
            ConvNormAct(out_channels, out_channels, k=3, g=out_channels),
            ConvNormAct(out_channels, out_channels, k=1),
            ConvNormAct(out_channels, out_channels, k=3, g=out_channels),
        )

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        s4, s8, s16, s32 = inputs
        s16 = self.fuse_to_s16(s32, s16)
        # ic(s16.shape)
        s8 = self.fuse_to_s8(s16, s8)
        # ic(s8.shape)
        s4 = self.fuse_to_s4(s8, s4)
        # ic(s4.shape)
        s4 = self.ext(s4)
        return s4


# class FPN(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  num_outs,
#                  start_level=0,
#                  end_level=-1,
#                  conv_cfg=None,
#                  norm_cfg=None,
#                  activation=None
#                  ):
#         super(FPN, self).__init__()
#         assert isinstance(in_channels, list)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_ins = len(in_channels)
#         self.num_outs = num_outs
#         # self.fp16_enabled = False
#
#         if end_level == -1:
#             self.backbone_end_level = self.num_ins
#             assert num_outs >= self.num_ins - start_level
#         else:
#             # if end_level < inputs, no extra level is allowed
#             self.backbone_end_level = end_level
#             assert end_level <= len(in_channels)
#             assert num_outs == end_level - start_level
#         self.start_level = start_level
#         self.end_level = end_level
#         self.lateral_convs = nn.ModuleList()
#
#         for i in range(self.start_level, self.backbone_end_level):
#             l_conv = ConvModule(
#                 in_channels[i],
#                 out_channels,
#                 1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg,
#                 activation=activation,
#                 inplace=False)
#
#             self.lateral_convs.append(l_conv)
#         self.init_weights()
#
#     # default init_weights for conv(msra) and norm in ConvModule
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 xavier_init(m, distribution='uniform')
#
#     def forward(self, inputs):
#         assert len(inputs) == len(self.in_channels)
#
#         # build laterals
#         laterals = [
#             lateral_conv(inputs[i + self.start_level])
#             for i, lateral_conv in enumerate(self.lateral_convs)
#         ]
#
#         # build top-down path
#         used_backbone_levels = len(laterals)
#         for i in range(used_backbone_levels - 1, 0, -1):
#             prev_shape = laterals[i - 1].shape[2:]
#             laterals[i - 1] += F.interpolate(
#                 laterals[i], size=prev_shape, mode='bilinear')
#
#         # build outputs
#         outs = [
#             # self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
#             laterals[i] for i in range(used_backbone_levels)
#         ]
#         return tuple(outs)

if __name__ == '__main__':
#     from model.backbone.shufflenetv2 import ShuffleNetV2
    inc = [24, 116, 232, 464]
    m = UFPN(inc, out_channels=96)
    inp = [torch.randn((2, 24, 80, 80)),
           torch.randn((2, 116, 40, 40)),
           torch.randn((2, 232, 20, 20)),
           torch.randn((2, 464, 10, 10))]
    print(m(inp).shape)
    # print(m)