import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module.layers import ConvNormAct
from model.module.init_weights import xavier_init
from icecream import ic

class PyramidUpsampleModulev2(nn.Module):
    """
    Pyramid Interpolation Module
    """
    def __init__(self, inch, outch, d_k=3):
        super(PyramidUpsampleModulev2, self).__init__()  # 3 -> 5  2 -> 3  4-> 6  5 -> 8
        # self.feature_align = ConvNormAct(inch, outch, k=1, s=1)
        self.maxpool = nn.AdaptiveAvgPool2d(1)
        self.feature_middle = nn.Sequential(
            ConvNormAct(outch, outch, k=5, s=1, g=outch),
            ConvNormAct(outch, outch, k=1, s=1),
        )
        self.feature_dilate = nn.Sequential(
            ConvNormAct(outch, outch, k=5, s=1, d=3, p=6, g=outch),
            ConvNormAct(outch, outch, k=1, s=1),
        )
        self.end = ConvNormAct(outch * 3, outch, k=1, s=1, act=None)
        self.up_sample =  nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        # x = self.feature_align(x)
        x = torch.cat([self.feature_dilate(x), self.feature_middle(x), self.maxpool(x) + x], dim=1)
        x = self.end(x)
        x = self.up_sample(x)
        # ic(x.shape)
        return F.relu(x, inplace=True)


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        self.in_channels = in_channels
        self.lateral_convs = nn.ModuleList()
        for inch in self.in_channels[:-1]:
            self.lateral_convs.append(
                nn.Sequential(
                    ConvNormAct(inch, out_channels, 1, 1),
                    ConvNormAct(out_channels, out_channels, 3, 1, g=out_channels),
                    ConvNormAct(out_channels, out_channels, 1, 1),
            ))
        self.lateral_convs.append(ConvNormAct(in_channels[-1], out_channels, 1, 1))
        # self.up_block = PyramidUpsampleModulev2(in_channels[-1], out_channels)
        self.merge_convs = nn.ModuleList()
        for inch in range(len(in_channels) - 1):
            self.merge_convs.append(
                nn.Sequential(
                    # ConvNormAct(out_channels, out_channels, 1, 1),
                    ConvNormAct(out_channels, out_channels, 3, 1, g=out_channels),
                    ConvNormAct(out_channels, out_channels, 1, 1),
            ))

        # self.shortcut1 = nn.Sequential(
        #     ConvNormAct(in_channels[-1], out_channels // 2, 1, 1),
        #     nn.UpsamplingBilinear2d(scale_factor=8)
        # )
        # self.shortcut2 = nn.Sequential(
        #     ConvNormAct(in_channels[-2], out_channels // 2, 1, 1),
        #     nn.UpsamplingBilinear2d(scale_factor=4)
        # )
        self.align = nn.Sequential(
            # ConvNormAct(out_channels * 2, out_channels, 1, 1),
            ConvNormAct(out_channels, out_channels, 3, 1, g=out_channels),
            ConvNormAct(out_channels, out_channels, 1, 1),
            # ConvNormAct(out_channels, out_channels, 3, 1, g=out_channels),
            # ConvNormAct(out_channels, out_channels, 1, 1),
        )
        self.out_channels = out_channels
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        # s4, s8, s16, s32 = inputs
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        for i in range(len(inputs) - 2, -1, -1):
            feat_size = laterals[i].shape[2:]
            # if i == len(inputs) - 2:
            #     up = self.up_block(laterals[i + 1])
            # else:
            up = F.interpolate(laterals[i + 1],
                               size=feat_size,
                               mode='bilinear',
                               align_corners=True)
            laterals[i] = self.merge_convs[i](up + laterals[i])

        # s1 = self.shortcut1(inputs[-1])
        # s2 = self.shortcut2(inputs[-2])
        # x = self.align(torch.cat((s1, s2, x), dim=1))

        return self.align(laterals[0])






if __name__ == '__main__':
    inc = [24, 116, 232, 464]
    m_ = FPN(inc, out_channels=64)
    inp = [torch.randn((2, 24, 80, 80)),
           torch.randn((2, 116, 40, 40)),
           torch.randn((2, 232, 20, 20)),
           torch.randn((2, 464, 10, 10))]
    print(m_(inp).shape)
    # for i in range(4, 0, -1):
    #     print(i)