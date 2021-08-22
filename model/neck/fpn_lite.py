import torch
import torch.nn as nn
from model.module.layers import ConvNormAct, LiteConv
from model.module.init_weights import xavier_init


class DeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeConv, self).__init__()
        self.convs = nn.Sequential(
            ConvNormAct(in_channels, out_channels, k=1, s=1, act='ReLU6'),
            nn.ConvTranspose2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(4, 4),
                               padding=(1, 1),
                               stride=(2, 2),
                               groups=out_channels,
                               bias=False),
            nn.BatchNorm2d(out_channels),
            ConvNormAct(out_channels, out_channels, k=1, s=1, act='ReLU6'),
        )

    def forward(self, x):
        return self.convs(x)


class LiteUpsamle(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LiteUpsamle, self).__init__()
        self.deconv = DeConv(in_channels, out_channels)
        self.conv_up = nn.Sequential(
            LiteConv(in_channels, out_channels),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

    def forward(self, x):
        return self.deconv(x) + self.conv_up(x)


class ShortCut(nn.Module):
    def __init__(self, ch_in, ch_out, layer_num):
        super(ShortCut, self).__init__()
        self.shortcut_conv = nn.ModuleList()
        for i in range(layer_num):
            in_channels = ch_in if i == 0 else ch_out
            self.shortcut_conv.append(LiteConv(in_channels, ch_out, with_act=i < (layer_num - 1)))

    def forward(self, x):
        for sub_conv in  self.shortcut_conv:
            x = sub_conv(x)
        return x


class FPNLite(nn.Module):
    def __init__(self,
                 in_channels,
                 planes=(64, 48, 48),
                 shortcut_num=(3, 2, 1),
                 fusion_method='add'):
        super(FPNLite, self).__init__()
        self.planes = planes
        self.shortcut_num = shortcut_num[::-1]
        self.shortcut_len = len(shortcut_num)
        self.ch_in = in_channels[::-1]
        self.fusion_method = fusion_method

        self.upsample_list = nn.ModuleList()
        self.shortcut_list = nn.ModuleList()
        self.upper_list = list()

        for idx, out_c in enumerate(planes):
            in_c = self.ch_in[idx] if idx == 0 else self.upper_list[-1]
            self.upsample_list.append(LiteUpsamle(in_c, out_c))
            if idx < self.shortcut_len:
                self.shortcut_list.append(ShortCut(self.ch_in[idx + 1], out_c, self.shortcut_num[idx]))
                if self.fusion_method == 'add':
                    upper_c = out_c
                elif self.fusion_method == 'concat':
                    upper_c = out_c * 2
                else:
                    raise ValueError('Illegal fusion method. Expected add or concat, but received {}'.format(self.fusion_method))
                self.upper_list.append(upper_c)
        self.out_channels = self.upper_list[-1]
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        feat = inputs[-1]
        for idx, out_c in enumerate(self.planes):
            feat = self.upsample_list[idx](feat)
            if idx < self.shortcut_len:
                shortcut = self.shortcut_list[idx](inputs[-idx - 2])
                if self.fusion_method == 'add':
                    feat = feat + shortcut
                else:
                    feat = torch.cat((feat, shortcut), dim=1)
        return feat


if __name__ == '__main__':
    m_ = FPNLite(in_channels=(16, 32, 64, 128))
    inp = [torch.randn((4, 16, 80, 80)), torch.randn((4, 32, 40, 40)), torch.randn((4, 64, 20, 20)), torch.randn((4, 128, 10, 10))]
    print(m_(inp).shape)


