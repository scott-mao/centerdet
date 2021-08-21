import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNLayer(nn.Module):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 num_groups=1):
        super(ConvBNLayer, self).__init__()

        self._conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            bias=False)

        self._batch_norm = nn.BatchNorm2d(num_filters)

    def forward(self, inputs, if_act=True):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if if_act:
            y = F.relu6(y, inplace=True)
        return y


class InvertedResidualUnit(nn.Module):
    def __init__(self, num_channels, num_in_filter, num_filters, stride,
                 filter_size, padding, expansion_factor):
        super(InvertedResidualUnit, self).__init__()
        num_expfilter = int(round(num_in_filter * expansion_factor))
        self._expand_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_expfilter,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1)

        self._bottleneck_conv = ConvBNLayer(
            num_channels=num_expfilter,
            num_filters=num_expfilter,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            num_groups=num_expfilter)

        self._linear_conv = ConvBNLayer(
            num_channels=num_expfilter,
            num_filters=num_filters,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1)

    def forward(self, inputs, ifshortcut):
        y = self._expand_conv(inputs, if_act=True)
        y = self._bottleneck_conv(y, if_act=True)
        y = self._linear_conv(y, if_act=False)
        if ifshortcut:
            y = inputs + y
        return y


class InvresiBlocks(nn.Module):
    def __init__(self, in_c, t, c, n, s):
        super(InvresiBlocks, self).__init__()

        self._first_block = InvertedResidualUnit(
            num_channels=in_c,
            num_in_filter=in_c,
            num_filters=c,
            stride=s,
            filter_size=3,
            padding=1,
            expansion_factor=t)

        self._block_list = nn.ModuleList()
        for i in range(1, n):
            block = InvertedResidualUnit(
                num_channels=c,
                num_in_filter=c,
                num_filters=c,
                stride=1,
                filter_size=3,
                padding=1,
                expansion_factor=t)
            self._block_list.append(block)

    def forward(self, inputs):
        y = self._first_block(inputs, ifshortcut=False)
        for block in self._block_list:
            y = block(y, ifshortcut=True)
        return y


class MobileNetV2(nn.Module):
    def __init__(self, scale=1.0, pretrain=True):
        super(MobileNetV2, self).__init__()
        self.scale = scale

        bottleneck_params_list = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        self.conv1 = ConvBNLayer(
            num_channels=3,
            num_filters=int(32 * scale),
            filter_size=3,
            stride=2,
            padding=1)

        self.block_list = nn.ModuleList()
        i = 1
        in_c = int(32 * scale)
        for layer_setting in bottleneck_params_list:
            t, c, n, s = layer_setting
            i += 1
            block = InvresiBlocks(
                in_c=in_c,
                t=t,
                c=int(c * scale),
                n=n,
                s=s)
            self.block_list.append(block)
            in_c = int(c * scale)

        self._init_weights(pretrain)
        # self.out_c = int(1280 * scale) if scale > 1.0 else 1280
        # self.conv9 = ConvBNLayer(
        #     num_channels=in_c,
        #     num_filters=self.out_c,
        #     filter_size=1,
        #     stride=1,
        #     padding=0)

    def _init_weights(self, pretrain):
        if pretrain and self.scale == 1.:
            ckpt_path = 'samples/MobileNetV2_ssld_pretrained.pt'
            pretrain_ckpt = torch.load(ckpt_path)
            self.load_state_dict(pretrain_ckpt, strict=True)
            print('=> loading pretrained model from: {}'.format(ckpt_path))
            del pretrain_ckpt

    def forward(self, inputs):
        y = self.conv1(inputs, if_act=True)
        for block in self.block_list:
            y = block(y)
        # y = self.conv9(y, if_act=True)
        return y



if __name__ == '__main__':
    inp = torch.randn((1, 3, 320, 320))
    m = MobileNetV2()
    # for k, v in m.state_dict().items():
    #     print(k, v.shape)



