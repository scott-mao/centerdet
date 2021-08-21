import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from numbers import Integral
from icecream import ic


def hard_sigmod(x, inplace: bool=False):
    if inplace:
        return (1.2 * x).add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(1.2 * x + 3.) / 6


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 filter_size,
                 stride,
                 padding,
                 num_groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__()
        self.act = act
        self.conv = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=(filter_size, filter_size),
            stride=(stride, stride),
            padding=(padding, padding),
            groups=num_groups,
            bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            if self.act == 'relu':
                x = F.relu(x)
            elif self.act == 'relu6':
                x = F.relu6(x)
            elif self.act == 'hard_swish':
                x = F.hardswish(x)
            else:
                raise NotImplementedError(
                    'The activation function is selected incorrectly.')
        return x


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_channels = int(channel // reduction)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=mid_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0))
        self.conv2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=channel,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0))

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = hard_sigmod(outputs, inplace=True)
        return inputs * outputs


class ResidualUnit(nn.Module):
    def __init__(self,
                 in_c,
                 mid_c,
                 out_c,
                 filter_size,
                 stride,
                 use_se,
                 act=None,
                 return_list=False):
        super(ResidualUnit, self).__init__()
        self.if_shortcut = stride == 1 and in_c == out_c
        self.use_se = use_se
        self.return_list = return_list

        self.expand_conv = ConvBNLayer(
            in_c=in_c,
            out_c=mid_c,
            filter_size=1,
            stride=1,
            padding=0,
            act=act)
        self.bottleneck_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=mid_c,
            filter_size=filter_size,
            stride=stride,
            padding=int((filter_size - 1) // 2),
            num_groups=mid_c,
            act=act)
        if self.use_se:
            self.mid_se = SEModule(mid_c)
        self.linear_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=out_c,
            filter_size=1,
            stride=1,
            padding=0,
            act=None)

    def forward(self, inputs):
        y = self.expand_conv(inputs)
        x = self.bottleneck_conv(y)
        if self.use_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = inputs + x
        if self.return_list:
            return [y, x]
        else:
            return x


class ExtraBlockDW(nn.Module):
    def __init__(self,
                 in_c,
                 ch_1,
                 ch_2,
                 stride):
        super(ExtraBlockDW, self).__init__()
        self.pointwise_conv = ConvBNLayer(
            in_c=in_c,
            out_c=ch_1,
            filter_size=1,
            stride=1,
            padding=0,
            act='relu6')
        self.depthwise_conv = ConvBNLayer(
            in_c=ch_1,
            out_c=ch_2,
            filter_size=3,
            stride=stride,
            padding=1,
            num_groups=int(ch_1),
            act='relu6')
        self.normal_conv = ConvBNLayer(
            in_c=ch_2,
            out_c=ch_2,
            filter_size=1,
            stride=1,
            padding=0,
            act='relu6')

    def forward(self, inputs):
        x = self.pointwise_conv(inputs)
        x = self.depthwise_conv(x)
        x = self.normal_conv(x)
        return x


class MobileNetV3(nn.Module):
    def __init__(
            self,
            scale=1.0,
            model_size='large',
            feature_maps=None,
            with_extra_blocks=False,
            load_pretrain=True):
        super(MobileNetV3, self).__init__()
        if feature_maps is None:
            feature_maps = [6, 12, 15]
        if isinstance(feature_maps, Integral):
            feature_maps = [feature_maps]
        self.feature_maps = feature_maps
        self.with_extra_blocks = with_extra_blocks
        self.scale = scale
        inplanes = 16
        if model_size == 'large':
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', 1], # 1
                [3, 64, 24, False, 'relu', 2], # 2
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', 2],  # RCNN output
                [5, 120, 40, True, 'relu', 1], # 5
                [5, 120, 40, True, 'relu', 1],  # YOLOv3 output
                [3, 240, 80, False, 'hard_swish', 2], # RCNN output
                [3, 200, 80, False, 'hard_swish', 1], # 8
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1], # 10
                [3, 480, 112, True, 'hard_swish', 1],
                [3, 672, 112, True, 'hard_swish', 1],  # YOLOv3 output
                [5, 672, 160, True, 'hard_swish', 2],  # SSD/SSDLite/RCNN output
                [5, 960, 160, True, 'hard_swish', 1], # 14
                [5, 960, 160, True, 'hard_swish', 1],  # YOLOv3 output
            ]
        elif model_size == 'small':
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', 2],
                [3, 72, 24, False, 'relu', 2],  # RCNN output
                [3, 88, 24, False, 'relu', 1],  # YOLOv3 output
                [5, 96, 40, True, 'hard_swish', 2],  # RCNN output
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 120, 48, True, 'hard_swish', 1],
                [5, 144, 48, True, 'hard_swish', 1],  # YOLOv3 output
                [5, 288, 96, True, 'hard_swish', 2],  # SSD/SSDLite/RCNN output
                [5, 576, 96, True, 'hard_swish', 1],
                [5, 576, 96, True, 'hard_swish', 1],  # YOLOv3 output
            ]
        else:
            raise NotImplementedError(
                'mode[{}_model] is not implemented!'.format(model_size))

        self.conv1 = ConvBNLayer(
            in_c=3,
            out_c=make_divisible(inplanes * scale),
            filter_size=3,
            stride=2,
            padding=1,
            num_groups=1,
            act='hard_swish')

        self.out_channels = []
        self.block_list = []
        i = 0
        inplanes = make_divisible(inplanes * scale)
        for (k, exp, c, se, nl, s) in self.cfg:
            # for SSD/SSDLite, first head input is after ResidualUnit expand_conv
            return_list = self.with_extra_blocks and i + 2 in self.feature_maps
            block = ResidualUnit(
                    in_c=inplanes,
                    mid_c=make_divisible(scale * exp),
                    out_c=make_divisible(scale * c),
                    filter_size=k,
                    stride=s,
                    use_se=se,
                    act=nl,
                    return_list=return_list)
            self.block_list.append(block)
            inplanes = make_divisible(scale * c)
            i += 1
            self._update_out_channels(make_divisible(scale * exp) if return_list else inplanes, i + 1, feature_maps)

        if self.with_extra_blocks:
            extra_out_c = make_divisible(scale * self.cfg[-1][1])
            conv_extra = ConvBNLayer(
                    in_c=inplanes,
                    out_c=extra_out_c,
                    filter_size=1,
                    stride=1,
                    padding=0,
                    num_groups=1,
                    act='hard_swish')
            self.block_list.append(conv_extra)
            i += 1
            self._update_out_channels(extra_out_c, i + 1, feature_maps)
        self.block = nn.Sequential(*self.block_list)
        self._init_params(pretrain=load_pretrain)

    def _update_out_channels(self, channel, feature_idx, feature_maps):
        if feature_idx in feature_maps:
            self.out_channels.append(channel)

    def _init_params(self, pretrain):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        if pretrain and self.scale == 1.:
            ckpt_path = 'samples/MobileNetV3_large_x1_0_ssld_pretrained.pth'
            pretrain_ckpt = torch.load(ckpt_path)
            self.load_state_dict(pretrain_ckpt, strict=True)
            print('=> loading pretrained model from: {}'.format(ckpt_path))
            del pretrain_ckpt

    def forward(self, inputs):
        x = self.conv1(inputs)
        outs = []
        for idx, block in enumerate(self.block):
            x = block(x)
            if idx + 2 in self.feature_maps:
                if isinstance(x, list):
                    outs.append(x[0])
                    x = x[1]
                else:
                    outs.append(x)
        return outs


if __name__ == '__main__':
    from collections import OrderedDict
    m = MobileNetV3(scale=1., feature_maps=[5, 8, 14, 17], with_extra_blocks=True)

    # print(len(m.state_dict()))

    # print(m.output_channels)
    inp = torch.randn((2, 3, 320, 320))
    for i in m(inp):
        print(i.shape)
