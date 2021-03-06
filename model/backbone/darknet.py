"""
Copy from YOLOX

"""
import torch
import torch.nn as nn
from model.module.layers import act_layers


# def get_activation(name="silu", inplace=True):
#     if name == "silu":
#         module = nn.SiLU(inplace=inplace)
#     elif name == "relu":
#         module = nn.ReLU(inplace=inplace)
#     elif name == "lrelu":
#         module = nn.LeakyReLU(0.1, inplace=inplace)
#     else:
#         raise AttributeError("Unsupported act type: {}".format(name))
#     return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="SiLU"):
        super(BaseConv, self).__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act_layers(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="SiLU"):
        super(DWConv, self).__init__()
        self.dconv = BaseConv(
            in_channels, in_channels, ksize=ksize,
            stride=stride, groups=in_channels, act=act
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1,
            stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="SiLU"):
        super(Focus, self).__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1,
        )
        return self.conv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self, in_channels, out_channels, shortcut=True,
        expansion=0.5, depthwise=False, act="SiLU"
    ):
        super(Bottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="SiLU"):
        super(SPPBottleneck, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""
    def __init__(
        self, in_channels, out_channels, n=1,
        shortcut=True, expansion=0.5, depthwise=False, act="SiLU"
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act)
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class CSPDarknet(nn.Module):
    def __init__(
        self, dep_mul=0.33, wid_mul=0.25,
        out_features=['dark2', 'dark3', 'dark4', 'dark5'],
        depthwise=False, act="SiLU", pretrain=True):
        super(CSPDarknet, self).__init__()

        assert out_features, 'please provide output features of Darknet'
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        if dep_mul == 0.33 and wid_mul == 0.25:
            self.model_tag = 'nano'
        else:
            self.model_tag = None

        self.out_channels = [base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)
        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act)
        )
        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act)
        )
        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act)
        )
        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act)
        )
        self._load_pretrain(pretrain)


    def _load_pretrain(self, is_pretrain):
        if is_pretrain and self.model_tag == 'nano':
            ckpt_path = '/home/tjm/Documents/python/pycharmProjects/centerdet/samples/cspdarknet_nano.pt'
            pretrain_ckpt = torch.load(ckpt_path)
            self.load_state_dict(pretrain_ckpt)
            print('=> loading pretrained model from: {}'.format(ckpt_path))


    def forward(self, x):
        output = []
        x = self.stem(x)
        for i in range(2, 6):
            stage_tag = 'dark{}'.format(i)
            stage = getattr(self, stage_tag)
            x = stage(x)
            if stage_tag in self.out_features:
                output.append(x)
        return output




if __name__ == '__main__':

    inp = torch.randn((2, 3, 320, 320))
    model = CSPDarknet(0.33, 0.25, depthwise=True, act="SiLU")


    # print(model)
    # flops_info(model)

    for x in model(inp):
        print(x.shape)

