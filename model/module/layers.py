import torch
import torch.nn as nn


activations = {
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'ReLU6': nn.ReLU6,
    'SELU': nn.SELU,
    'ELU': nn.ELU,
    'None': nn.Identity,
}


def act_layers(name):
    assert name in activations.keys()
    if name == 'LeakyReLU':
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    else:
        return activations[name](inplace=True)


class ConvNormAct(nn.Module):
    def __init__(self, inch, outch, k=1, s=1, p=None, g=1, d=1, act='ReLU'):
        super(ConvNormAct, self).__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inch,
                      out_channels=outch,
                      kernel_size=(k, k),
                      stride=(s, s),
                      padding=(p, p),
                      dilation=(d, d),
                      groups=g,
                      bias=False),
            nn.BatchNorm2d(outch),
            act_layers(act)
            # nn.ReLU(inplace=True) if act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)


class LiteConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, with_act=True):
        super(LiteConv, self).__init__()
        self.convs = nn.Sequential(
            ConvNormAct(in_channels, in_channels, k=5, s=stride, g=in_channels, act='ReLU6'),
            ConvNormAct(in_channels, out_channels, k=1, s=stride, act='ReLU6') if with_act
            else ConvNormAct(in_channels, out_channels, k=1, s=stride, act='None'),
            ConvNormAct(out_channels, out_channels, k=1, s=stride, act='ReLU6'),
            ConvNormAct(out_channels, out_channels, k=5, s=stride, g=out_channels, act='ReLU6') if with_act
            else ConvNormAct(out_channels, out_channels, k=5, s=stride, g=out_channels, act='None'),
        )

    def forward(self, x):
        return self.convs(x)


class ASGModule(nn.Module):
    def __init__(self, lam=1.):
        super(ASGModule, self).__init__()
        self.lam = lam

    def forward(self, x, gaussian_k):
        # x -> [b, 80, feat_w, feat_h]
        # gaussian_k -> [b, 80, feat_w, feat_h]
        b, ch, feat_w, feat_h = x.shape
        max_reduce_map, _ = torch.max(x, dim=1, keepdim=False)
        softmax = torch.softmax(max_reduce_map.view(b, -1), dim=-1).view(b, 1, feat_w, feat_h)
        ags = softmax * gaussian_k
        return (1 - self.lam) + self.lam * ags



if __name__ == '__main__':
    m = ConvNormAct(3, 3)
    print(m)

