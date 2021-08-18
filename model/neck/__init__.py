import copy
from .ufpn import UFPN
from .fpn import FPN
from .fpn_lite import FPNLite

def build_neck(cfg):
    fpn_cfg = copy.deepcopy(cfg)
    name = fpn_cfg.pop('name')
    if name == 'UFPN':
        return UFPN(**fpn_cfg)
    elif name == 'FPN':
        return FPN(**fpn_cfg)
    elif name == 'FPNLite':
        return FPNLite(**fpn_cfg)
    else:
        raise NotImplementedError