import copy
from .center_head import CenterHead
from .ttf_head import TTFHead

def build_head(cfg):
    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop('name')
    if name == 'CenterHead':
        return CenterHead(**head_cfg)
    elif name == 'TTFHead':
        return TTFHead(**head_cfg)
    else:
        raise NotImplementedError