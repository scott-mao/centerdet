import copy
from .resnet import ResNet
from .ghostnet import GhostNet
from .shufflenetv2 import ShuffleNetV2
from .mobilenetv2 import MobileNetV2
from .efficientnet_lite import EfficientNetLite
from .mobilenetv3 import MobileNetV3_Small
from .pp_mbnetv3 import MobileNetV3
# from .custom_csp import CustomCspNet
# from .repvgg import RepVGG


def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop('name')
    if name == 'ResNet':
        return ResNet(**backbone_cfg)
    elif name == 'ShuffleNetV2':
        return ShuffleNetV2(**backbone_cfg)
    elif name == 'GhostNet':
        return GhostNet(**backbone_cfg)
    elif name == 'MobileNetV2':
        return MobileNetV2(**backbone_cfg)
    elif name == 'EfficientNetLite':
        return EfficientNetLite(**backbone_cfg)
    # elif name == 'MobileNetV3_Small':
    #     return MobileNetV3_Small(**backbone_cfg)
    elif name == 'PP-MobileNetV3':
        return MobileNetV3(**backbone_cfg)
    # elif name == 'CustomCspNet':
    #     return CustomCspNet(**backbone_cfg)
    # elif name == 'RepVGG':
    #     return RepVGG(**backbone_cfg)
    else:
        raise NotImplementedError

