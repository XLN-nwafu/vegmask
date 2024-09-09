# Copyright (c) Facebook, Inc. and its affiliates.
from .build import build_backbone, BACKBONE_REGISTRY  # noqa F401 isort:skip

from .backbone import Backbone
from .fpn import FPN
from .regnet import RegNet
from .resnet import (
    BasicStem,
    ResNet,
    ResNetBlockBase,
    build_resnet_backbone,
    make_stage,
    BottleneckBlock,
    # build_resnet_pafpn_backbone,
)
from .vit import ViT, SimpleFeaturePyramid, get_vit_lr_decay_rate
from .swin import SwinTransformer
from .mobilenetv2 import build_mnv2_backbone, build_mnv2_fpn_backbone, build_fcos_mnv2_fpn_backbone,build_mnv2_pafpn_backbone
from .mobilenetv3 import build_mnv3l_backbone,build_mnv3s_backbone,build_mnv3s_fpn_backbone,build_mnv3l_fpn_backbone,build_mnv3l_pafpn_backbone
from .mobilenext import build_mnext_fpn_backbone,build_mnext_backbone
from .mv2_ca import build_mnv2_ca_fpn_backbone,build_mnv2_ca_backbone,build_mnv2_ca_pafpn_backbone
from .efficientnet import build_efficientnet_fpn_backbone,build_efficientnet_bifpn_backbone,build_efficientnet_pafpn_backbone
from .PAFPN import PAFPN,build_resnet_pafpn_backbone
from .IEFPN import IEFPN


__all__ = [k for k in globals().keys() if not k.startswith("_")]
# TODO can expose more resnet blocks after careful consideration
