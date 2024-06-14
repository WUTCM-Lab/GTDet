# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmcv.cnn import bias_init_with_prob, normal_init
from ..utils import ORConv2d
from ..builder import ROTATED_HEADS
# from .rotated_anchor_head import RotatedAnchorHead
# from .rotated_retina_head import RotatedRetinaHead
from .odm_refine_head import ODMRefineHead

class ESEAttn(nn.Module):
    def __init__(self, feat_channels):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        # self.conv = ConvBNLayer(feat_channels, feat_channels, 1, act=act)
        self.conv = nn.Conv2d(
            in_channels=feat_channels,
            out_channels=feat_channels,
            kernel_size=1
        )
        self.bn = nn.BatchNorm2d(feat_channels)
        self.relu = nn.ReLU(inplace=True)
        self.init_weights()

    def init_weights(self):
        normal_init(self.fc.weight, std=0.01)

    def forward(self, feat, avg_feat):
        weight = F.sigmoid(self.fc(avg_feat))
        return self.relu(self.bn(self.conv(feat * weight)))

@ROTATED_HEADS.register_module()
class FineHead(ODMRefineHead):
    r"""An anchor-based head used in `RotatedRetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int, optional): Number of stacked convolutions.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        anchor_generator (dict): Config dict for anchor generator
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='PseudoAnchorGenerator',
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):

        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(FineHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_ese = ESEAttn(self.feat_channels)
        self.reg_ese = ESEAttn(self.feat_channels)
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 5, 3, padding=1)
        self.or_conv = ORConv2d(
            self.feat_channels,
            int(self.feat_channels / 8),
            kernel_size=3,
            padding=1,
            arf_config=(1, 8))
    # def init_weights(self):
    #     bias_cls = bias_init_with_prob(0.01)

    #     normal_init(self.retina_cls, std=0.01, bias=bias_cls)
    #     normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (torch.Tensor): Features of a single scale level.

        Returns:
            tuple (torch.Tensor):

                - cls_score (torch.Tensor): Cls scores for a single scale \
                    level the channels number is num_anchors * num_classes.
                - bbox_pred (torch.Tensor): Box energies / deltas for a \
                    single scale level, the channels number is num_anchors * 5.
        """
        x = self.or_conv(x)
        avg_feat = F.adaptive_avg_pool2d(x, (1, 1))
        cls_feat = self.cls_ese(x, avg_feat)
        reg_feat = self.reg_ese(x, avg_feat)

        cls_score = self.retina_cls(cls_feat + x)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred



