# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv

from mmcv.ops import DeformConv2d
from mmcv.cnn import Scale
from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmdet.core import multi_apply, reduce_mean
from .rotated_atss_head import RotatedATSSHead
from mmrotate.core import build_bbox_coder, multiclass_nms_rotated, build_prior_generator, build_bbox_coder
from mmrotate.core.bbox.iou_calculators import rbbox_overlaps
from ..builder import ROTATED_HEADS, build_loss
from .rotated_anchor_free_head import RotatedAnchorFreeHead
from .rotated_fcos_head import RotatedFCOSHead
# from ...core.bbox.coder.distance_angle_point_coder import obb2distance, distance2obb
# from ..detectors.utils import AlignConv
INF = 1e8

class AlignConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deform_groups=1):
        super(AlignConv, self).__init__()
        self.kernel_size = kernel_size
        # self.stride = stride
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deform_groups=deform_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.deform_conv, std=0.01)

    @torch.no_grad()
    def get_offset(self, anchors, featmap_size, stride):
        """Get the offset of AlignConv."""
        dtype, device = anchors.dtype, anchors.device
        feat_h, feat_w = featmap_size
        pad = (self.kernel_size - 1) // 2
        idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(idx, idx)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        # get sampling locations of default conv
        xc = torch.arange(0, feat_w, device=device, dtype=dtype)
        yc = torch.arange(0, feat_h, device=device, dtype=dtype)
        yc, xc = torch.meshgrid(yc, xc)
        xc = xc.reshape(-1)
        yc = yc.reshape(-1)
        x_conv = xc[:, None] + xx
        y_conv = yc[:, None] + yy

        # get sampling locations of anchors
        x_ctr, y_ctr, w, h, a = torch.unbind(anchors, dim=1)
        x_ctr, y_ctr, w, h = \
            x_ctr / stride, y_ctr / stride, \
            w / stride, h / stride
        cos, sin = torch.cos(a), torch.sin(a)
        dw, dh = w / self.kernel_size, h / self.kernel_size
        x, y = dw[:, None] * xx, dh[:, None] * yy
        xr = cos[:, None] * x - sin[:, None] * y
        yr = sin[:, None] * x + cos[:, None] * y
        x_anchor, y_anchor = xr + x_ctr[:, None], yr + y_ctr[:, None]
        # get offset filed
        offset_x = x_anchor - x_conv
        offset_y = y_anchor - y_conv
        # x, y in anchors is opposite in image coordinates,
        # so we stack them with y, x other than x, y
        offset = torch.stack([offset_y, offset_x], dim=-1)
        offset = offset.reshape(anchors.size(0),
                                -1).permute(1, 0).reshape(-1, feat_h, feat_w)
        return offset

    def forward(self, x, anchors, stride):
        """Forward function of AlignConv."""
        anchors = anchors.reshape(x.shape[0], x.shape[2], x.shape[3], 5)
        num_imgs, H, W = anchors.shape[:3]
        offset_list = [
            self.get_offset(anchors[i].reshape(-1, 5), (H, W), stride)
            for i in range(num_imgs)
        ]
        offset_tensor = torch.stack(offset_list, dim=0)
        x = self.relu(self.deform_conv(x, offset_tensor.detach()))
        return x

class TaskDecomposition(nn.Module):
    """Task decomposition module in task-aligned predictor of TOOD.

    Args:
        feat_channels (int): Number of feature channels in TOOD head.
        stacked_convs (int): Number of conv layers in TOOD head.
        la_down_rate (int): Downsample rate of layer attention.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
    """

    def __init__(self,
                 feat_channels,
                 stacked_convs,
                 la_down_rate=8,
                 conv_cfg=None,
                 norm_cfg=None):
        super(TaskDecomposition, self).__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        self.norm_cfg = norm_cfg
        self.layer_attention = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.in_channels // la_down_rate,
                self.stacked_convs,
                1,
                padding=0), nn.Sigmoid())

        self.reduction_conv = ConvModule(
            self.in_channels,
            self.feat_channels,
            1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=norm_cfg is None)

    def init_weights(self):
        for m in self.layer_attention.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        normal_init(self.reduction_conv.conv, std=0.01)

    def forward(self, feat, avg_feat=None):
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.layer_attention(avg_feat)

        # here we first compute the product between layer attention weight and
        # conv weight, and then compute the convolution between new conv weight
        # and feature map, in order to save memory and FLOPs.
        conv_weight = weight.reshape(
            b, 1, self.stacked_convs,
            1) * self.reduction_conv.conv.weight.reshape(
                1, self.feat_channels, self.stacked_convs, self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels,
                                          self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w)
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h,
                                                    w)
        if self.norm_cfg is not None:
            feat = self.reduction_conv.norm(feat)
        feat = self.reduction_conv.activate(feat)

        return feat

@ROTATED_HEADS.register_module()
class RotatedFCOSHead_VFL_Thead(RotatedAnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.
    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        separate_angle (bool): If true, angle prediction is separated from
            bbox regression loss. Default: False.
        scale_angle (bool): If true, add scale to angle pred branch. Default: True.
        h_bbox_coder (dict): Config of horzional bbox coder, only used when separate_angle is True.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_angle (dict): Config of angle loss, only used when separate_angle is True.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
    Example:
        >>> self = RotatedFCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, angle_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 scale_angle=True,
                 bbox_coder=dict(type='DistanceAnglePointCoder', angle_version='le90'),
                 loss_cls=dict(
                     type='VarifocalLoss',
                     use_sigmoid=True,
                     alpha=0.75,
                     gamma=2.0,
                     iou_weighted=True,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.is_scale_angle = scale_angle

        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        # self.loss_centerness = build_loss(loss_centerness)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        # if self.separate_angle: # we set it False
        #     self.loss_angle = build_loss(loss_angle)
        #     self.h_bbox_coder = build_bbox_coder(h_bbox_coder)
        # Angle predict length

    def _init_layers(self):
        """Initialize layers of the head."""
        self.inter_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            conv_cfg = self.conv_cfg
            chn = self.in_channels if i == 0 else self.feat_channels
            self.inter_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fuse_conv_cls = ConvModule(
                    self.feat_channels * 2,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg)
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.re_conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.conv_angle = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.re_conv_angle = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        if self.is_scale_angle:
            self.scale_angle = Scale(1.0)
        self.cls_decomp = TaskDecomposition(self.feat_channels,
                                            self.stacked_convs,
                                            self.stacked_convs * 8,
                                            self.conv_cfg, self.norm_cfg)
        self.reg_decomp = TaskDecomposition(self.feat_channels,
                                            self.stacked_convs,
                                            self.stacked_convs * 8,
                                            self.conv_cfg, self.norm_cfg)
        self.alignConv = AlignConv(self.feat_channels, self.feat_channels)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            # chn = self.in_channels if i == 0 else self.feat_channels
            self.reg_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.cls_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
    def init_weights(self):
        """Initialize weights of the head."""
        bias_cls = bias_init_with_prob(0.01)
        for m in self.inter_convs:
            normal_init(m.conv, std=0.01)
        self.cls_decomp.init_weights()
        self.reg_decomp.init_weights()

        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01)
        normal_init(self.conv_angle, std=0.01)
        normal_init(self.re_conv_reg, std=0.01)
        normal_init(self.re_conv_angle, std=0.01)
        normal_init(self.fuse_conv_cls.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)            
        
        self.alignConv.init_weights()

    def forward(self, feats):
        """Forward features from the upstream network.
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                angle_preds (list[Tensor]): Box angle for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides, list(range(len(feats))))

    def forward_single(self, x, scale, stride, idx):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.
        Returns:
            tuple: scores for each class, bbox predictions, angle predictions \
                and centerness predictions of input feature maps.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        # inter_feats = []
        # for inter_conv in self.inter_convs:
        #     x = inter_conv(x)
        #     inter_feats.append(x)
        # feat = torch.cat(inter_feats, 1)

        # avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        # cls_feat = self.cls_decomp(feat, avg_feat)
        # reg_feat = self.reg_decomp(feat, avg_feat)

        bbox_pred = self.conv_reg(reg_feat)

        # no use centerness 
        # if self.centerness_on_reg:
        #     centerness = self.conv_centerness(reg_feat)
        # else:
        #     centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16

        # bbox_pred = scale(bbox_pred).float()
        # bbox_pred = bbox_pred.clamp(min=0)
        # bbox_pred *= stride
        
        angle_pred = self.conv_angle(reg_feat)
        if self.is_scale_angle:
            angle_pred = self.scale_angle(angle_pred).float()
        # return cls_score, bbox_pred, angle_pred, centerness

        bbox_pred = torch.cat([bbox_pred, angle_pred], 1)

        num_imgs = x.size(0)
        points = self.prior_generator.single_level_grid_priors(
            angle_pred.size()[-2:],
            level_idx=idx,
            dtype=angle_pred.dtype,
            device=angle_pred.device)
        points = points.repeat(num_imgs, 1)
        anchor = self.bbox_coder.decode(points, bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5))
        align_feat = self.alignConv(reg_feat, anchor, stride)
        bbox_pred = self.re_conv_reg(align_feat)

        # norm box
        if self.norm_on_bbox:
            # bbox_pred needed for gradient computation has been modified
            # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
            # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()


        angle_pred = self.re_conv_angle(align_feat)
        if self.is_scale_angle:
            angle_pred = self.scale_angle(angle_pred).float()
        bbox_pred = torch.cat([bbox_pred, angle_pred], 1)
        # cls_feat concat
        # cls_score = self.conv_cls(self.fuse_conv_cls(torch.cat([cls_feat, align_feat], 1)))
        cls_score = self.conv_cls(cls_feat)
        return cls_score, bbox_pred

    # no use centerness 
    # @force_fp32(
    #     apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centernesses'))
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            angle_preds (list[Tensor]): Box angle for each scale level, \
                each is a 4D-tensor, the channel number is num_points * 1.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) 
            #    == len(angle_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        # we just use get_target 
        labels, bbox_targets= self.get_targets(
            all_level_points, gt_bboxes, gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
            for bbox_pred in bbox_preds
        ]
        # flatten_angle_preds = [
        #     angle_pred.permute(0, 2, 3, 1).reshape(-1, 1)
        #     for angle_pred in angle_preds
        # ]
        # flatten_centerness = [
        #     centerness.permute(0, 2, 3, 1).reshape(-1)
        #     for centerness in centernesses
        # ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        # flatten_angle_preds = torch.cat(flatten_angle_preds)
        # flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # flatten_angle_targets = torch.cat(angle_targets)

        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        # loss_cls = self.loss_cls(
        #     flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        # pos_angle_preds = flatten_angle_preds[pos_inds]
        # pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        # pos_angle_targets = flatten_angle_targets[pos_inds]
        # pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        pos_labels = flatten_labels[pos_inds]
        # centerness_denorm = max(
        #     reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            # if self.separate_angle:
            #     bbox_coder = self.h_bbox_coder
            # else:
            #     bbox_coder = self.bbox_coder
            #     pos_bbox_preds = torch.cat([pos_bbox_preds, pos_angle_preds],
            #                                dim=-1)
            #     pos_bbox_targets = torch.cat(
            #         [pos_bbox_targets, pos_angle_targets], dim=-1)
            pos_decoded_bbox_preds = self.bbox_coder.decode(pos_points,
                                                       pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)

            # cal iou_target for vfl 
            iou_targets = rbbox_overlaps(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds.detach(),
                is_aligned=True).clamp(min=1e-6)
            bbox_weights = iou_targets.clone().detach()
            bbox_avg_factor = reduce_mean(
                bbox_weights.sum()).clamp_(min=1).item()
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds.detach(),
                weight=bbox_weights,
                avg_factor=bbox_avg_factor)
            # loss_bbox = self.loss_bbox(
            #     pos_decoded_bbox_preds,
            #     pos_decoded_target_preds,
            #     weight=pos_centerness_targets,
            #     avg_factor=centerness_denorm)
            pos_ious = iou_targets.clone().detach()
            cls_iou_targets = torch.zeros_like(flatten_cls_scores)
            cls_iou_targets[pos_inds, pos_labels] = pos_ious

            # if self.separate_angle:
            #     loss_angle = self.loss_angle(
            #         pos_angle_preds, pos_angle_targets, avg_factor=num_pos)
            # loss_centerness = self.loss_centerness(
            #     pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum() * 0
            cls_iou_targets = torch.zeros_like(flatten_cls_scores)
            # loss_centerness = pos_centerness.sum()
            # if self.separate_angle:
            #     loss_angle = pos_angle_preds.sum()
        loss_cls = self.loss_cls(
                flatten_cls_scores,
                cls_iou_targets,
                avg_factor=num_pos)

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox)
        # if self.separate_angle:
        #     return dict(
        #         loss_cls=loss_cls,
        #         loss_bbox=loss_bbox,
        #         loss_angle=loss_angle)
            # return dict(
            #     loss_cls=loss_cls,
            #     loss_bbox=loss_bbox,
            #     loss_angle=loss_angle,
            #     loss_centerness=loss_centerness)
        # else:
            # return dict(
            #     loss_cls=loss_cls,
            #     loss_bbox=loss_bbox,
            #     loss_centerness=loss_centerness)
            # return dict(
            #     loss_cls=loss_cls,
            #     loss_bbox=loss_bbox)
            
    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
                concat_lvl_angle_targets (list[Tensor]): Angle targets of \
                    each level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list= multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        # angle_targets_list = [
        #     angle_targets.split(num_points, 0)
        #     for angle_targets in angle_targets_list
        # ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        # concat_lvl_angle_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            # angle_targets = torch.cat(
            #     [angle_targets[i] for angle_targets in angle_targets_list])
            if self.norm_on_bbox:
                bbox_targets =torch.cat( [bbox_targets[...,:4] / self.strides[i], bbox_targets[..., 4].unsqueeze(1)], 1)
            concat_lvl_bbox_targets.append(bbox_targets)
            # concat_lvl_angle_targets.append(angle_targets)
        # return (concat_lvl_labels, concat_lvl_bbox_targets,
        #         concat_lvl_angle_targets)
        return (concat_lvl_labels, concat_lvl_bbox_targets)

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression, classification and angle targets for a single
        image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 5))
                #    gt_bboxes.new_zeros((num_points, 4)), \
                #    gt_bboxes.new_zeros((num_points, 1))

        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)

        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(rot_matrix, offset[..., None])
        offset = offset.squeeze(-1)

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            stride = offset.new_zeros(offset.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
            inside_gt_bbox_mask = torch.logical_and(inside_center_bbox_mask,
                                                    inside_gt_bbox_mask)

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        angle_targets = gt_angle[range(num_points), min_area_inds]

        bbox_targets = torch.cat([bbox_targets, angle_targets], 1)
        # return labels, bbox_targets, angle_targets
        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)
        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    # @force_fp32(
    #     apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centernesses'))
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            angle_preds (list[Tensor]): Box angle for each scale level \
                with shape (N, num_points * 1, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 6) tensor, where the first 5 columns
                are bounding box positions (x, y, w, h, angle) and the 6-th
                column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        mlvl_points = self.prior_generator.grid_priors(featmap_sizes,
                                                       bbox_preds[0].dtype,
                                                       bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            # angle_pred_list = [
            #     angle_preds[i][img_id].detach() for i in range(num_levels)
            # ]
            # centerness_pred_list = [
            #     centernesses[i][img_id].detach() for i in range(num_levels)
            # ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list,
                                                #  angle_pred_list,
                                                #  centerness_pred_list,
                                                 mlvl_points, img_shape,
                                                 scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            angle_preds (list[Tensor]): Box angle for a single scale level \
                with shape (N, num_points * 1, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 1, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 6), where the first 5 columns
                are bounding box positions (x, y, w, h, angle) and the
                6-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        # mlvl_centerness = []
        for cls_score, bbox_pred, points in zip(
                cls_scores, bbox_preds, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            # centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            # angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, 1)
            # bbox_pred = torch.cat([bbox_pred, angle_pred], dim=1)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                # centerness = centerness[topk_inds]
            bboxes = self.bbox_coder.decode(
                points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            # mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            scale_factor = mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[..., :4] = mlvl_bboxes[..., :4] / scale_factor
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        # mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_nms_rotated(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=None)
        return det_bboxes, det_labels

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centerness'))
    def refine_bboxes(self, cls_scores, bbox_preds, angle_preds, centernesses):
        """This function will be used in S2ANet, whose num_anchors=1."""
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)
        num_imgs = cls_scores[0].size(0)
        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        # device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_points = self.prior_generator.grid_priors(featmap_sizes,
                                                       bbox_preds[0].dtype,
                                                       bbox_preds[0].device)
        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            bbox_pred = bbox_preds[lvl]
            angle_pred = angle_preds[lvl]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, 4)
            angle_pred = angle_pred.permute(0, 2, 3, 1)
            angle_pred = angle_pred.reshape(num_imgs, -1, 1)
            bbox_pred = torch.cat([bbox_pred, angle_pred], dim=-1)

            points = mlvl_points[lvl]

            for img_id in range(num_imgs):
                bbox_pred_i = bbox_pred[img_id]
                decode_bbox_i = self.bbox_coder.decode(points, bbox_pred_i)
                bboxes_list[img_id].append(decode_bbox_i.detach())

        return bboxes_list
