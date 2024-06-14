# Copyright (c) OpenMMLab. All rights reserved.

from cmath import exp
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import points_in_polygons
from mmcv.cnn import Scale
from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmdet.core import multi_apply, reduce_mean
from .rotated_atss_head import RotatedATSSHead
from mmrotate.core import build_bbox_coder, multiclass_nms_rotated, build_prior_generator
from mmrotate.core.bbox.iou_calculators import build_iou_calculator
# from mmrotate.core.bbox.iou_calculators import rbbox_overlaps
from ..builder import ROTATED_HEADS, build_loss
from .rotated_anchor_free_head import RotatedAnchorFreeHead
from .rotated_fcos_head import RotatedFCOSHead
from ...core.bbox.transforms import obb2poly
# from ...core.bbox.coder.distance_angle_point_coder import obb2distance, distance2obb
# obb2distance encode
# distance2obb decode 
INF = 1e8
EPS = 1e-12

def levels_to_images(mlvl_tensor):
    """Concat multi-level feature maps by image.

    [feature_level0, feature_level1...] -> [feature_image0, feature_image1...]
    将各个特征层的(N, C, H, W)此处的H, W是不相等的
    转变为N个image H*W, C 的列表。
    Convert the shape of each element in mlvl_tensor from (N, C, H, W) to
    (N, H*W , C), then split the element to N elements with shape (H*W, C), and
    concat elements in same image of all level along first dimension.

    Args:
        mlvl_tensor (list[torch.Tensor]): list of Tensor which collect from
            corresponding level. Each element is of shape (N, C, H, W)

    Returns:
        list[torch.Tensor]: A list that contains N tensors and each tensor is
            of shape (num_elements, C)
    """
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:
        t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]

class CenterPrior(nn.Module):
    def __init__(self,
                 soft_prior=True,
                 num_classes=80,
                 strides=(8, 16, 32, 64, 128)):
        super(CenterPrior, self).__init__()
        self.mean = nn.Parameter(torch.zeros(num_classes, 2), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones(num_classes, 2)+0.11, requires_grad=False)
        self.strides = strides
        self.soft_prior = soft_prior

    def forward(self, anchor_points_list, gt_bboxes, labels,
                inside_gt_bbox_mask):

        inside_gt_bbox_mask = inside_gt_bbox_mask.clone()
        num_gts = len(labels)
        num_points = sum([len(item) for item in anchor_points_list])
        if num_gts == 0:
            return gt_bboxes.new_zeros(num_points,
                                    num_gts), inside_gt_bbox_mask
        center_prior_list = []
        for slvl_points, stride in zip(anchor_points_list, self.strides):
            # slvl_points: points from single level in FPN, has shape (h*w, 2)
            # single_level_points has shape (h*w, num_gt, 2)
            single_level_points = slvl_points[:, None, :].expand(
                (slvl_points.size(0), len(gt_bboxes), 2))
            # gt_center_x = ((gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2)
            # gt_center_y = ((gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2)
            gt_center_x = gt_bboxes[:, 0]
            gt_center_y = gt_bboxes[:, 1] 
            gt_center = torch.stack((gt_center_x, gt_center_y), dim=1)
            gt_center = gt_center[None]
            # instance_center has shape (1, num_gt, 2)
            instance_center = self.mean[labels][None]
            # instance_sigma has shape (1, num_gt, 2)
            instance_sigma = self.sigma[labels][None]
            # distance has shape (num_points, num_gt, 2)
            distance = (((single_level_points - gt_center) / float(stride) -
                        instance_center)**2)
            center_prior = torch.exp(-distance /
                                     (2 * instance_sigma**2)).prod(dim=-1)
            center_prior_list.append(center_prior)
        center_prior_weights = torch.cat(center_prior_list, dim=0)
        if not self.soft_prior:
            prior_mask = center_prior_weights > 0.3
            center_prior_weights[prior_mask] = 1
            center_prior_weights[~prior_mask] = 0

        center_prior_weights[~inside_gt_bbox_mask] = 0
        return center_prior_weights, inside_gt_bbox_mask

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
class RotatedFCOSHead_Thead_DW(RotatedAnchorFreeHead):
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
                 strides=[8, 16, 32, 64, 128],
                 stacked_convs=6,
                 feat_channels=256,
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 separate_angle=False,
                 scale_angle=True,
                 soft_prior=True,
                 prior_offset=0.5,
                 bbox_coder=dict(type='DistanceAnglePointCoder', angle_version='le90'),
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
        self.angle_version ='le90'
        self.num_classes = num_classes
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        self.strides = strides
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.separate_angle = separate_angle
        self.is_scale_angle = scale_angle

        super().__init__(
            num_classes,
            in_channels,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.center_prior = CenterPrior(
            soft_prior=soft_prior,
            num_classes=self.num_classes,
            strides=self.strides)
        self.prior_generator.offset = prior_offset
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.iou_calculator = build_iou_calculator(dict(type='RBboxOverlaps2D'))

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
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.conv_angle = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
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
        normal_init(self.conv_centerness, std=0.01)

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
                           self.strides)

    def forward_single(self, x, scale, stride):
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

        inter_feats = []
        for inter_conv in self.inter_convs:
            x = inter_conv(x)
            inter_feats.append(x)
        feat = torch.cat(inter_feats, 1)

        avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        cls_feat = self.cls_decomp(feat, avg_feat)
        reg_feat = self.reg_decomp(feat, avg_feat)
        cls_score = self.conv_cls(cls_feat)
        bbox_pred = self.conv_reg(reg_feat)
        centerness = self.conv_centerness(reg_feat)
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            # bbox_pred needed for gradient computation has been modified
            # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
            # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        angle_pred = self.conv_angle(reg_feat)
        if self.is_scale_angle:
            angle_pred = self.scale_angle(angle_pred).float()
        return cls_score, bbox_pred, angle_pred, centerness

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'angle_preds','objectnesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             angle_preds,
             objectnesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) \
               == len(angle_preds) == len(objectnesses)
        all_num_gt = sum([len(gt_bbox) for gt_bbox in gt_bboxes])
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        # 正样本点在GT内
        inside_gt_bbox_mask_list, bbox_targets_list, angle_targets_list = self.get_targets(
            all_level_points, gt_bboxes) 
        ###
        center_prior_weight_list = []
        temp_inside_gt_bbox_mask_list = []
        for gt_bboxe, gt_label, inside_gt_bbox_mask in zip(gt_bboxes, gt_labels, inside_gt_bbox_mask_list):
            center_prior_weight, inside_gt_bbox_mask = self.center_prior(all_level_points, gt_bboxe, gt_label, inside_gt_bbox_mask)
            center_prior_weight_list.append(center_prior_weight)
            temp_inside_gt_bbox_mask_list.append(inside_gt_bbox_mask)
        inside_gt_bbox_mask_list = temp_inside_gt_bbox_mask_list

        mlvl_points = torch.cat(all_level_points, dim=0)
        # concat preds and angle
        # bbox_preds = torch.cat([bbox_preds, angle_preds], dim=1)
        bbox_preds = levels_to_images(bbox_preds)
        angle_preds = levels_to_images(angle_preds)
        cls_scores = levels_to_images(cls_scores)
        objectnesses = levels_to_images(objectnesses)

        reg_loss_list = []
        ious_list = []
        num_points = len(mlvl_points)

        for bbox_pred, angle_pred, gt_bboxe, gt_angle, inside_gt_bbox_mask in zip(
                bbox_preds, angle_preds, bbox_targets_list, angle_targets_list, inside_gt_bbox_mask_list):
            temp_num_gt = gt_bboxe.size(1)
            expand_mlvl_points = mlvl_points[:, None, :].expand(
                num_points, temp_num_gt, 2).reshape(-1, 2)
            gt_bboxe = gt_bboxe.reshape(-1, 4)
            gt_angle = gt_angle.reshape(-1, 1)
            expand_bbox_pred = bbox_pred[:, None, :].expand(
                num_points, temp_num_gt, 4).reshape(-1, 4)
            expand_angle_pred = angle_pred[:, None, :].expand(
                num_points, temp_num_gt, 1).reshape(-1, 1)
            expand_bbox_pred = torch.cat([expand_bbox_pred, expand_angle_pred], dim=1)
            decoded_bbox_preds = self.bbox_coder.decode(expand_mlvl_points,
                                               expand_bbox_pred)
            gt_bboxe = torch.cat([gt_bboxe, gt_angle], dim=1)
            decoded_target_preds = self.bbox_coder.decode(expand_mlvl_points, gt_bboxe)
            with torch.no_grad():
                ious = self.iou_calculator(
                    decoded_bbox_preds, decoded_target_preds, is_aligned=True)
                ious = ious.reshape(num_points, temp_num_gt)
                if temp_num_gt:
                    ious = ious
                else:
                    ious = ious.new_zeros(num_points, temp_num_gt)
                ious[~inside_gt_bbox_mask] = 0
                ious_list.append(ious)
            loss_bbox = self.loss_bbox(
                decoded_bbox_preds,
                decoded_target_preds, 
                weight=None,
                reduction_override='none')
            reg_loss_list.append(loss_bbox.reshape(num_points, temp_num_gt))

        cls_scores = [item.sigmoid() for item in cls_scores]
        objectnesses = [item.sigmoid() for item in objectnesses]
        cls_loss_list, loc_loss_list, cls_neg_loss_list, neg_avg_factor_list = multi_apply(self._loss_single, cls_scores,
                                    objectnesses, reg_loss_list, gt_labels, center_prior_weight_list, ious_list, inside_gt_bbox_mask_list)

        pos_avg_factor = reduce_mean(
            bbox_pred.new_tensor(all_num_gt)).clamp_(min=1)
        neg_avg_factor = sum(item.data.sum()
                            for item in neg_avg_factor_list).float()
        neg_avg_factor = reduce_mean(neg_avg_factor).clamp_(min=1)
        cls_loss = sum(cls_loss_list) / pos_avg_factor
        loc_loss = sum(loc_loss_list) / pos_avg_factor
        cls_neg_loss = sum(cls_neg_loss_list) / neg_avg_factor

        loss = dict(
            loss_cls_pos=cls_loss, loss_loc=loc_loss, loss_cls_neg=cls_neg_loss)
        return loss
            
    def get_targets(self, points, gt_bboxes_list):
        # 将point拼接，从get_target_single得到在gt内的mask
        concat_points = torch.cat(points, dim=0)
        inside_gt_bbox_mask_list, bbox_targets_list, angle_targets_list = multi_apply(
            self._get_target_single, gt_bboxes_list, points=concat_points)
        return inside_gt_bbox_mask_list, bbox_targets_list, angle_targets_list

    def _get_target_single(self, gt_bboxes, points):
        num_points = points.size(0)
        num_gts = gt_bboxes.size(0)
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
        angle_targets = gt_angle
        if num_gts:
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
            
        else:
            inside_gt_bbox_mask = bbox_targets.new_zeros((num_points, num_gts),
                                                         dtype=torch.bool)
        return inside_gt_bbox_mask, bbox_targets, angle_targets

    def _loss_single(self, cls_score, objectness, reg_loss, gt_labels,
                            center_prior_weights, ious, inside_gt_bbox_mask):
        num_gts = len(gt_labels)
        joint_conf = (cls_score * objectness)
        #To more precisely estimate the consistency degree between cls and reg heads, we represent IoU score as an expentional function of the reg loss.
        p_loc = torch.exp(-reg_loss*5)
        p_cls = (cls_score * objectness)[:, gt_labels] 
        p_pos = p_cls * p_loc
        
        p_neg_weight = torch.ones_like(joint_conf)
        neg_metrics = torch.zeros_like(ious).fill_(-1)
        alpha = 2
        t = lambda x: 1/(0.5**alpha-1)*x**alpha - 1/(0.5**alpha-1)
        if num_gts > 0:
            def normalize(x): 
                x_ = t(x)
                t1 = x_.min()
                t2 = min(1., x_.max())
                y = (x_ - t1 + EPS ) / (t2 - t1 + EPS )
                y[x<0.5] = 1
                return y
            for instance_idx in range(num_gts):
                idxs = inside_gt_bbox_mask[:, instance_idx]
                if idxs.any():
                    neg_metrics[idxs, instance_idx] = normalize(ious[idxs, instance_idx])
            foreground_idxs = torch.nonzero(neg_metrics != -1, as_tuple=True)
            p_neg_weight[foreground_idxs[0],
                         gt_labels[foreground_idxs[1]]] = neg_metrics[foreground_idxs]
        
        p_neg_weight = p_neg_weight.detach()
        neg_avg_factor = (1 - p_neg_weight).sum()
        p_neg_weight = p_neg_weight * joint_conf ** 2
        neg_loss = p_neg_weight * F.binary_cross_entropy(joint_conf, torch.zeros_like(joint_conf), reduction='none')
        neg_loss = neg_loss.sum() 
        
        p_pos_weight = (torch.exp(5*p_pos) * p_pos * center_prior_weights) / (torch.exp(3*p_pos) * p_pos * center_prior_weights).sum(0, keepdim=True).clamp(min=EPS)
        p_pos_weight = p_pos_weight.detach()
        
        cls_loss = F.binary_cross_entropy(
            p_cls,
            torch.ones_like(p_cls),
            reduction='none') * p_pos_weight 
        loc_loss = F.binary_cross_entropy(
            p_loc,
            torch.ones_like(p_loc),
            reduction='none') * p_pos_weight 
        cls_loss = cls_loss.sum() * 0.25
        loc_loss = loc_loss.sum() * 0.25
        
        return cls_loss, loc_loss, neg_loss, neg_avg_factor

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'objectnesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   angle_preds,
                   objectnesses,
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
            angle_pred_list = [
                angle_preds[i][img_id].detach() for i in range(num_levels)
            ]
            # centerness_pred_list = [
            #     centernesses[i][img_id].detach() for i in range(num_levels)
            # ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list,
                                                 angle_pred_list,
                                                #  centerness_pred_list,
                                                 mlvl_points, img_shape,
                                                 scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           angle_preds,
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
        for cls_score, bbox_pred, angle_pred, points in zip(
                cls_scores, bbox_preds, angle_preds,
                mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            # centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, 1)
            bbox_pred = torch.cat([bbox_pred, angle_pred], dim=1)
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


 
