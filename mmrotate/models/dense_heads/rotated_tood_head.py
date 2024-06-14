# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import images_to_levels, multi_apply, unmap, reduce_mean
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmrotate.core import obb2hbb, rotated_anchor_inside_flags
from ..builder import ROTATED_HEADS, build_loss
from .rotated_retina_head import RotatedRetinaHead
from .rotated_atss_head import RotatedATSSHead
from .utils import get_num_level_anchors_inside
from mmrotate.core import build_assigner
from mmcv.runner import force_fp32
# from .common import ContextBlock2d
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
        self.feat_channels = feat_channels # 256
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs # 256 * 6 
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
class RotatedTOODHead(RotatedATSSHead):
    r"""An anchor-based head used in `ATSS
    <https://arxiv.org/abs/1912.02424>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    """  # noqa: W605
    def __init__(self,
                 initial_loss_cls=dict(
                                type='FocalLoss',
                                use_sigmoid=True,
                                # activated=True,
                                gamma=2.0,
                                alpha=0.25,
                                loss_weight=1.0),
                 **kwargs):
        super(RotatedTOODHead, self).__init__(**kwargs)
        self.epoch = 0
        # self.num_dcn = 2
        if self.train_cfg:
            self.initial_epoch = self.train_cfg.initial_epoch
            self.initial_assigner = build_assigner(
                self.train_cfg.initial_assigner)
            self.initial_loss_cls = build_loss(initial_loss_cls)
            self.assigner = self.initial_assigner
            self.alignment_assigner = build_assigner(self.train_cfg.assigner)
            self.alpha = self.train_cfg.alpha
            self.beta = self.train_cfg.beta

        # train_cfg=dict(
        # initial_epoch=4,
        # initial_assigner=dict(type='ATSSAssigner', topk=9),
        # assigner=dict(type='TaskAlignedAssigner', topk=13),
        # alpha=1,
        # beta=6,
        # allowed_border=-1,
        # pos_weight=-1,
        # debug=False),
    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)

        # self.inter_convs = nn.ModuleList()
        # for i in range(self.stacked_convs):
        #     # if i < self.num_dcn:
        #     #     conv_cfg = dict(type='DCNv2', deform_groups=4)
        #     # else:
        #     #     conv_cfg = self.conv_cfg
        #     conv_cfg = self.conv_cfg
        #     chn = self.in_channels if i == 0 else self.feat_channels
        #     self.inter_convs.append(
        #         ConvModule(
        #             chn,
        #             self.feat_channels,
        #             3,
        #             stride=1,
        #             padding=1,
        #             conv_cfg=conv_cfg,
        #             norm_cfg=self.norm_cfg))

        # self.cls_decomp = TaskDecomposition(self.feat_channels,
        #                                     self.stacked_convs,
        #                                     self.stacked_convs * 8,
        #                                     self.conv_cfg, self.norm_cfg)
        # self.reg_decomp = TaskDecomposition(self.feat_channels,
        #                                     self.stacked_convs,
        #                                     self.stacked_convs * 8,
        #                                     self.conv_cfg, self.norm_cfg)
        
        # self.tood_cls = nn.Conv2d(
        #     self.feat_channels,
        #     self.num_anchors * self.cls_out_channels,
        #     3,
        #     padding=1)
        # self.tood_reg = nn.Conv2d(
        #     self.feat_channels, self.num_anchors * 5, 3, padding=1)
        # self.tood_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 5, 3, padding=1)

    def init_weights(self):
        """Initialize weights of the head."""
        bias_cls = bias_init_with_prob(0.01)
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)

        # self.cls_decomp.init_weights()
        # self.reg_decomp.init_weights()

        # normal_init(self.tood_cls, std=0.01, bias=bias_cls)
        # normal_init(self.tood_reg, std=0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)


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
        cls_feat = x
        reg_feat = x

        # extract task interactive features
        # inter_feats = []
        # for inter_conv in self.inter_convs:
        #     x = inter_conv(x)
        #     inter_feats.append(x)
        # feat = torch.cat(inter_feats, 1)
        
        # task decomposition
        # avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        # cls_feat = self.cls_decomp(feat, avg_feat)
        # reg_feat = self.reg_decomp(feat, avg_feat)
        

        # add some global information 
        # reg_feat = self.gcBlock(feat, reg_feat)

        # cls_score = self.tood_cls(cls_feat)
        # bbox_pred = self.tood_reg(reg_feat)
        # objectness = self.tood_centerness(reg_feat)

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)

        # return cls_score, bbox_pred, objectness
        return cls_score, bbox_pred

    def _get_targets_single(self,   
                            cls_scores,
                            bbox_preds,
                            flat_anchors,
                            valid_flags,
                            num_level_anchors,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (torch.Tensor): Multi-level anchors of the image,
                which are concatenated into a single tensor of shape \
                (num_anchors, 5)
            valid_flags (torch.Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of \
                shape (num_anchors,).
            num_level_anchors (torch.Tensor): Number of anchors of each \
                scale level
            gt_bboxes (torch.Tensor): Ground truth bboxes of the image,
                shape (num_gts, 5).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (torch.Tensor): Ground truth bboxes to be \
                ignored, shape (num_ignored_gts, 5).
            img_meta (dict): Meta info of the image.
            gt_labels (torch.Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original \
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of all anchor
                label_weights_list (list[Tensor]): Label weights of all anchor
                bbox_targets_list (list[Tensor]): BBox targets of all anchor
                bbox_weights_list (list[Tensor]): BBox weights of all anchor
                pos_inds (int): Indices of positive anchor
                neg_inds (int): Indices of negative anchor
                sampling_result: object `SamplingResult`, sampling result.
        """
        inside_flags = rotated_anchor_inside_flags(
            flat_anchors, valid_flags, img_meta['img_shape'][:2],
            self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        # decoded bbox 
        decoded_bboxes = self.bbox_coder.decode(anchors, bbox_preds)

        if self.assign_by_circumhbbox is not None:
            num_level_anchors_inside = get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
            gt_bboxes_assign = obb2hbb(gt_bboxes, self.assign_by_circumhbbox)
            assign_result = self.assigner.assign(
                anchors, num_level_anchors_inside, gt_bboxes_assign,
                gt_bboxes_ignore, None if self.sampling else gt_labels)
        else:
            assign_result = self.alignment_assigner.assign(
                cls_scores,
                decoded_bboxes,
                anchors, 
                # num_level_anchors_inside,
                gt_bboxes, gt_bboxes_ignore,
                None if self.sampling else gt_labels)
        assign_ious = assign_result.max_overlaps
        assign_metrics = assign_result.assign_metrics

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        norm_alignment_metrics = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        # ## 
        class_assigned_gt_inds = torch.unique(
            sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = pos_inds[sampling_result.pos_assigned_gt_inds ==
                                     gt_inds]
            pos_alignment_metrics = assign_metrics[gt_class_inds]
            pos_ious = assign_ious[gt_class_inds]
            pos_norm_alignment_metrics = pos_alignment_metrics / (
                pos_alignment_metrics.max() + 10e-8) * pos_ious.max()
            norm_alignment_metrics[gt_class_inds] = pos_norm_alignment_metrics
        
        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            norm_alignment_metrics = unmap(norm_alignment_metrics,
                                           num_total_anchors, inside_flags)
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result, norm_alignment_metrics)

    def get_targets(self,
                    cls_scores,
                    bbox_preds,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each \
                image. The outer list indicates images, and the inner list \
                corresponds to feature levels of the image. Each element of \
                the inner list is a tensor of shape (num_anchors, 5).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of \
                each image. The outer list indicates images, and the inner \
                list corresponds to feature levels of the image. Each element \
                of the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be \
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original \
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of \
                    each level
                - bbox_targets_list (list[Tensor]): BBox targets of each level
                - bbox_weights_list (list[Tensor]): BBox weights of each level
                - num_total_pos (int): Number of positive samples in all images
                - num_total_neg (int): Number of negative samples in all images
            additional_returns: This function enables user-defined returns \
                from self._get_targets_single`. These returns are currently \
                refined to properties at each feature map (HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        if self.epoch < self.initial_epoch:
            (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
             pos_inds_list, neg_inds_list, sampling_results_list) = multi_apply(
                super()._get_targets_single,
                concat_anchor_list,
                concat_valid_flag_list,
                num_level_anchors_list,
                gt_bboxes_list,
                gt_bboxes_ignore_list,
                gt_labels_list,
                img_metas,
                label_channels=label_channels,
                unmap_outputs=unmap_outputs)
            all_assign_metrics = [
                weight[..., 0] for weight in all_bbox_weights
            ]
        else:
            (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
             pos_inds_list, neg_inds_list, sampling_results_list, all_assign_metrics) = multi_apply(
                self._get_targets_single,
                cls_scores,
                bbox_preds,
                concat_anchor_list,
                concat_valid_flag_list,
                num_level_anchors_list,
                gt_bboxes_list,
                gt_bboxes_ignore_list,
                gt_labels_list,
                img_metas,
                label_channels=label_channels,
                unmap_outputs=unmap_outputs)
        # (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
        #  pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        # rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        norm_alignment_metrics_list = images_to_levels(all_assign_metrics,
                                                       num_level_anchors)
        # res = (labels_list, label_weights_list, bbox_targets_list,
        #        bbox_weights_list, num_total_pos, num_total_neg)
        # if return_sampling_results:
        #     res = res + (sampling_results_list, )
        # for i, r in enumerate(rest_results):  # user-added return values
        #     rest_results[i] = images_to_levels(r, num_level_anchors)

        # return res + tuple(rest_results)
        # print(norm_alignment_metrics_list)
        return (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg, norm_alignment_metrics_list)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(img_metas)
        
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        # 将每层的cls 叠加在一起，用于label assign
        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ], 1)
        flatten_bbox_preds = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 5)
            for bbox_pred in bbox_preds
        ], 1)

        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bbox_preds,
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, alignment_metrics_list) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i, _ in enumerate(anchor_list):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)


        losses_cls, losses_bbox,\
        cls_avg_factors, bbox_avg_factors = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            alignment_metrics_list,
            num_total_samples=num_total_samples
            )

        # 为每个x除这个cls_avg_factor
        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))  

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, alignment_metrics, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (torch.Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (torch.Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W).
            anchors (torch.Tensor): Box reference for each scale level with
                shape (N, num_total_anchors, 5).
            labels (torch.Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (torch.Tensor): Label weights of each anchor with
                shape (N, num_total_anchors)
            bbox_targets (torch.Tensor): BBox regression targets of each anchor
            weight shape (N, num_total_anchors, 5).
            bbox_weights (torch.Tensor): BBox regression loss weights of each
                anchor with shape (N, num_total_anchors, 5).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            tuple (torch.Tensor):

                - loss_cls (torch.Tensor): cls. loss for each scale level.
                - loss_bbox (torch.Tensor): reg. loss for each scale level.
        """
        # classification loss
        labels = labels.reshape(-1)
        alignment_metrics = alignment_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels).contiguous()
        # target
        targets = labels if self.epoch < self.initial_epoch else (
            labels, alignment_metrics)
        cls_loss_func = self.initial_loss_cls \
            if self.epoch < self.initial_epoch else self.loss_cls

        loss_cls = cls_loss_func(
            cls_score, targets, label_weights, avg_factor=1.0)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 5)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)

        # loss_bbox = self.loss_bbox(
        #     bbox_pred,
        #     bbox_targets,
        #     bbox_weights,
        #     avg_factor=num_total_samples)
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            # pos_anchors = anchors[pos_inds]

            # we not decode 
            # pos_decode_bbox_pred = pos_bbox_pred
            # pos_decode_bbox_targets = pos_bbox_targets / stride[0]

            # regression loss
            # pos_bbox_weight = alignment_metrics[
            #     pos_inds].repeat(5).reshape(-1, 5)

            pos_bbox_weight = bbox_weights[pos_inds] if self.epoch < self.initial_epoch else alignment_metrics[
                pos_inds].repeat(5).reshape(-1, 5)


            loss_bbox = self.loss_bbox(
                pos_bbox_pred,
                pos_bbox_targets,
                weight=pos_bbox_weight,
                avg_factor=1.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.)
        return loss_cls, loss_bbox, alignment_metrics.sum(), pos_bbox_weight.sum()

