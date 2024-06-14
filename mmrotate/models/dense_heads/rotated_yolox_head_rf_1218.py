# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32
from mmdet.core import bbox_cxcywh_to_xyxy, multi_apply, reduce_mean
from mmdet.models.dense_heads import YOLOXHead
# from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, 
from mmrotate.core import build_bbox_coder, norm_angle
from ..builder import ROTATED_HEADS, build_loss
from mmcv.ops import DeformConv2d
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
        # [num_imgs]
        anchors = anchors.reshape(x.shape[0], x.shape[2], x.shape[3], 5)
        num_imgs, H, W = anchors.shape[:3]
        offset_list = [
            self.get_offset(anchors[i].reshape(-1, 5), (H, W), stride)
            for i in range(num_imgs)
        ]
        offset_tensor = torch.stack(offset_list, dim=0)
        x = self.relu(self.deform_conv(x, offset_tensor.detach()))
        return x


@ROTATED_HEADS.register_module()
class RotatedYOLOXHead_refine_1218(YOLOXHead):
    """Rotated YOLOXHead head used in `YOLOX.

    <https://arxiv.org/abs/2107.08430>`_.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels in stacking convs.
            Default: 256
        stacked_convs (int): Number of stacking convs of the head.
            Default: 2.
        strides (tuple): Downsample factor of each feature map.
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        dcn_on_last_conv (bool): If true, use dcn in the last layer of
            towers. Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer. Default: None.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_obj (dict): Config of objectness loss.
        loss_l1 (dict): Config of L1 loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
        separate_angle (bool): If true, angle prediction is separated from
            bbox regression loss. Default: False.
        with_angle_l1 (bool): If true, compute L1 loss with angle.
            Default: True.
        angle_norm_factor (float): Regularization factor of angle. Only
            used when with_angle_l1 is True
        angle_coder (dict): Config of angle coder.
        loss_angle (dict): Config of angle loss, only used when
            separate_angle is True.
    """

    def __init__(self,
                 separate_angle=False,
                 with_angle_l1=True,
                 angle_norm_factor=3.14,
                 edge_swap=None,
                 angle_coder=dict(type='PseudoAngleCoder'),
                 loss_angle=None,
                 loss_cls_rf=None,
                 loss_bbox_rf=dict(
                     type='IoULoss',
                     mode='square',
                     eps=1e-16,
                     reduction='sum',
                     loss_weight=5.0),
                 loss_l1_rf=None,
                 **kwargs):

        self.angle_coder = build_bbox_coder(angle_coder)
        self.angle_len = self.angle_coder.coding_len

        super().__init__(**kwargs)

        self.separate_angle = separate_angle
        self.with_angle_l1 = with_angle_l1
        self.angle_norm_factor = angle_norm_factor
        self.edge_swap = edge_swap
        if self.edge_swap:
            assert self.edge_swap in ['oc', 'le90', 'le135']
        if self.separate_angle:
            assert loss_angle is not None, \
                'loss_angle must be specified when separate_angle is True'
            self.loss_angle = build_loss(loss_angle)
        self.loss_bbox_rf = build_loss(loss_bbox_rf)
        self.loss_cls_rf = build_loss(loss_cls_rf)
        self.loss_l1_rf = build_loss(loss_l1_rf)
        
    def _init_layers(self):
        self.multi_level_cls_convs = nn.ModuleList()
        self.multi_level_cls_convs_rf = nn.ModuleList()
        self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_reg_convs_rf = nn.ModuleList()
        self.multi_level_conv_cls = nn.ModuleList()
        self.multi_level_conv_reg = nn.ModuleList()
        self.multi_level_conv_obj = nn.ModuleList()
        self.multi_level_conv_ang = nn.ModuleList()
        self.multi_level_conv_reg_rf = nn.ModuleList()
        self.multi_level_align_conv = nn.ModuleList()
        self.multi_level_conv_ang_rf = nn.ModuleList()
        self.multi_level_conv_cls_rf = nn.ModuleList()
        for _ in self.strides:
            self.multi_level_cls_convs.append(self._build_stacked_convs())
            self.multi_level_cls_convs_rf.append(self._build_stacked_convs())
            self.multi_level_reg_convs.append(self._build_stacked_convs())
            self.multi_level_reg_convs_rf.append(self._build_stacked_convs())
            conv_cls, conv_reg, conv_obj, conv_ang, conv_reg_rf, conv_ang_rf, conv_cls_rf= self._build_predictor()
            self.multi_level_conv_cls_rf.append(conv_cls_rf)
            self.multi_level_conv_cls.append(conv_cls)
            self.multi_level_conv_reg.append(conv_reg)
            self.multi_level_conv_obj.append(conv_obj)
            self.multi_level_conv_ang.append(conv_ang)
            self.multi_level_conv_reg_rf.append(conv_reg_rf)
            self.multi_level_conv_ang_rf.append(conv_ang_rf)
            self.multi_level_align_conv.append(AlignConv(self.feat_channels, self.feat_channels))
            

    def _build_predictor(self):
        """Initialize predictor layers of a single level head."""
        conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        conv_reg = nn.Conv2d(self.feat_channels, 4, 1)
        conv_obj = nn.Conv2d(self.feat_channels, 1, 1)
        conv_ang = nn.Conv2d(self.feat_channels, self.angle_len, 1)
        conv_reg_rf = nn.Conv2d(self.feat_channels, 4, 1)
        conv_ang_rf = nn.Conv2d(self.feat_channels, self.angle_len, 1)
        conv_cls_rf = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        return conv_cls, conv_reg, conv_obj, conv_ang, conv_reg_rf, conv_ang_rf, conv_cls_rf
    

    def init_weights(self):
        super(RotatedYOLOXHead_refine_1218, self).init_weights()
        # Use prior in model initialization to improve stability
        bias_init = bias_init_with_prob(0.01)
        for align_conv in self.multi_level_align_conv:
            align_conv.init_weights()
        for conv_ang in self.multi_level_conv_ang:
            conv_ang.bias.data.fill_(bias_init)
        for conv_ang_rf in self.multi_level_conv_ang_rf:
            conv_ang_rf.bias.data.fill_(bias_init)        

    def forward_single(self, x, cls_convs, reg_convs, conv_cls, conv_cls_rf, conv_reg,
                       conv_obj, conv_ang, conv_reg_rf, conv_ang_rf, cls_convs_rf, reg_convs_rf, align_conv, stride, idx):
        """Forward feature of a single scale level."""

        cls_feat = cls_convs(x)
        reg_feat = reg_convs(x)

        bbox_pred = conv_reg(reg_feat)
        angle_pred = conv_ang(reg_feat)
        cls_score = conv_cls(cls_feat)
        # 构造priors
        priors = self.prior_generator.single_level_grid_priors(
            angle_pred.size()[-2:],
            level_idx=idx,
            dtype=angle_pred.dtype,
            device=angle_pred.device,
            with_stride=True
        )
        num_imgs = x.size(0)
        priors = priors.repeat(num_imgs, 1)
        anchor, theta = self._bbox_decode_cxcywha(priors, bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4), angle_pred.permute(0, 2, 3, 1).reshape(-1, 1))
        rbox = torch.cat([anchor, theta], dim=-1)

        feat = align_conv(x, rbox, stride)

        cls_feat = cls_convs_rf(feat)
        reg_feat = reg_convs_rf(feat)
        
        cls_score_rf = conv_cls_rf(cls_feat)
        bbox_pred_rf = conv_reg_rf(reg_feat) 
        angle_pred_rf = conv_ang_rf(reg_feat) 
        objectness = conv_obj(reg_feat)
        rbox = rbox.reshape(num_imgs, -1, 5)
        return cls_score_rf, bbox_pred_rf, angle_pred_rf, objectness, rbox, cls_score, bbox_pred, angle_pred

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """

        return multi_apply(
            self.forward_single, feats, self.multi_level_cls_convs,
            self.multi_level_reg_convs, self.multi_level_conv_cls,self.multi_level_conv_cls_rf,
            self.multi_level_conv_reg, self.multi_level_conv_obj,
            self.multi_level_conv_ang, self.multi_level_conv_reg_rf,
            self.multi_level_conv_ang_rf,self.multi_level_cls_convs_rf,
            self.multi_level_reg_convs_rf,
            self.multi_level_align_conv,
            self.strides, list(range(len(feats))) )

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'objectnesses', 'rboxes', 'cls_rf', 'box_rf', 'angle_rf'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   angle_preds,
                   objectnesses,
                   rboxes,
                   cls_rf,
                   box_rf,
                   angle_rf,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network outputs of a batch into bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            angle_preds (list[Tensor]): Box angles for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.
        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 6) tensor, where the first 5 columns
                are bounding box positions (x, y, w, h, a) and the 5-th column
                is a score between 0 and 1. The second item is a (n,) tensor
                 where each item is the predicted class label of the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        cfg = self.test_cfg if cfg is None else cfg
        scale_factors = np.array(
            [img_meta['scale_factor'] for img_meta in img_metas])

        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                   self.angle_len)
            for angle_pred in angle_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]
        # flatten_rbox = [ rbox for rbox in rboxes]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_angle_preds = torch.cat(flatten_angle_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        flatten_priors = torch.cat(mlvl_priors)

        flatten_decoded_angle = self.angle_coder.decode(
            flatten_angle_preds).unsqueeze(-1)

        flatten_hbboxes_cxcywh, flatten_decoded_angle = \
            self._bbox_decode_cxcywha(
                flatten_priors, flatten_bbox_preds, flatten_decoded_angle)

        flatten_rbboxes = torch.cat(
            [flatten_hbboxes_cxcywh, flatten_decoded_angle], dim=-1)

        if rescale:
            flatten_rbboxes[..., :4] /= flatten_rbboxes.new_tensor(
                scale_factors).unsqueeze(1)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_scores = flatten_cls_scores[img_id]
            score_factor = flatten_objectness[img_id]
            bboxes = flatten_rbboxes[img_id]

            result_list.append(
                self._bboxes_nms(cls_scores, bboxes, score_factor, cfg))

        return result_list

    def _bboxes_nms(self, cls_scores, bboxes, score_factor, cfg):
        max_scores, labels = torch.max(cls_scores, 1)
        valid_mask = score_factor * max_scores >= cfg.score_thr

        bboxes = bboxes[valid_mask]
        scores = max_scores[valid_mask] * score_factor[valid_mask]
        labels = labels[valid_mask]

        if labels.numel() == 0:
            return bboxes, labels
        else:
            dets, keep = batched_nms(bboxes, scores, labels, cfg.nms)
            return dets, labels[keep]

    def _bbox_decode_cxcywha(self, priors, bbox_preds, decoded_angle):
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:]

        if self.edge_swap:
            w = whs[..., 0:1]
            h = whs[..., 1:2]
            w_regular = torch.where(w > h, w, h)
            h_regular = torch.where(w > h, h, w)
            theta_regular = torch.where(w > h, decoded_angle,
                                        decoded_angle + np.pi / 2)
            theta_regular = norm_angle(theta_regular, self.edge_swap)
            return torch.cat([xys, w_regular, h_regular],
                             dim=-1), theta_regular
        else:
            return torch.cat([xys, whs], -1), decoded_angle

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'objectnesses', 'rboxes', 'cls_rf', 'box_rf', 'angle_rf'))
    def loss(self,
             cls_scores,
             bbox_preds,
             angle_preds,
             objectnesses,
             rboxes,
             cls_rf,
             box_rf,
             angle_rf,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            angle_preds (list[Tensor]): Box angles for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [x, y, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """
        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)
        flatten_cls_preds_rf = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels)
            for cls_pred in cls_rf
        ]
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # rf
        flatten_bbox_preds_rf = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in box_rf
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                   self.angle_len)
            for angle_pred in angle_preds
        ]
        # rf
        flatten_angle_preds_rf = [
            angle_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                   self.angle_len)
            for angle_pred in angle_rf
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]
        flatten_rbboxes_rf = [ rbox for rbox in rboxes]
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_cls_preds_rf = torch.cat(flatten_cls_preds_rf, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_bbox_preds_rf = torch.cat(flatten_bbox_preds_rf, dim=1)
        flatten_angle_preds = torch.cat(flatten_angle_preds, dim=1)
        flatten_angle_preds_rf = torch.cat(flatten_angle_preds_rf, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_rbboxes_rf = torch.cat(flatten_rbboxes_rf, dim=1)
        flatten_priors = torch.cat(mlvl_priors)

        flatten_decoded_angle = self.angle_coder.decode(
            flatten_angle_preds).unsqueeze(-1)
        flatten_decoded_angle_rf = self.angle_coder.decode(
            flatten_angle_preds_rf).unsqueeze(-1)
        
        
        flatten_hbboxes_cxcywh, flatten_decoded_angle = \
            self._bbox_decode_cxcywha(
                flatten_priors, flatten_bbox_preds, flatten_decoded_angle)

        flatten_rbboxes = torch.cat(
            [flatten_hbboxes_cxcywh, flatten_decoded_angle], dim=-1)
        # flatten_hbboxes = bbox_cxcywh_to_xyxy(flatten_hbboxes_cxcywh)

        (pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets,
         num_fg_imgs) = multi_apply(
             self._get_target_single, flatten_cls_preds.detach(),
             flatten_objectness.detach(),
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_rbboxes.detach(), gt_bboxes, gt_labels)
        # rf
        (pos_masks_rf, cls_targets_rf, bbox_targets_rf, l1_targets_rf,
         num_fg_imgs_rf) = multi_apply(
             self._get_target_single_rf, flatten_cls_preds_rf.detach(),
             flatten_objectness.detach(),
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_rbboxes_rf.detach(), gt_bboxes, gt_labels)

        # The experimental results show that ‘reduce_mean’ can improve
        # performance on the COCO dataset.
        num_pos = torch.tensor(
            sum(num_fg_imgs),
            dtype=torch.float,
            device=flatten_cls_preds.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        # rf
        num_pos_rf = torch.tensor(
            sum(num_fg_imgs_rf),
            dtype=torch.float,
            device=flatten_cls_preds.device)
        num_total_samples_rf = max(reduce_mean(num_pos_rf), 1.0)

        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        
        pos_masks_rf = torch.cat(pos_masks_rf, 0)
        cls_targets_rf = torch.cat(cls_targets_rf, 0)
        bbox_targets_rf = torch.cat(bbox_targets_rf, 0)
        
        
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)
            l1_targets_rf = torch.cat(l1_targets_rf, 0)

        # Loss Bbox
        loss_bbox_rf = self.loss_bbox_rf(
            flatten_rbboxes_rf.view(-1, 5)[pos_masks_rf],
            bbox_targets_rf) / num_total_samples_rf

        loss_bbox = self.loss_bbox(
            flatten_rbboxes.view(-1, 5)[pos_masks],
            bbox_targets) / num_total_samples

        # Loss Objectness and Loss Cls
        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1),
                                 obj_targets) / num_total_samples
        loss_cls = self.loss_cls(
            flatten_cls_preds.view(-1, self.num_classes)[pos_masks],
            cls_targets) / num_total_samples
        loss_cls_rf = self.loss_cls_rf(
                    flatten_cls_preds_rf.view(-1, self.num_classes)[pos_masks_rf],
                    cls_targets_rf) / num_total_samples_rf
        
        
        loss_dict = dict(
            loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj, loss_cls_rf=loss_cls_rf, loss_bbox_rf=loss_bbox_rf)

        # if self.separate_angle:
        #     loss_dict.update(loss_angle=loss_angle)

        # Loss L1
        if self.use_l1:
            if self.with_angle_l1:
                flatten_rbbox_preds = torch.cat([
                    flatten_bbox_preds,
                    flatten_decoded_angle / self.angle_norm_factor
                ],
                                                dim=-1)
                loss_l1 = self.loss_l1(
                    flatten_rbbox_preds.view(-1, 5)[pos_masks],
                    l1_targets) / num_total_samples
                # rf
                flatten_rbbox_preds_rf = torch.cat([
                    flatten_bbox_preds_rf,
                    flatten_decoded_angle_rf / self.angle_norm_factor
                ],
                                                dim=-1)
                loss_l1_rf = self.loss_l1(
                    flatten_rbbox_preds_rf.view(-1, 5)[pos_masks_rf],
                    l1_targets_rf) / num_total_samples_rf
            else:
                loss_l1 = self.loss_l1(
                    flatten_bbox_preds.view(-1, 4)[pos_masks],
                    l1_targets) / num_total_samples

            loss_dict.update(loss_l1=loss_l1, loss_l1_rf=loss_l1_rf)

        return loss_dict

    @torch.no_grad()
    def _get_target_single(self, cls_preds, objectness, priors, decoded_bboxes,
                           gt_bboxes, gt_labels):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 5] in [x, y, w, h, a]
                format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 5] in [x, y, w, h, a] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """

        num_priors = priors.size(0)
        num_gts = gt_labels.size(0)
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 5))
            if self.with_angle_l1:
                l1_target = cls_preds.new_zeros((0, 5))
            else:
                l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return (foreground_mask, cls_target, obj_target, bbox_target,
                    l1_target, 0)

        offset_priors = torch.cat(
            [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)
        
        assign_result = self.assigner.assign(
            cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid(),
            offset_priors, decoded_bboxes, gt_bboxes, gt_labels)
        
        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)
        
        pos_ious = assign_result.max_overlaps[pos_inds]
        # IOU aware classification score
        cls_target = F.one_hot(sampling_result.pos_gt_labels,
                               self.num_classes) * pos_ious.unsqueeze(-1)
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        l1_target = cls_preds.new_zeros((num_pos_per_img, 4))
        if self.use_l1:
            l1_target = self._get_l1_target(l1_target, bbox_target,
                                            priors[pos_inds])
        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1
        return (foreground_mask, cls_target, obj_target, bbox_target,
                l1_target, num_pos_per_img)

    @torch.no_grad()
    def _get_target_single_rf(self, cls_preds, objectness, priors, decoded_bboxes,
                           gt_bboxes, gt_labels):
        num_priors = priors.size(0)
        num_gts = gt_labels.size(0)
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 5))
            if self.with_angle_l1:
                l1_target = cls_preds.new_zeros((0, 5))
            else:
                l1_target = cls_preds.new_zeros((0, 4))
            # obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            # return (foreground_mask, cls_target, bbox_target,
            #         l1_target, 0)
            return (foreground_mask, cls_target, bbox_target, l1_target, 0)

        offset_priors = torch.cat(
            [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)
        
        assign_result = self.assigner.assign(
            cls_preds.sigmoid(),
            offset_priors, decoded_bboxes, gt_bboxes, gt_labels)
        
        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)
        
        pos_ious = assign_result.max_overlaps[pos_inds]
        # IOU aware classification score
        cls_target = F.one_hot(sampling_result.pos_gt_labels,
                               self.num_classes) * pos_ious.unsqueeze(-1)
        # obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        # obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        l1_target = cls_preds.new_zeros((num_pos_per_img, 4))
        if self.use_l1:
            l1_target = self._get_l1_target(l1_target, bbox_target,
                                            priors[pos_inds])
        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1
        return (foreground_mask, cls_target, bbox_target, l1_target, num_pos_per_img)

    def _get_l1_target(self, l1_target, gt_bboxes, priors, eps=1e-8):
        """Convert gt bboxes to center offset and log width height."""
        gt_cxcywh = gt_bboxes[..., :4]
        l1_target[:, :2] = (gt_cxcywh[:, :2] - priors[:, :2]) / priors[:, 2:]
        l1_target[:, 2:] = torch.log(gt_cxcywh[:, 2:] / priors[:, 2:] + eps)
        if self.with_angle_l1:
            angle_target = gt_bboxes[..., 4:5] / self.angle_norm_factor
            return torch.cat([l1_target, angle_target], dim=-1)
        else:
            return l1_target
