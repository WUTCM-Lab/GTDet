# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
from mmcv.cnn import Scale
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean
from mmrotate.core.bbox.iou_calculators import rbbox_overlaps
from mmrotate.core import build_bbox_coder, multiclass_nms_rotated
from ..builder import ROTATED_HEADS, build_loss
from .rotated_anchor_free_head import RotatedAnchorFreeHead

INF = 1e8


@ROTATED_HEADS.register_module()
class RotatedFCOSHeadGFL(RotatedAnchorFreeHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 scale_angle=True,
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
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        # self.norm_on_bbox = norm_on_bbox
        # self.centerness_on_reg = centerness_on_reg
        # self.separate_angle = separate_angle
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
        # if self.separate_angle:
        #     self.loss_angle = build_loss(loss_angle)
        #     self.h_bbox_coder = build_bbox_coder(h_bbox_coder)
        # Angle predict length
        self.sampling = False
        # if self.train_cfg:
        #     self.assigner = build_assigner(self.train_cfg.assigner)
        #     # sampling=False so use PseudoSampler
        #     sampler_cfg = dict(type='PseudoSampler')
        #     self.sampler = build_sampler(sampler_cfg, context=self)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.conv_angle = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        if self.is_scale_angle:
            self.scale_angle = Scale(1.0)

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
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        # if self.centerness_on_reg:
        #     centerness = self.conv_centerness(reg_feat)
        # else:
        #     centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        # if self.norm_on_bbox:
        #     # bbox_pred needed for gradient computation has been modified
        #     # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
        #     # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
        #     bbox_pred = bbox_pred.clamp(min=0)
        #     if not self.training:
        #         bbox_pred *= stride
        # else:
        #     bbox_pred = bbox_pred.exp()
        bbox_pred = bbox_pred.exp()
        angle_pred = self.conv_angle(reg_feat)
        if self.is_scale_angle:
            angle_pred = self.scale_angle(angle_pred).float()
        return cls_score, bbox_pred, angle_pred

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             angle_preds,
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
        assert len(cls_scores) == len(bbox_preds) \
               == len(angle_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)

        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs,-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs,-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(num_imgs,-1, 1)
            for angle_pred in angle_preds
        ]
        print(flatten_cls_scores[0].shape)
        print(flatten_cls_scores[1].shape)
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1)
        print(flatten_cls_scores.shape)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_angle_preds = torch.cat(flatten_angle_preds, dim=1)

        flatten_bbox_preds = torch.cat([flatten_bbox_preds, flatten_angle_preds], dim = -1)   
        pos_inds, cls_iou_targets, bbox_targets, bbox_weights, decoded_bboxes = self.get_targets(
             flatten_cls_scores, flatten_bbox_preds, gt_bboxes, gt_labels,
             all_level_points
        )
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])
        # bbox_targets 暂时没拿到
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_cls_iou_targets = torch.cat(cls_iou_targets)
        flatten_bbox_weights = torch.cat(bbox_weights)
        flatten_decoded_bbox = torch.cat(decoded_bboxes)
        flatten_pos_inds = torch.cat(pos_inds)
        print(f'flatten_bbox_targets.shape:{flatten_bbox_targets.shape}')
        print(f'flatten_cls_iou_targets.shape:{flatten_cls_iou_targets.shape}')
        print(f'flatten_bbox_weights.shape:{flatten_bbox_weights.shape}')
        print(f'flatten_decoded_bbox.shape:{flatten_decoded_bbox.shape}')
        print(f'flatten_pos_inds.shape:{flatten_pos_inds.shape}')
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        # bg_class_ind = self.num_classes
        # pos_inds = ((flatten_labels >= 0)
        #             & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(flatten_pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(flatten_pos_inds), 1.0)
        # pos_bbox_preds = flatten_bbox_preds[pos_inds]
        # pos_angle_preds = flatten_angle_preds[pos_inds]
        # pos_centerness = flatten_centerness[pos_inds]
        # pos_bbox_targets = flatten_bbox_targets[pos_inds]
        # pos_angle_targets = flatten_angle_targets[pos_inds]
        # pos_centerness_targets = self.centerness_target(pos_bbox_targets)


        # centerness weighted iou loss
        bbox_avg_factor = reduce_mean(
                flatten_bbox_weights.sum()).clamp_(min=1).item()
        if len(flatten_pos_inds) > 0:
            loss_bbox = self.loss_bbox(
                flatten_decoded_bbox[pos_inds],
                flatten_bbox_targets,
                weight=flatten_bbox_weights,
                avg_factor=bbox_avg_factor)
            # loss_centerness = self.loss_centerness(
            #     pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = flatten_bbox_preds[flatten_pos_inds].sum()
            # loss_centerness = pos_centerness.sum()
        loss_cls = self.loss_cls(
            flatten_cls_scores,
            flatten_cls_iou_targets, avg_factor=num_pos)
        return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox)

    @torch.no_grad()
    def _get_target_single(self, cls_preds, bbox_preds, gt_bboxes, gt_labels, points, num_points_per_lvl):
        eps=1e-7
        num_gts = gt_labels.size(0)
        num_points = points.size(0)
        num_bboxes = bbox_preds.size(0)
        decoded_bboxes = self.bbox_coder.decode(points, bbox_preds)
        cls_preds = cls_preds.sigmoid()

        if num_gts == 0:
            # cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 5))
            cls_iou_targets = torch.zeros_like(cls_preds)
            matched_gt_inds = decoded_bboxes.new_full((num_bboxes, ),
                                                   0,
                                                   dtype=torch.long)
            bbox_weights =  decoded_bboxes.new_full((num_bboxes, ),
                                                 0,
                                                 dtype=torch.float32)               
            return (matched_gt_inds, cls_iou_targets, bbox_target, bbox_weights)

        # inside GT
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
        # inside gt 
        valid_mask = bbox_targets.min(-1)[0] > 0
        print(num_gts)
        print(valid_mask.shape)
        print(bbox_targets.shape)
        # condition1: inside a `center ebbox`
        radius = self.center_sample_radius
        stride = offset.new_zeros(offset.shape)

        # project the points on current lvl back to the `original` sizes
        lvl_begin = 0
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
            lvl_begin = lvl_end

        is_in_boxes_and_center = (abs(offset) < stride).all(dim=-1)
        

        valid_mask = valid_mask.squeeze(dim=1)

        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = cls_preds[valid_mask]
        num_valid = valid_decoded_bbox.size(0)
        pairwise_ious = rbbox_overlaps(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + eps)

        # (N, gt_label, num_classes)
        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64),
                      valid_pred_scores.shape[-1]).float().unsqueeze(0).repeat(
                          num_valid, 1, 1))
        # (N, num_gt, pred_scores)
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)

        cls_cost = (
            F.binary_cross_entropy(
                valid_pred_scores.to(dtype=torch.float32).sqrt_(),
                gt_onehot_label,
                reduction='none',
            ).sum(-1).to(dtype=valid_pred_scores.dtype))

        cost_matrix = (
            cls_cost * self.cls_weight + iou_cost * self.iou_weight +
            (~is_in_boxes_and_center) * INF)

        matched_pred_ious, matched_gt_inds = \
            self.dynamic_k_matching(
                cost_matrix, pairwise_ious, num_gt, valid_mask)

        # assign_result = self.assigner.assign(
        #     cls_preds.sigmoid(), decoded_bboxes, gt_bboxes, gt_labels)

        # sampling_result = self.sampler.sample(assign_result, decoded_bboxes, gt_bboxes)
        # pos_inds = sampling_result.pos_inds
        num_pos_per_img = matched_gt_inds.size(0)

        # pos_ious = assign_result.max_overlaps[pos_inds]
        max_overlaps = decoded_bboxes.new_full((num_bboxes, ),
                                                 -INF,
                                                 dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious

        assigned_labels = decoded_bboxes.new_full((num_bboxes, ), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()

        bbox_weights = max_overlaps.clamp(min=1e-6).clone().detach()
        
        cls_iou_targets = torch.zeros_like(cls_preds)
        cls_iou_targets[matched_gt_inds, gt_labels[matched_gt_inds].long()] = pos_ious 

        bbox_target = torch.cat([bbox_targets[matched_gt_inds], gt_angle[matched_gt_inds]],dim = -1)

        return matched_gt_inds, cls_iou_targets, bbox_target, bbox_weights, decoded_bboxes

    def get_targets(self, flatten_cls_scores, flatten_bbox_preds, gt_bboxes, gt_labels, points):
        concat_points = torch.cat(points, dim=0)
        num_points = [center.size(0) for center in points]
        pos_inds, cls_iou_targets, bbox_targets, bbox_weights, decoded_bboxes = multi_apply(
             self._get_target_single, flatten_cls_scores.detach(),
             flatten_bbox_preds.detach(), gt_bboxes, gt_labels, points=concat_points, num_points_per_lvl=num_points)
            
        return pos_inds, cls_iou_targets, bbox_targets, bbox_weights, decoded_bboxes

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   angle_preds,
                   centernesses,
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
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list,
                                                 angle_pred_list,
                                                 centerness_pred_list,
                                                 mlvl_points, img_shape,
                                                 scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           angle_preds,
                           centernesses,
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
        mlvl_centerness = []
        for cls_score, bbox_pred, angle_pred, centerness, points in zip(
                cls_scores, bbox_preds, angle_preds, centernesses,
                mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, 1)
            bbox_pred = torch.cat([bbox_pred, angle_pred], dim=1)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = self.bbox_coder.decode(
                points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            scale_factor = mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[..., :4] = mlvl_bboxes[..., :4] / scale_factor
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_nms_rotated(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
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