# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.ops import points_in_polygons
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner

from ..builder import ROTATED_BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from ..transforms import obb2poly


@ROTATED_BBOX_ASSIGNERS.register_module()
class RotatedTaskAlignAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): Number of bbox selected in each level.
    """

    def __init__(self,
                 topk,
                 angle_version='oc',
                 iou_calculator=dict(type='RBboxOverlaps2D')):
        self.topk = topk
        self.angle_version = angle_version
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.sigmoid = nn.Sigmoid()

    def assign(self,
               pred_scores,
               decoded_bbox,
               anchors,
            #    num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               alpha=1.0,
               beta=6.0):
        
        INF = 100000000
        num_gt, num_bboxes = gt_bboxes.shape[0], decoded_bbox.shape[0]

        # compute iou between all bbox and gt
        overlaps = self.iou_calculator(decoded_bbox, gt_bboxes)
        bbox_scores = self.sigmoid(pred_scores[:, gt_labels].detach())
        
        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)
        assign_metrics = overlaps.new_zeros((num_bboxes, ))

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            # 添加assign_metric
            assign_result = AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
            assign_result.assign_metrics = assign_metrics
            return assign_result

        # compute center distance between all bbox and gt
        # the center of gt and bbox
        # 计算所有gt中心和bbox的距离
        # gt_points = gt_bboxes[:, :2]

        # distances = (bboxes_points[:, None, :] -
        #              gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        # Calculate ratios of gt_bboxes and beta 
        # gt_bboxes_ratios = self.AspectRatio(gt_bboxes)
        # print("gt_bboxes_ratios:", gt_bboxes_ratios)
        # beta = torch.exp((1 / 4) * gt_bboxes_ratios)
        # print("beta1:", beta)

        # print("overlaps",overlaps.max())
        # print("bbox-scores", bbox_scores.max())
        # print("num_gt, num_bboxes", num_gt, "   ", num_bboxes)
        # select top-k bboxes as candidates for each gt
        alignment_metrics = bbox_scores**alpha * overlaps**beta
        topk = min(self.topk, alignment_metrics.size(0))
        _, candidate_idxs = alignment_metrics.topk(topk, dim=0, largest=True)
        candidate_metrics = alignment_metrics[candidate_idxs,
                                              torch.arange(num_gt)]
        is_pos = candidate_metrics > 0


        # # Selecting candidates based on the center distance
        # candidate_idxs = []
        # start_idx = 0
        # for level, bboxes_per_level in enumerate(num_level_bboxes):
        #     # on each pyramid level, for each gt,
        #     # select k bbox whose center are closest to the gt center
        #     # 在每个金字塔层，每个gt选择k个bbox中心最接近gt中心的样本
        #     end_idx = start_idx + bboxes_per_level
        #     distances_per_level = distances[start_idx:end_idx, :]
        #     _, topk_idxs_per_level = distances_per_level.topk(
        #         self.topk, dim=0, largest=False)
        #     candidate_idxs.append(topk_idxs_per_level + start_idx)
        #     start_idx = end_idx
        # candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # # get corresponding iou for the these candidates, and compute the
        # # mean and std, set mean + std as the iou threshold
        # # 在候选框中计算mean和std并设置mean + std 为iou threshold
        

        # candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        # overlaps_mean_per_gt = candidate_overlaps.mean(0)
        # overlaps_std_per_gt = candidate_overlaps.std(0)
        # overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        # is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # limit the positive sample's center in gt
        # 限制正样本在gt中心
        bboxes_points = anchors[:, :2]
        gt_bboxes = obb2poly(gt_bboxes, self.angle_version)
        inside_flag = points_in_polygons(bboxes_points, gt_bboxes)
        is_in_gts = inside_flag[candidate_idxs,
                                torch.arange(num_gt)].to(is_pos.dtype)

        is_pos = is_pos & is_in_gts
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        candidate_idxs = candidate_idxs.view(-1)

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
        assign_metrics[max_overlaps != -INF] = alignment_metrics[
            max_overlaps != -INF, argmax_overlaps[max_overlaps != -INF]]

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        assign_result = AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        assign_result.assign_metrics = assign_metrics
        return assign_result

    def AspectRatio(self, gt_rbboxes):
        # gt_rbboxes = torch.squeeze(gt_rbboxes)
        # print('AspectRatio.gt_rbboxes')
        # print(gt_rbboxes.size())
        # gt_rbboxes = rotated_box_to_poly(gt_rbboxes.to(torch.float32)).contiguous().to(torch.float64)

        # pt1, pt2, pt3, pt4 = gt_rbboxes[..., :8].chunk(4, 1)
        #
        # edge1 = torch.sqrt(
        #     torch.pow(pt1[..., 0] - pt2[..., 0], 2) + torch.pow(pt1[..., 1] - pt2[..., 1], 2))
        # edge2 = torch.sqrt(
        #     torch.pow(pt2[..., 0] - pt3[..., 0], 2) + torch.pow(pt2[..., 1] - pt3[..., 1], 2))
        edge1 = gt_rbboxes[..., 2]
        edge2 = gt_rbboxes[..., 3]
        edges = torch.stack([edge1, edge2], dim=1)

        width, _ = torch.max(edges, 1)
        height, _ = torch.min(edges, 1)

        ratios = (width / height)
        return ratios
