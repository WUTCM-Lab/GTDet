import torch
import torch.nn.functional as F
from mmdet.core import AssignResult
from mmdet.core.bbox.assigners import SimOTAAssigner

from mmrotate.core.bbox.builder import ROTATED_BBOX_ASSIGNERS
from mmrotate.core.bbox.iou_calculators import rbbox_overlaps

def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma

def kld_distance(valid_decoded_bbox, gt_bboxes):

    mu_p, sigma_p = xy_wh_r_2_xy_sigma(valid_decoded_bbox)
    mu_t, sigma_t = xy_wh_r_2_xy_sigma(gt_bboxes)

    N = mu_p.size()[0]                          # mu_p = (N, 2)
    K = mu_t.size()[0]                          # mu_t = (K, 2)
    
    mu_p = mu_p.unsqueeze(1)                    # mu_p = (N, 1, 2)
    sigma_p = sigma_p.unsqueeze(1)              # sigma_p = (N, 1, 2, 2)
    
    mu_t = mu_t.reshape(1, K, 2)                # mu_t = (1, K, 2)
    sigma_t = sigma_t.reshape(1, K, 2, 2)       # mu_t = (1, K, 2, 2)
 
    delta = (mu_p - mu_t).unsqueeze(-1)         # delta = (N, K, 2)
    sigma_t_inv = torch.inverse(sigma_t)
    term1 = delta.transpose(-1,
                            -2).matmul(sigma_t_inv).matmul(delta).reshape(N, K)
    term2 = torch.diagonal(
        sigma_t_inv.matmul(sigma_p),
        dim1=-2, dim2=-1).sum(dim=-1) + \
        torch.log(torch.det(sigma_t) / torch.det(sigma_p))
    
    dis = 0.5 * (term1 + term2) - 1
    dis_agg = dis.clamp(min=1e-6)
    overlaps = 1 / (2 + dis_agg)
    
    return overlaps

@ROTATED_BBOX_ASSIGNERS.register_module()
class RSimOTAAssigner_softlable_distance_kld(SimOTAAssigner):

    def __init__(self,
                 angle_version='le90',
                 candidate_topk=10,
                 **kwargs):
        super(RSimOTAAssigner_softlable_distance_kld, self).__init__(**kwargs)
        self.angle_version = angle_version
        self.candidate_topk = candidate_topk
    def _assign(self,
                pred_scores,
                priors,
                decoded_bboxes,
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore=None,
                eps=1e-7):
        """Assign gt to priors using SimOTA.
        Args:
            pred_scores (Tensor): Classification scores of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Predicted bboxes, a 2D-Tensor with shape
                [num_priors, 5] in [x, y, w, h ,a] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 5] in [x, y, w, h ,a] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            eps (float): A value added to the denominator for numerical
                stability. Default 1e-7.
        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        INF = 100000.0
        num_gt = gt_bboxes.size(0)
        num_bboxes = decoded_bboxes.size(0)

        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes, ),
                                                   0,
                                                   dtype=torch.long)
        valid_mask, soft_center_prior = self.get_in_gt_and_in_center_info(
            priors, gt_bboxes)
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)

        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
                                                          -1,
                                                          dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        pairwise_ious = kld_distance(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + eps)

        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64),
                      pred_scores.shape[-1]).float().unsqueeze(0).repeat(
                          num_valid, 1, 1))

        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)

        soft_label = gt_onehot_label * pairwise_ious[..., None]
        scale_factor = soft_label - valid_pred_scores

        # cls_cost = (
        #     F.binary_cross_entropy(
        #         valid_pred_scores.to(dtype=torch.float32).sqrt_(),
        #         gt_onehot_label,
        #         reduction='none',
        #     ).sum(-1).to(dtype=valid_pred_scores.dtype))

        soft_cls_cost = F.binary_cross_entropy_with_logits(
            valid_pred_scores, soft_label,
            reduction='none') * scale_factor.abs().pow(2.0)
        soft_cls_cost = soft_cls_cost.sum(dim=-1)

        # cost_matrix = (
        #     cls_cost * self.cls_weight + iou_cost * self.iou_weight +
        #     (~is_in_boxes_and_center) * INF)

        cost_matrix = soft_cls_cost * self.cls_weight + iou_cost * self.iou_weight + soft_center_prior

        matched_pred_ious, matched_gt_inds = \
            self.dynamic_k_matching(
                cost_matrix, pairwise_ious, num_gt, valid_mask)

        # convert to AssignResult format
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes, ),
                                                 -INF,
                                                 dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def get_in_gt_and_in_center_info(self, priors, gt_bboxes):
        num_gt = gt_bboxes.size(0)

        is_in_gts = self.points_in_rbboxes(priors[..., :2], gt_bboxes) > 0
        is_in_gts_all = is_in_gts.sum(dim=1) > 0

        # 每个特征点对应的x, y的位置 
        repeated_x = priors[:, 0].unsqueeze(1).repeat(1, num_gt)
        repeated_y = priors[:, 1].unsqueeze(1).repeat(1, num_gt)
        # 特征点之间的间隔 
        repeated_stride_x = priors[:, 2].unsqueeze(1).repeat(1, num_gt)
        # repeated_stride_y = priors[:, 3].unsqueeze(1).repeat(1, num_gt)
        # gt_cxs 和 gt_cys 为GT的中心点的位置
        gt_cxs = gt_bboxes[:, 0]
        gt_cys = gt_bboxes[:, 1]

        # 计算中心点到各个特征点的距离 
        distance = torch.sqrt((repeated_x - gt_cxs) ** 2 + (repeated_y - gt_cys) ** 2) / repeated_stride_x
        
        # 将小于center_radius * repeated_stride_x 的为is_in_cts
        is_in_cts = distance < self.center_radius 
        is_in_cts_all = is_in_cts.sum(dim=1) > 0

        # in boxes or in centers, shape: [num_priors]
        is_in_gts_or_centers = is_in_gts_all | is_in_cts_all

        # both in boxes and centers, shape: [num_fg, num_gt]
        soft_center_prior = torch.pow(10, distance[is_in_gts_or_centers] - self.center_radius)


        return is_in_gts_or_centers, soft_center_prior

    def points_in_rbboxes(self, points, rbboxes):
        """Judging whether points are inside rotated bboxes.
        Args:
            points (torch.Tensor): It has shape (B, 2), indicating (x, y).
                M means the number of predicted points.
            rbboxes (torch.Tensor): It has shape (M, 5), indicating
                (x, y, w, h, a). M means the number of rotated bboxes.
        Returns:
            torch.Tensor: Return the result with the shape of (B, M).
        """
        num_gt = rbboxes.size(0)
        num_prior = points.size(0)
        points = points[:, None, :].expand(num_prior, num_gt, 2)
        rbboxes = rbboxes[None].expand(num_prior, num_gt, 5)
        # is prior centers in gt
        ctr, wh, angle = torch.split(rbboxes, [2, 2, 1], dim=2)

        cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=-1).reshape(num_prior, num_gt, 2, 2)
        offset = points - ctr
        offset = torch.matmul(rot_matrix, offset[..., None])
        offset = offset.squeeze(-1)

        w, h = wh[..., 0], wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        deltas = torch.stack((left, top, right, bottom), dim=1)
        return deltas.min(dim=1).values




if __name__ == '__main__':
    # get_in_gt_and_in_center_info(self, priors, gt_bboxes)
    priors =  torch.tensor([[1.0, 1.0, 1.0, 1.0], [2.0,2.0,1.0,1.0], [10.0,10.0,1.0,1.0]])# [bn, num_prior,  4] (x, y, w, h) 3,4
    gt_bboxes = torch.tensor([[2.0,2.0,1.0,1.0, 0.0]])                                 # [bn, num_gt, 4]
    ota = RSimOTAAssigner_softlable_distance()
    is_in_gts_or_centers, soft_center_prior = ota.get_in_gt_and_in_center_info(priors, gt_bboxes)
    print(is_in_gts_or_centers) # 是否在 gts内或者 centers
    print(is_in_gts_or_centers.shape)
    print(soft_center_prior) # 在 gts内 并且在 centers内
    print(soft_center_prior.shape)