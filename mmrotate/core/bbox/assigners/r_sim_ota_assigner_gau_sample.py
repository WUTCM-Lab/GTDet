import torch
import torch.nn.functional as F
from mmdet.core import AssignResult
from mmdet.core.bbox.assigners import SimOTAAssigner

from mmrotate.core.bbox.builder import ROTATED_BBOX_ASSIGNERS
from mmrotate.core.bbox.iou_calculators import rbbox_overlaps


@ROTATED_BBOX_ASSIGNERS.register_module()
class RSimOTAAssigner_gau_sample(SimOTAAssigner):

    def __init__(self,
                 angle_version='le90',
                 candidate_topk=10,
                 **kwargs):
        super(RSimOTAAssigner_gau_sample, self).__init__(**kwargs)
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
        valid_mask= self.get_in_gt_and_in_center_info(
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

        pairwise_ious = rbbox_overlaps(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + eps)

        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64),
                      pred_scores.shape[-1]).float().unsqueeze(0).repeat(
                          num_valid, 1, 1))

        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)

        
        soft_label = gt_onehot_label * pairwise_ious[..., None]
        scale_factor = soft_label - valid_pred_scores

        # cls_cost = (
        #    F.binary_cross_entropy(
        #        valid_pred_scores.to(dtype=torch.float32).sqrt_(),
        #        gt_onehot_label,
        #        reduction='none',
        #    ).sum(-1).to(dtype=valid_pred_scores.dtype))

        soft_cls_cost = F.binary_cross_entropy(
            valid_pred_scores, soft_label,
            reduction='none') * scale_factor.abs().pow(2.0)
        soft_cls_cost = soft_cls_cost.sum(dim=-1)

        # gau_cost = -torch.log(gau_dis + eps)
        
        cost_matrix = soft_cls_cost * self.cls_weight + iou_cost * self.iou_weight 

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
        
        # is_in_gts = self.points_in_rbboxes(priors[..., :2], gt_bboxes) > 0
        # is_in_gts_all = is_in_gts.sum(dim=1) > 0

        # center point
        # priors_cts = priors[:, :2] 

        # cal gau_dis
        is_in_center = self.get_gau_dis(gt_bboxes, priors[:, :2] ) > 0.6
        is_in_center_all = is_in_center.sum(dim=1) > 0

        # return is_in_gts_all, gau_dis
        return is_in_center_all

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

    def xy_wh_r_2_xy_sigma(self, xywhr):
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
    
    def get_gau_dis(self, gt_bboxes, priors_cts):
        (gt_xy, gt_sigma) = self.xy_wh_r_2_xy_sigma(gt_bboxes)  # K, 2  K, 2, 2
        
        # N = priors_cts.size(0)                                  # N, 2
        K = gt_xy.size(0)
        
        priors_cts = priors_cts.unsqueeze(1)                    # N, 1, 2
        
        gt_xy = gt_xy.reshape(1, K, 2)                          # 1, K, 2
        gt_sigma = gt_sigma.reshape(1, K, 2, 2)                 # 1, K, 2, 2

        diff = (priors_cts - gt_xy).unsqueeze(-1)               # ((N, 1, 2) - (1, K, 2)).unsqueeze(-1) = (N, K, 2, 1)
        
        diff_t = diff.transpose(-1, -2)                         # N, K, 1, 2
        
        sigma_inv = torch.inverse(gt_sigma)                     # 1, K, 2, 2
        
        value = torch.exp(-0.5 * diff_t.matmul(sigma_inv).matmul(diff).squeeze(-1).squeeze(-1))        # N, K
        return value

        

if __name__ == '__main__':

    import numpy as np 
    import matplotlib.pyplot as plt
    eps = 1e-7
    assigner = RSimOTAAssigner_gau_sample()
    gt = torch.tensor([[10.0, 10.0, 10.0, 10.0, 0.0]]) # N, 5
    point = torch.tensor([  [10.0, 10.0], [11.0, 10.0], [12.0, 10.0], [13.0, 10.0],
                            [14.0, 10.0], [15.0, 10.0], [16.0, 10.0],
                            [17.0, 10.0], [18.0, 10.0], [19.0, 10.0],
                            [20.0, 10.0],
                            # [21.0, 10.0], [22.0, 10.0],
                            # [23.0, 10.0], [24.0, 10.0], [25.0, 10.0],
                            # [26.0, 10.0], [27.0, 10.0], [28.0, 10.0],
                            # [29.0, 10.0], [30.0, 10.0], [31.0, 10.0],
                            # [32.0, 10.0], [33.0, 10.0], [34.0, 10.0],
                            # [35.0, 10.0], [36.0, 10.0], [37.0, 10.0],
                            # [38.0, 10.0], [39.0, 10.0], [40.0, 10.0],
                        ]) # K, 2
    x = np.linspace(10, 20, 11, endpoint=True)

    gau_dis = assigner.get_gau_dis(gt, point).clamp(min=0.01)
    
    gau_log = -torch.log(gau_dis)
    print(gau_dis)
    print(gau_dis.shape)
    gau_power  = torch.exp((1 - gau_dis) * 10)
    print(gau_power)
    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(x, gau_log, linestyle="solid", marker='v', c='red', label='gau_log')
    plt.plot(x, gau_dis, linestyle="--", marker='+', c='blue', label='gau_dis')
    # plt.plot(x, gau_power, linestyle="-", marker='.', c='green', label='gau_power')
    plt.title('gau_dis') 
    plt.legend(loc='best', fontsize=14, markerscale=0.5)
    # plt.show()
    plt.savefig("/home/server4/mmrotate/figure/gau_dis.jpg")