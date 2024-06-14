# Copyright (c) OpenMMLab. All rights reserved.
from .atss_kld_assigner import ATSSKldAssigner
from .atss_obb_assigner import ATSSObbAssigner
from .convex_assigner import ConvexAssigner
from .max_convex_iou_assigner import MaxConvexIoUAssigner
from .sas_assigner import SASAssigner
from .rotated_task_align_assigner import RotatedTaskAlignAssigner
from .r_sim_ota_assigner import RSimOTAAssigner
from .r_sim_ota_assigner_distance import RSimOTAAssigner_distance
from .r_sim_ota_assigner_softlabel_distance import RSimOTAAssigner_softlable_distance
from .r_sim_ota_assigner_gau import RSimOTAAssigner_gau
from .r_sim_ota_assigner_gau_1205 import RSimOTAAssigner_gau_1205
from .r_sim_ota_assigner_softlabel_distance_kld import RSimOTAAssigner_softlable_distance_kld
from .r_sim_ota_assigner_kld import RSimOTAAssigner_kld
from .r_sim_ota_assigner_gau_sample import RSimOTAAssigner_gau_sample
from .r_sim_ota_assigner_softlabel import RSimOTAAssigner_softlabel
from .r_sim_ota_assigner_focal import RSimOTAAssigner_focal
from .r_sim_ota_assigner_gau_distance import RSimOTAAssigner_gau_distance

__all__ = [
    'ConvexAssigner', 'MaxConvexIoUAssigner', 'SASAssigner', 'ATSSKldAssigner',
    'ATSSObbAssigner', 'RotatedTaskAlignAssigner', 'RSimOTAAssigner',
    'RSimOTAAssigner_distance', 'RSimOTAAssigner_softlable_distance',
    'RSimOTAAssigner_gau', 'RSimOTAAssigner_gau_1205',
    'RSimOTAAssigner_softlable_distance_kld', 'RSimOTAAssigner_kld',
    'RSimOTAAssigner_gau_sample', 'RSimOTAAssigner_softlabel',
    'RSimOTAAssigner_focal', 'RSimOTAAssigner_gau_distance'
]
