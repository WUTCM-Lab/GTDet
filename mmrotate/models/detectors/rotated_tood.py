from .rotated_retinanet import RotatedRetinaNet
from ..builder import ROTATED_DETECTORS
@ROTATED_DETECTORS.register_module()
class RotatedTood(RotatedRetinaNet):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RotatedRetinaNet,
              self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                             pretrained, init_cfg)
    
    def set_epoch(self, epoch):
        self.bbox_head.epoch = epoch
        # self.odm_head.epoch = epoch