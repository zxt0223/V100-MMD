import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.models.roi_heads import StandardRoIHead

@MODELS.register_module()
class DecoupledRoIHead(StandardRoIHead):
    """
    双流解耦头：
    1. 分类分支：走全连接层 (FC)
    2. 回归分支：走卷积层 (Conv)
    3. Mask分支：独立分支
    """
    def __init__(self, **kwargs):
        super(DecoupledRoIHead, self).__init__(**kwargs)

    def forward_bbox_train(self, x, img_metas, proposal_list, gt_bboxes, gt_labels, rois=None):
        if rois is None:
            rois = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], proposal_list)
        
        # 核心解耦逻辑：在 bbox_head 内部处理分类和回归的特征分离
        bbox_results = self.bbox_head(rois)
        
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes, gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'], bbox_results['bbox_pred'], rois, *bbox_targets)
        
        return dict(loss_bbox=loss_bbox, bbox_results=bbox_results)
