import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.models.roi_heads.mask_heads import FCNMaskHead
from mmdet.models.necks.fpn import FPN

# ==========================================
# 抢救方案一：高分辨率边缘感知 FPN (EA-FPN)
# ==========================================
@MODELS.register_module()
class EdgeEnhancedFPN(FPN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('scharr_x', torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32).view(1,1,3,3))
        self.register_buffer('scharr_y', torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32).view(1,1,3,3))
        self.edge_fusion = nn.Conv2d(self.out_channels + 1, self.out_channels, kernel_size=1)

    def forward(self, inputs):
        outs = list(super().forward(inputs))
        p2 = outs[0]
        p2_mean = p2.mean(dim=1, keepdim=True)
        p2_pad = F.pad(p2_mean, (1, 1, 1, 1), mode='replicate')
        edge_x = F.conv2d(p2_pad, self.scharr_x)
        edge_y = F.conv2d(p2_pad, self.scharr_y)
        edge_mag = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        edge_mag = edge_mag / (edge_mag.max() + 1e-6)
        p2_enhanced = torch.cat([p2, edge_mag], dim=1)
        outs[0] = self.edge_fusion(p2_enhanced)
        return tuple(outs)

# ==========================================
# 抢救方案二：带边缘软化的多任务 Mask Head
# ==========================================
@MODELS.register_module()
class SmoothEdgeMaskHead(FCNMaskHead):
    def __init__(self, edge_loss_weight=0.2, **kwargs):
        super().__init__(**kwargs)
        self.edge_loss_weight = edge_loss_weight
        self.edge_conv = nn.Conv2d(self.conv_out_channels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        for conv in self.convs: x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv': x = self.relu(x)
        mask_pred = self.conv_logits(x)
        edge_pred = self.edge_conv(x)
        return (mask_pred, edge_pred) if self.training else mask_pred

    def loss_and_target(self, mask_preds, sampling_results, batch_gt_instances, rcnn_train_cfg):
        mask_pred, edge_pred = mask_preds
        loss_dict = super().loss_and_target(mask_pred, sampling_results, batch_gt_instances, rcnn_train_cfg)
        
        # 💥 核心修复：去掉了下划线，完美适配 MMDetection 3.3.0
        mask_targets = self.get_targets(sampling_results, batch_gt_instances, rcnn_train_cfg)
        
        if mask_targets.numel() == 0:
            loss_dict['loss_edge'] = edge_pred.sum() * 0
            return loss_dict
        laplacian = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32, device=mask_pred.device).view(1, 1, 3, 3)
        gt_masks = mask_targets.unsqueeze(1).float()
        gt_edges = torch.clamp(torch.abs(F.conv2d(F.pad(gt_masks, (1,1,1,1), mode='replicate'), laplacian)), 0, 1)
        # 边缘软化（膨胀）
        gt_edges = F.max_pool2d(gt_edges, kernel_size=3, stride=1, padding=1)
        num_total = gt_edges.numel() + 1e-6
        beta = (num_total - gt_edges.sum()) / num_total
        bce = F.binary_cross_entropy_with_logits(edge_pred, gt_edges, reduction='none')
        loss_edge = (bce * torch.where(gt_edges > 0.5, beta, 1 - beta)).sum() / num_total
        loss_dict['loss_edge'] = loss_edge * self.edge_loss_weight
        return loss_dict
