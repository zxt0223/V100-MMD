import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.models.roi_heads.mask_heads import FCNMaskHead

# ==========================================
# 方案一：特征拼接流 (FeatureCatMaskHead) - 保持不变
# ==========================================
@MODELS.register_module()
class FeatureCatMaskHead(FCNMaskHead):
    def __init__(self, **kwargs):
        original_in_channels = kwargs.get('in_channels', 256)
        kwargs['in_channels'] = original_in_channels + 1
        super().__init__(**kwargs)
        self.register_buffer('scharr_x', torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32).view(1,1,3,3))
        self.register_buffer('scharr_y', torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32).view(1,1,3,3))

    def forward(self, x):
        x_mean = x.mean(dim=1, keepdim=True)
        x_pad = F.pad(x_mean, (1,1,1,1), mode='replicate')
        edge_x = F.conv2d(x_pad, self.scharr_x)
        edge_y = F.conv2d(x_pad, self.scharr_y)
        edge_mag = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        edge_mag = edge_mag / (edge_mag.max() + 1e-6)
        x_new = torch.cat([x, edge_mag], dim=1)
        return super().forward(x_new)

# ==========================================
# 方案二：晚期辅助损失流 (OreSegEdgeMaskHead) - 修复版
# ==========================================
@MODELS.register_module()
class OreSegEdgeMaskHead(FCNMaskHead):
    def __init__(self, edge_loss_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.edge_loss_weight = edge_loss_weight
        self.edge_conv = nn.Conv2d(self.conv_out_channels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        edge_pred = self.edge_conv(x)
        
        if self.training:
            return mask_pred, edge_pred
        return mask_pred

    # 适配 MMDet 3.x 的最新 Loss 接口
    def loss_and_target(self, mask_preds, sampling_results, batch_gt_instances, rcnn_train_cfg):
        mask_pred, edge_pred = mask_preds
        # 1. 计算原生的 Mask Loss
        loss_dict = super().loss_and_target(mask_pred, sampling_results, batch_gt_instances, rcnn_train_cfg)
        
        # 2. 从框架中提取真实的 mask_targets
        if hasattr(self, '_get_targets'):
            mask_targets = self._get_targets(sampling_results, batch_gt_instances, rcnn_train_cfg)
        else:
            mask_targets = self.get_targets(sampling_results, batch_gt_instances, rcnn_train_cfg)
            
        if mask_targets.numel() == 0:
            loss_dict['loss_edge'] = edge_pred.sum() * 0
            return loss_dict

        # 3. 利用拉普拉斯算子计算 Ground Truth 物理边缘
        laplacian = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32, device=mask_pred.device).view(1, 1, 3, 3)
        gt_masks = mask_targets.unsqueeze(1).float()
        gt_masks_padded = F.pad(gt_masks, (1, 1, 1, 1), mode='replicate')
        gt_edges = F.conv2d(gt_masks_padded, laplacian)
        gt_edges = torch.clamp(torch.abs(gt_edges), 0, 1)

        # 4. 加权交叉熵边缘约束 (Weighted BCE Loss)
        num_total_pixels = gt_edges.numel() + 1e-6
        num_edge_pixels = gt_edges.sum()
        beta = (num_total_pixels - num_edge_pixels) / num_total_pixels
        
        bce = F.binary_cross_entropy_with_logits(edge_pred, gt_edges, reduction='none')
        weight = torch.where(gt_edges > 0.5, beta, 1 - beta)
        
        loss_edge = (bce * weight).sum() / num_total_pixels
        loss_dict['loss_edge'] = loss_edge * self.edge_loss_weight
        
        return loss_dict
