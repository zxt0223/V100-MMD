import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS

@MODELS.register_module()
class BoundaryAwareMaskLoss(nn.Module):
    """
    SCI 核心组件：基于形态学梯度的边缘感知掩码损失 (Boundary-Aware Mask Loss)
    (最终修复版：解决所有四维张量广播与对齐问题)
    """
    def __init__(self, use_mask=True, loss_weight=1.0, boundary_weight=3.0, kernel_size=5):
        super().__init__()
        self.use_mask = use_mask
        self.loss_weight = loss_weight
        self.boundary_weight = boundary_weight 
        self.kernel_size = kernel_size

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        # 1. 极其重要：把真实标签强行扩充为 (N, 1, 28, 28)，与 cls_score 完美对齐
        target_float = label.float().unsqueeze(1) 
        
        # 2. 形态学提取边界 (全过程保持 4 维运算)
        dilated = F.max_pool2d(target_float, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        eroded = -F.max_pool2d(-target_float, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        boundary = dilated - eroded # 形状: (N, 1, 28, 28)
        
        # 3. 构建空间注意力权重矩阵
        pixel_weights = torch.ones_like(target_float) + boundary * self.boundary_weight
        
        # 4. 权重维度对齐
        if weight is not None:
            # 强行将一维权重 (N,) 扩充为 (N, 1, 1, 1)
            if weight.dim() == 1:
                weight = weight.view(-1, 1, 1, 1)
            elif weight.dim() == 2:
                weight = weight.view(-1, weight.shape[1], 1, 1)
            pixel_weights = pixel_weights * weight
            
        # 5. 计算底层的二值交叉熵 (注意这里传入的是 target_float 而不是 label)
        loss = F.binary_cross_entropy_with_logits(cls_score, target_float, reduction='none')
        
        # 6. 施加边缘惩罚力场
        loss = loss * pixel_weights
        
        if avg_factor is not None:
            loss = loss.sum() / (avg_factor + 1e-5)
        else:
            loss = loss.mean()
            
        return loss * self.loss_weight
