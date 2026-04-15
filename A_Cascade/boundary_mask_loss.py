import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS

@MODELS.register_module()
class BoundaryAwareMaskLoss(nn.Module):
    """
    SCI 核心组件：基于形态学梯度的边缘感知掩码损失 (Boundary-Aware Mask Loss)
    (绝对防御版：完全封杀 PyTorch 的隐式广播维度变异)
    """
    def __init__(self, use_mask=True, loss_weight=1.0, boundary_weight=3.0, kernel_size=5):
        super().__init__()
        self.use_mask = use_mask
        self.loss_weight = loss_weight
        self.boundary_weight = boundary_weight 
        self.kernel_size = kernel_size
        self.use_sigmoid = True

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        # ��️ 绝对防御：强制对齐所有张量到绝对标准的 4D (N, 1, 28, 28)
        _cls = cls_score.unsqueeze(1) if cls_score.dim() == 3 else cls_score
        _lbl = label.float().unsqueeze(1) if label.dim() == 3 else label.float()
        
        # 1. 形态学提取边界 (在 4D 下安全运行)
        dilated = F.max_pool2d(_lbl, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        eroded = -F.max_pool2d(-_lbl, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        boundary = dilated - eroded 
        
        # 2. 构建空间注意力权重矩阵
        pixel_weights = torch.ones_like(_lbl) + boundary * self.boundary_weight
        
        # 3. 计算底层的二值交叉熵 (4D vs 4D，绝不会发生 NxN 广播！)
        loss = F.binary_cross_entropy_with_logits(_cls, _lbl, reduction='none')
        
        # 4. 施加边缘惩罚力场
        loss = loss * pixel_weights
        
        # 5. 外部权重维度对齐
        if weight is not None:
            if weight.dim() == 1:
                weight = weight.view(-1, 1, 1, 1)
            elif weight.dim() == 2:
                weight = weight.view(-1, weight.shape[1], 1, 1)
            elif weight.dim() == 3: 
                weight = weight.unsqueeze(1) 
            loss = loss * weight
            
        if avg_factor is not None:
            loss = loss.sum() / (avg_factor + 1e-5)
        else:
            loss = loss.mean()
            
        return loss * self.loss_weight
