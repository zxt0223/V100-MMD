import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS

@MODELS.register_module()
class BoundaryAwareMaskLoss(nn.Module):
    def __init__(self, use_mask=True, loss_weight=1.0, boundary_weight=1.0, kernel_size=3):
        super().__init__()
        self.use_mask = use_mask
        self.loss_weight = loss_weight
        
        # 🛡️ 强制平滑：3x3 卷积核 + 1.0 边缘惩罚，从根源杜绝卷积层撕裂出二维码
        self.boundary_weight = boundary_weight
        self.kernel_size = kernel_size
        self.use_sigmoid = True

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        # 1. 绝对 4D 结界，屏蔽一切 MMDetection 的降维干扰
        _cls = cls_score.unsqueeze(1) if cls_score.dim() == 3 else cls_score
        _lbl = label.float().unsqueeze(1) if label.dim() == 3 else label.float()
        
        # 2. 提取极细且平滑的形态学边界
        dilated = F.max_pool2d(_lbl, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        eroded = -F.max_pool2d(-_lbl, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        boundary = dilated - eroded 
        
        # 3. 计算原生纯净的 BCE Loss [N, 1, 28, 28]
        loss = F.binary_cross_entropy_with_logits(_cls, _lbl, reduction='none')
        
        # 4. 施加边界注意力
        pixel_weights = torch.ones_like(_lbl) + boundary * self.boundary_weight
        loss = loss * pixel_weights
        
        # 5. 🌟 终极免疫系统：安全处理所有奇葩的外部 Weight 和均值
        if weight is not None:
            weight = weight.float()
            # 彻底解决 198 vs 28 的形状冲突
            if weight.dim() == 1:
                weight = weight.view(-1, 1, 1, 1)
            elif weight.dim() == 2:
                weight = weight.view(-1, weight.shape[1], 1, 1)
            elif weight.dim() == 3:
                weight = weight.unsqueeze(1)
                
            loss = loss * weight
            # 彻底解决 784 倍的二维码爆炸！精准计算有效像素总数
            valid_pixels = weight.expand_as(loss).sum()
            loss = loss.sum() / (valid_pixels + 1e-5)
        else:
            loss = loss.mean()
            
        return loss * self.loss_weight
