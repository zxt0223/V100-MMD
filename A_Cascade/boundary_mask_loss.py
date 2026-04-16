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

        # 🛡️ 强制平滑：3x3 卷积核 + 边缘惩罚，从根源杜绝卷积层撕裂出二维码
        self.boundary_weight = boundary_weight
        self.kernel_size = kernel_size
        self.use_sigmoid = True

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        # 1. 确保输入为 4D 张量 [N, 1, H, W]
        if cls_score.dim() == 3:
            cls_score = cls_score.unsqueeze(1)  # [N, H, W] -> [N, 1, H, W]
        if label.dim() == 3:
            label = label.unsqueeze(1)  # [N, H, W] -> [N, 1, H, W]

        label = label.float()

        # 2. 提取边界（仅在前景区域）
        # 创建二值前景掩码 (label > 0)
        with torch.no_grad():
            # 使用阈值 0.5 将 label 二值化，因为 label 可能是 0/1 或浮点数
            binary_label = (label > 0.5).float()
            # 形态学操作获取边界
            dilated = F.max_pool2d(binary_label, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
            eroded = -F.max_pool2d(-binary_label, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
            boundary = dilated - eroded  # 边界像素值为 1，其他为 0

        # 3. 计算基础 BCE 损失
        loss = F.binary_cross_entropy_with_logits(cls_score, label, reduction='none')

        # 4. 施加边界注意力：边界像素获得更高权重
        # 基础权重为 1，边界像素额外增加 boundary_weight
        pixel_weights = torch.ones_like(label) + boundary * self.boundary_weight
        loss = loss * pixel_weights

        # 5. 应用外部权重（如果提供）
        if weight is not None:
            weight = weight.float()
            # 调整权重形状以匹配 loss
            if weight.dim() == 1:
                weight = weight.view(-1, 1, 1, 1)
            elif weight.dim() == 2:
                weight = weight.view(-1, weight.shape[1], 1, 1)
            elif weight.dim() == 3:
                weight = weight.unsqueeze(1)

            loss = loss * weight

        # 6. 计算最终损失
        # 如果有 avg_factor，使用它；否则使用有效像素数
        if avg_factor is not None:
            # avg_factor 通常由 MMDetection 提供，表示有效样本数
            loss = loss.sum() / avg_factor
        elif weight is not None:
            # 使用权重之和作为归一化因子
            valid_pixels = weight.expand_as(loss).sum()
            loss = loss.sum() / (valid_pixels + 1e-5)
        else:
            loss = loss.mean()

        return loss * self.loss_weight
