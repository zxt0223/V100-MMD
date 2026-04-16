import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


@MODELS.register_module()
class BoundaryAwareMaskLoss(nn.Module):
    def __init__(
        self, use_mask=True, loss_weight=1.0, boundary_weight=1.0, kernel_size=3
    ):
        super().__init__()
        self.use_mask = use_mask
        self.loss_weight = loss_weight

        # 🛡️ 强制平滑：3x3 卷积核 + 边缘惩罚，从根源杜绝卷积层撕裂出二维码
        self.boundary_weight = boundary_weight
        self.kernel_size = kernel_size
        self.use_sigmoid = True

    def forward(
        self,
        cls_score,
        label,
        class_label=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):
        if reduction_override not in (None, "none", "mean", "sum"):
            raise ValueError(f"Unsupported reduction_override: {reduction_override}")

        # MMDetection mask heads pass the ROI class label as the 3rd positional arg.
        # For multi-class masks we must pick the matching channel before BCE.
        if cls_score.dim() == 3:
            cls_score = cls_score.unsqueeze(1)

        if cls_score.dim() != 4:
            raise ValueError(
                f"cls_score must be 3D or 4D, got shape {tuple(cls_score.shape)}"
            )

        if label.dim() == 2:
            label = label.unsqueeze(0)
        if label.dim() == 3:
            label = label.unsqueeze(1)

        if label.dim() != 4:
            raise ValueError(f"label must be 3D or 4D, got shape {tuple(label.shape)}")

        if cls_score.size(1) > 1:
            if class_label is None:
                raise ValueError(
                    "class_label is required when cls_score has multiple channels"
                )
            class_label = class_label.long().view(-1, 1, 1, 1)
            gather_index = class_label.expand(
                -1, 1, cls_score.size(2), cls_score.size(3)
            )
            cls_score = torch.gather(cls_score, dim=1, index=gather_index)
        else:
            cls_score = cls_score[:, :1]

        label = label.float()

        with torch.no_grad():
            binary_label = (label > 0.5).float()
            dilated = F.max_pool2d(
                binary_label,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            )
            eroded = -F.max_pool2d(
                -binary_label,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            )
            boundary = dilated - eroded

        loss = F.binary_cross_entropy_with_logits(cls_score, label, reduction="none")
        pixel_weights = torch.ones_like(label) + boundary * self.boundary_weight
        loss = loss * pixel_weights

        reduction = reduction_override or "mean"
        if reduction == "none":
            return loss * self.loss_weight
        if reduction == "sum":
            return loss.sum() * self.loss_weight

        if avg_factor is not None:
            if torch.is_tensor(avg_factor):
                avg_factor = avg_factor.clamp(min=1e-6)
            else:
                avg_factor = max(float(avg_factor), 1e-6)
            loss = loss.sum() / avg_factor
        else:
            loss = loss.mean()

        return loss * self.loss_weight
