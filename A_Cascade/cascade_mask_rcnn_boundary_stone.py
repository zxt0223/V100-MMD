_base_ = ['./cascade_mask_rcnn_stone.py']

# 告诉 MMDetection 加载我们的自定义 Loss 模块
custom_imports = dict(imports=['A_Cascade.boundary_mask_loss'], allow_failed_imports=False)

# 重写模型结构，精准替换 Mask Head 的损失函数
model = dict(
    roi_head=dict(
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            # ====== 核心替换：换掉普通的 CrossEntropyLoss ======
            loss_mask=dict(
                type='BoundaryAwareMaskLoss',
                use_mask=True,
                loss_weight=1.0,
                boundary_weight=0.5, # 降低边界惩罚权重，避免过度关注边界导致内部区域学习不足
                kernel_size=3        # 减小卷积核尺寸，获取更细的边界
            )
            # ===================================================
        )
    )
)

# 独立的工作目录，方便你做消融实验对比
work_dir = './A-Out/cascade_boundary_workdir'

# 再次确保 NMS 和测试数量解除封印
test_cfg = dict(
    rcnn=dict(
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.6), 
        max_per_img=500, 
        mask_thr_binary=0.5)
)
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth'
