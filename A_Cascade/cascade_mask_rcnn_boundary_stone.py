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
                boundary_weight=3.0, # 尝试 3 倍惩罚，强力撕开粘连
                kernel_size=5
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
