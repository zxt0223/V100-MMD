# 继承你之前写好的“双流解耦头”配置
_base_ = [
    './legnet_decoupled_stone.py'
]

# 注册所有必需的自定义模块 (保证 LWEGNet 能被找到)
custom_imports = dict(
    imports=[
        'A_LegNet.custom_legnet',        
        'A_LegNet.decoupled_head',       
        'A_Cascade.boundary_mask_loss'   
    ], 
    allow_failed_imports=False
)

# 💥 核心修复：删掉了导致报错的 use_sigmoid=True
model = dict(
    roi_head=dict(
        mask_head=dict(
            loss_mask=dict(
                type='BoundaryAwareMaskLoss',
                loss_weight=1.0,
                boundary_weight=0.2,     # 黄金甜点权重
                kernel_size=3)
        )
    )
)

# 释放边缘预测阈值
test_cfg = dict(
    rcnn=dict(
        score_thr=0.05,
        mask_thr_binary=0.1,
        max_per_img=500
    )
)

work_dir = './A-Out/legnet_decoupled_boundary_workdir'
