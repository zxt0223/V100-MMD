# 继承针对 ResNet50 的双流解耦配置
_base_ = [
    './resnet_decoupled_stone.py'
]

# 导入必要的自定义模块
custom_imports = dict(
    imports=[
        'A_LegNet.decoupled_head',       # 注册解耦头
        'A_Cascade.boundary_mask_loss'   # 注册边界感知损失
    ], 
    allow_failed_imports=False
)

# 替换 Mask Loss，去掉了引起报错的 use_sigmoid
model = dict(
    roi_head=dict(
        mask_head=dict(
            loss_mask=dict(
                type='BoundaryAwareMaskLoss',
                loss_weight=1.0,
                boundary_weight=0.2,     # 甜点惩罚权重
                kernel_size=3)
        )
    )
)

# 配合边界惩罚，下调推理二值化阈值释放边缘
test_cfg = dict(
    rcnn=dict(
        score_thr=0.05,
        mask_thr_binary=0.1,
        max_per_img=500
    )
)

# 独立的 ResNet 融合版本输出目录
work_dir = './A-Out/resnet_decoupled_boundary_workdir'
