_base_ = ['../configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py']

work_dir = './A-Out/cascade_workdir'

# ================= 核心修正区 =================
# 将 roi_head 和 test_cfg 全部放进这唯一的一个 model 字典中！
model = dict(
    # 1. 修改级联的三个 BBox Head 和 Mask Head 的类别数为 1
    roi_head=dict(
        bbox_head=[
            dict(type='Shared2FCBBoxHead', in_channels=256, fc_out_channels=1024, roi_feat_size=7, num_classes=1, bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0., 0., 0., 0.], target_stds=[0.1, 0.1, 0.2, 0.2]), reg_class_agnostic=True, loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0), loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(type='Shared2FCBBoxHead', in_channels=256, fc_out_channels=1024, roi_feat_size=7, num_classes=1, bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0., 0., 0., 0.], target_stds=[0.05, 0.05, 0.1, 0.1]), reg_class_agnostic=True, loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0), loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(type='Shared2FCBBoxHead', in_channels=256, fc_out_channels=1024, roi_feat_size=7, num_classes=1, bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0., 0., 0., 0.], target_stds=[0.033, 0.033, 0.067, 0.067]), reg_class_agnostic=True, loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0), loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_head=dict(type='FCNMaskHead', num_convs=4, in_channels=256, conv_out_channels=256, num_classes=1, loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
    ),
    
    # 2. 测试配置：放宽分数阈值，解除 100 个数量的硬上限
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.01,  # 💥 极大放宽录取分数，让边缘模糊的石头也显形
            nms=dict(type='nms', iou_threshold=0.6), # 放宽 NMS，防密集粘连误杀
            max_per_img=300, # 💥 解除 100 个的封印，允许单图输出 300 块石头
            mask_thr_binary=0.5)
    )
)
# ==============================================

# 显卡安全线：Cascade 有三个头，非常吃显存，Batch Size 必须为 2
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# 120 轮训练策略
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=120, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=120, by_epoch=True, milestones=[80, 110], gamma=0.1)
]

dataset_type = 'CocoDataset'
data_root = '/mnt/old_home/chenjinming/Datas/'
metainfo = {'classes': ('stone',), 'palette': [(0, 255, 0)]} # 绿色 Mask

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=2, num_workers=4,
    dataset=dict(_delete_=True, type=dataset_type, data_root=data_root, metainfo=metainfo, ann_file='annotations/instances_train2017.json', data_prefix=dict(img='train2017/'), filter_cfg=dict(filter_empty_gt=True, min_size=32), pipeline=train_pipeline)
)
val_dataloader = dict(batch_size=1, num_workers=2, dataset=dict(_delete_=True, type=dataset_type, data_root=data_root, metainfo=metainfo, ann_file='annotations/instances_val2017.json', data_prefix=dict(img='val2017/'), test_mode=True, pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(_delete_=True, type='CocoMetric', ann_file=data_root + 'annotations/instances_val2017.json', metric=['bbox', 'segm'])
test_evaluator = val_evaluator

# 定向保存权重
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, out_dir='/mnt/old_home/chenjinming/MMD1/mmdetection/A-Out/weights/cascade')
)

# ================= 核心：挂载 TensorBoard 监控 =================
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend', save_dir='A-Out/train_logs/cascade_vis'),
        dict(type='TensorboardVisBackend', save_dir='A-Out/train_logs/cascade_tb') # 生成 TensorBoard 文件
    ],
    name='visualizer'
)