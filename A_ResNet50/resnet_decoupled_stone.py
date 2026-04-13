_base_ = ['../configs/double_heads/dh-faster-rcnn_r50_fpn_1x_coco.py']

work_dir = './A-Out/resnet_decoupled'

model = dict(
    type='MaskRCNN',
    roi_head=dict(
        type='DoubleHeadRoIHead',
        bbox_head=dict(num_classes=1),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
    ),
    train_cfg=dict(rcnn=dict(mask_size=28)),
    test_cfg=dict(rcnn=dict(mask_thr_binary=0.5))
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# ======== 核心修改：120 轮策略 ========
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=120, val_interval=10)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    # 学习率在第 80 轮和 110 轮时进行衰减
    dict(type='MultiStepLR', begin=0, end=120, by_epoch=True, milestones=[80, 110], gamma=0.1)
]

dataset_type = 'CocoDataset'
data_root = '/mnt/old_home/chenjinming/Datas/'
metainfo = {'classes': ('stone',), 'palette': [(220, 20, 60)]}

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
    batch_size=2, # 保持绝对安全的显存占用
    num_workers=4,
    dataset=dict(
        _delete_=True, type=dataset_type, data_root=data_root, metainfo=metainfo,
        ann_file='annotations/instances_train2017.json', data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32), pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=1, num_workers=2,
    dataset=dict(
        _delete_=True, type=dataset_type, data_root=data_root, metainfo=metainfo,
        ann_file='annotations/instances_val2017.json', data_prefix=dict(img='val2017/'),
        test_mode=True, pipeline=test_pipeline)
)
test_dataloader = val_dataloader
val_evaluator = dict(_delete_=True, type='CocoMetric', ann_file=data_root + 'annotations/instances_val2017.json', metric=['bbox', 'segm'])
test_evaluator = val_evaluator

# ======== 核心修改：强制写死 pth 保存的绝对路径 ========
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3, out_dir='/mnt/old_home/chenjinming/MMD1/mmdetection/A-Out/weights/resnet')
)

visualizer = dict(type='DetLocalVisualizer', vis_backends=[dict(type='LocalVisBackend', save_dir='A-Out/train_logs/resnet')], name='visualizer')
