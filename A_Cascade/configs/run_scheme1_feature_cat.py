_base_ = '../cascade_mask_rcnn_stone.py'
custom_imports = dict(imports=['A_Cascade.oreseg_custom_heads'], allow_failed_imports=False)
model = dict(
    neck=dict(
        type='EdgeEnhancedFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5
    )
)
