_base_ = '../cascade_mask_rcnn_stone.py'
custom_imports = dict(imports=['A_Cascade.oreseg_custom_heads'], allow_failed_imports=False)
model = dict(
    roi_head=dict(
        mask_head=[
            dict(type='FCNMaskHead', num_convs=4, in_channels=256, conv_out_channels=256, roi_feat_size=14, num_classes=1, loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(type='FCNMaskHead', num_convs=4, in_channels=256, conv_out_channels=256, roi_feat_size=14, num_classes=1, loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(type='SmoothEdgeMaskHead', edge_loss_weight=0.2, num_convs=4, in_channels=256, conv_out_channels=256, roi_feat_size=14, num_classes=1, loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
        ]
    )
)
