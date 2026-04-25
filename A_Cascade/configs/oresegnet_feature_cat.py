_base_ = '../cascade_mask_rcnn_stone.py'
# 方案一魔改点：修改 Mask Head 的输入通道为 257 (256语义 + 1物理边缘)
model = dict(
    roi_head=dict(
        mask_head=dict(
            in_channels=257,
            conv_out_channels=256,
        )
    )
)
