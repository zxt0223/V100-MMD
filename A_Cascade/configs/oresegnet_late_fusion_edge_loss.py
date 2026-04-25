_base_ = '../cascade_mask_rcnn_stone.py'
# 方案二魔改点：在 Head 中开启边缘检测分支与辅助损失
model = dict(
    roi_head=dict(
        mask_head=dict(
            with_edge_branch=True, # 需确保你的代码已实现此参数
            loss_edge=dict(
                type='WeightedCrossEntropyLoss',
                loss_weight=1.0)
        )
    )
)
