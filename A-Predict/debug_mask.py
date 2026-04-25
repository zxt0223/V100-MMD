import sys
import os
sys.path.insert(0, os.getcwd())
import torch
from mmdet.apis import init_detector, inference_detector

# 请确保这里的配置文件路径正确
config_file = 'A_Cascade/cascade_mask_rcnn_boundary_stone.py'
# ⚠️ 注意：请将下面的路径替换为你实际的 120 轮权重路径！
checkpoint_file = '/mnt/old_home/chenjinming/MMD1/mmdetection/A-Out/weights/cascade/cascade_boundary_workdir/epoch_120.pth' 

model = init_detector(config_file, checkpoint_file, device='cuda:0')
result = inference_detector(model, '/mnt/old_home/chenjinming/Datas/test1/IMG_20251223142653.jpg')

masks = result.pred_instances.masks
print("\n" + "🚀 " + "="*50)
print(f"📦 完美检测到石头的数量 (Boxes): {len(result.pred_instances.bboxes)}")
print(f"🧩 模型输出的 Mask 矩阵形状: {masks.shape}")
print(f"🟩 整个 Mask 矩阵中判定为石头的像素总数: {masks.sum().item()}")
print("="*52 + "\n")
