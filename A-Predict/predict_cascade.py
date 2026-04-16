import sys
import os
from mmdet.apis import DetInferencer

# 将当前目录加入环境变量，防止找不到自定义的模块
sys.path.insert(0, os.getcwd())

# 配置文件和刚出炉的 120 轮权重
config = 'A_Cascade/cascade_mask_rcnn_stone.py'
checkpoint = '/mnt/old_home/chenjinming/MMD1/mmdetection/A-Out/weights/cascade/cascade_workdir/epoch_2.pth'

# 输入输出路径
img_dir = '/mnt/old_home/chenjinming/Datas/test1'
out_dir = 'A-Predict/cascade_results'

print("🌟 正在加载 Cascade Mask R-CNN (120E) 终极模型...")
# 初始化推理器
inferencer = DetInferencer(model=config, weights=checkpoint, device='cuda:0')

# 执行预测 (阈值设为 0.3，你可以根据实际效果调高到 0.5)
inferencer(img_dir, out_dir=out_dir, pred_score_thr=0.3, no_save_pred=False)

print(f"✅ Cascade 预测完成！绝美图片已保存至: {out_dir}/vis/")
