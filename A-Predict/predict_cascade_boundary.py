import sys
import os
from mmdet.apis import DetInferencer

# 将当前目录加入环境变量，防止找不到自定义的模块
sys.path.insert(0, os.getcwd())

# 配置文件 (如果你起的名字不是这个，请自行修改一下这行)
config = 'A_Cascade/cascade_mask_rcnn_boundary_stone.py' 
# 刚出炉的 120 轮 Boundary 权重
checkpoint = '/mnt/old_home/chenjinming/MMD1/mmdetection/A-Out/weights/cascade/cascade_boundary_workdir/epoch_120.pth'

# 输入输出路径
img_dir = '/mnt/old_home/chenjinming/Datas/test1'
out_dir = 'A-Predict/cascade_boundary_results'

print("🌟 正在加载 Cascade Mask R-CNN (Boundary IoU) 120E 终极模型...")
# 注意：外部隔离了显卡，内部这里依然填 cuda:0
inferencer = DetInferencer(model=config, weights=checkpoint, device='cuda:0')

# 执行预测 (之前测试过 0.05 左右效果最好，如果漏检可以降到 0.01)
inferencer(img_dir, out_dir=out_dir, pred_score_thr=0.05, no_save_pred=False)

print(f"✅ Boundary 预测完成！绝美锋利图片已保存至: {out_dir}/vis/")
