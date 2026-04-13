import os
import sys
from mmdet.apis import DetInferencer

# 确保当前目录在路径中
sys.path.insert(0, os.getcwd())

# 配置文件与权重路径
config = 'A_LegNet/legnet_decoupled_stone.py'
checkpoint = 'A-Out/legnet_decoupled/weights/legnet/epoch_50.pth'

# 输入与输出设置
img_dir = '/mnt/old_home/chenjinming/Datas/test1'
out_dir = 'A-Predict/legnet_results'

# 初始化推理器 (使用 1 号显卡)
inferencer = DetInferencer(model=config, weights=checkpoint, device='cuda:1')

print(f"开始执行 LEGNet + 双流解耦版预测...")
inferencer(img_dir, out_dir=out_dir, pred_score_thr=0.3, no_save_pred=False)
print(f"预测完成！结果保存在: {out_dir}/vis/")
