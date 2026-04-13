import os
import sys
from mmdet.apis import DetInferencer

# 确保当前目录在路径中，以便加载 A_ResNet50 下的自定义模块
sys.path.insert(0, os.getcwd())

# 配置文件与权重路径（对应 A-Out 下的 50 轮产物）
config = 'A_ResNet50/resnet_decoupled_stone.py'
checkpoint = 'A-Out/resnet_decoupled/weights/resnet/epoch_50.pth'

# 输入与输出设置
img_dir = '/mnt/old_home/chenjinming/Datas/test1'
out_dir = 'A-Predict/resnet_results'

# 初始化推理器 (使用 0 号显卡)
# pred_score_thr=0.3 是初次预测建议值，可根据石头粘连程度调整
inferencer = DetInferencer(model=config, weights=checkpoint, device='cuda:0')

print(f"开始执行 ResNet50 + 双流解耦版预测...")
inferencer(img_dir, out_dir=out_dir, pred_score_thr=0.3, no_save_pred=False)
print(f"预测完成！结果保存在: {out_dir}/vis/")
