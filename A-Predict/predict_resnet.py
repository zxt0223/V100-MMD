import sys
import os
from mmdet.apis import DetInferencer

sys.path.insert(0, os.getcwd())
config = 'A_ResNet50/resnet_decoupled_stone.py'
checkpoint = '/mnt/old_home/chenjinming/MMD1/mmdetection/A-Out/weights/resnet/resnet_decoupled/epoch_120.pth'
img_dir = '/mnt/old_home/chenjinming/Datas/test1'
out_dir = 'A-Predict/resnet_results'

print("正在加载 ResNet50(120E) 模型...")
inferencer = DetInferencer(model=config, weights=checkpoint, device='cuda:0')
inferencer(img_dir, out_dir=out_dir, pred_score_thr=0.3, no_save_pred=False)
print(f"✅ ResNet50 预测完成！图片已保存至: {out_dir}/vis/")
