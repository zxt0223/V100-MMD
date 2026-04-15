import os
import torch
import cv2
import mmcv
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS

# ================= 配置区 =================
config_file = 'A_Cascade/cascade_mask_rcnn_boundary_stone.py'
checkpoint_file = '/mnt/old_home/chenjinming/MMD1/mmdetection/A-Out/weights/cascade/cascade_boundary_workdir/epoch_120.pth'
img_dir = '/mnt/old_home/chenjinming/Datas/test1'
out_dir = 'A-Predict/cascade_area_results'
report_path = os.path.join(out_dir, 'area_report.csv')

os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.join(out_dir, 'vis'), exist_ok=True)

# 1. 初始化模型
print("🚀 正在加载模型至 GPU 1...")
model = init_detector(config_file, checkpoint_file, device='cuda:0')

visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
images = [f for f in os.listdir(img_dir) if f.lower().endswith(img_extensions)]

# 准备写入 CSV 报告（你可以直接用 Excel 打开）
with open(report_path, 'w', encoding='utf-8') as f_report:
    f_report.write("图片名称,石头编号,置信度得分,Mask像素面积\n")

    print(f"📊 开始处理并提取底层 Mask 数据...")
    
    for img_name in images:
        img_path = os.path.join(img_dir, img_name)
        
        # --- 核心推理 ---
        result = inference_detector(model, img_path)
        
        # 设定分数门槛 0.05
        thr = 0.05
        valid_idx = result.pred_instances.scores > thr
        pred_instances = result.pred_instances[valid_idx]
        
        # --- 💥 核心修改：提取底层 Mask 矩阵计算面积 ---
        if hasattr(pred_instances, 'masks'):
            # 将 GPU 上的布尔矩阵转移到 CPU 并转为 numpy 数组
            # 形状为 [石头数量, 图像高度, 图像宽度]
            masks = pred_instances.masks.cpu().numpy() 
            scores = pred_instances.scores.cpu().numpy()
            
            for i in range(len(masks)):
                # 一块石头的 Mask 就是一个二维数组，里面石头区域是 True (1)，背景是 False (0)
                # 直接求和，就能得到这块石头的绝对像素面积！
                area_pixels = masks[i].sum() 
                
                # 写入报告
                f_report.write(f"{img_name},{i},{scores[i]:.4f},{area_pixels}\n")
        
        # --- 修复报错的渲染部分 ---
        img = mmcv.imread(img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        
        visualizer.add_datasample(
            name=img_name,
            image=img,
            data_sample=result,
            draw_gt=False,        # 不画真实框
            draw_pred=True,       # 只要有 pred_instances.masks，这里默认就会画出 Mask！
            show=False,
            wait_time=0,
            out_file=os.path.join(out_dir, 'vis', img_name),
            pred_score_thr=thr
        )
        
        print(f"✅ 已处理: {img_name} [发现 {len(pred_instances)} 块石头, 面积已记录]")

print(f"\n✨ 任务完成！")
print(f"📁 可视化图片已保存至: {out_dir}/vis/")
print(f"📄 面积统计表格已保存至: {report_path} (可直接用 Excel 打开分析粒度分布！)")
