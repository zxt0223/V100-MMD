import os
import json
import torch
import cv2
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from mmengine.structures import InstanceData

# ================= 配置区 =================
config_file = 'A_Cascade/cascade_mask_rcnn_boundary_stone.py'
checkpoint_file = '/mnt/old_home/chenjinming/MMD1/mmdetection/A-Out/weights/cascade/cascade_boundary_workdir/epoch_120.pth'
img_dir = '/mnt/old_home/chenjinming/Datas/test1'
out_dir = 'A-Predict/cascade_boundary_results' # 干净的输出目录
report_path = os.path.join(out_dir, 'count_diff_report.txt')

os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.join(out_dir, 'vis'), exist_ok=True)

# 1. 初始化模型 (GPU 1)
print("🚀 正在加载模型至 GPU 1...")
model = init_detector(config_file, checkpoint_file, device='cuda:0') # 外部指定了GPU1，这里填cuda:0

# 2. 初始化可视化器 (强制开启 Mask 渲染并精确控制阈值)
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

# 3. 开始遍历文件
img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
images = [f for f in os.listdir(img_dir) if f.lower().endswith(img_extensions)]

with open(report_path, 'w', encoding='utf-8') as f_report:
    # 增加'差值'列
    f_report.write(f"{'文件名':<30} | {'真实数量(GT)':<12} | {'预测数量(Pred)':<12} | {'差值(GT-Pred)':<14}\n")
    f_report.write("-" * 80 + "\n")

    print(f"📊 开始处理 {len(images)} 张图片...")
    
    for img_name in images:
        img_path = os.path.join(img_dir, img_name)
        json_path = os.path.join(img_dir, os.path.splitext(img_name)[0] + '.json')
        
        # --- 计算真实标注 (LabelMe JSON) ---
        gt_count = 0
        if os.path.exists(json_path):
            with open(json_path, 'r') as f_json:
                labelme_data = json.load(f_json)
                gt_count = len(labelme_data.get('shapes', []))
        
        # --- 执行推理 ---
        result = inference_detector(model, img_path)
        
        # 这里过滤一个能看到更多 Mask 的低分数门槛 (比如设为 0.1)
        # 即使这里过滤了框，如果下面可视化时的阈值太高，Mask 像素依然不会画出来
        mask_count_score_thr = 0.1 
        pred_instances = result.pred_instances[result.pred_instances.scores > mask_count_score_thr]
        pred_count = len(pred_instances)
        
        # 计算差值
        diff_count = gt_count - pred_count
        
        # --- 渲染并保存图片 (💥 这里的参数是解决无 Mask 的关键) ---
        img = mmcv.imread(img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        
        visualizer.add_datasample(
            img_name,
            img,
            data_sample=result,
            draw_gt=False,
            show=False,
            wait_time=0,
            out_file=os.path.join(out_dir, 'vis', img_name),
            draw_gt_mask=False,
            draw_gt_bbox=False,
            # 💥 强制将可视化时的掩码二值化阈值调低到 0.1！
            mask_thr_binary=0.1, 
            pred_score_thr=0.1  # 显示框的分数也同步调低
        )
        
        # --- 写入报告 (标注差值) ---
        f_report.write(f"{img_name:<30} | {gt_count:<12} | {pred_count:<12} | {diff_count:<14}\n")
        print(f"✅ 已处理: {img_name} [GT: {gt_count} | Pred: {pred_count} | Diff: {diff_count}]")

print(f"\n✨ 任务完成！")
print(f"📁 可视化图片 (带绿色 Mask): {out_dir}/vis/")
print(f"📄 对比报告 (带差值标注): {report_path}")
