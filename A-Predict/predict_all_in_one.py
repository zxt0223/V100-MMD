import os
import sys
sys.path.insert(0, os.getcwd())
import json
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

out_dir = 'A-Predict/ultimate_results'
vis_out_dir = os.path.join(out_dir, 'vis')
json_out_dir = os.path.join(out_dir, 'jsons')
area_report_path = os.path.join(out_dir, 'area_report.csv')
count_report_path = os.path.join(out_dir, 'count_summary.txt') # 💥 加回来的对比报告

os.makedirs(vis_out_dir, exist_ok=True)
os.makedirs(json_out_dir, exist_ok=True)

print("🚀 正在加载模型至 GPU 1...")
model = init_detector(config_file, checkpoint_file, device='cuda:2')

# 召唤官方的绝美可视化器
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]

# 同时打开两个报告文件写入
with open(area_report_path, 'w', encoding='utf-8') as f_area, \
     open(count_report_path, 'w', encoding='utf-8') as f_count:
     
    # 写入表头
    f_area.write("图片名称,石头编号,置信度得分,Mask像素面积\n")
    f_count.write(f"{'图片名称':<30} | {'真实数量(GT)':<12} | {'预测数量(Pred)':<14} | {'差值(Pred-GT)':<10}\n")
    f_count.write("-" * 75 + "\n")
    
    print(f"📊 开始执行：画图 + 对比个数差值 + 算面积 + 存JSON ...")
    
    for img_name in images:
        img_path = os.path.join(img_dir, img_name)
        img = mmcv.imread(img_path)
        img_rgb = mmcv.imconvert(img, 'bgr', 'rgb')
        img_h, img_w = img.shape[:2]
        
        # --- 💥 功能 1：读取你原有的真实标注 (GT Count) ---
        json_path = os.path.join(img_dir, os.path.splitext(img_name)[0] + '.json')
        gt_count = 0
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f_gt:
                try:
                    gt_data = json.load(f_gt)
                    gt_count = len(gt_data.get('shapes', []))
                except Exception as e:
                    print(f"读取 {img_name} 的 JSON 失败，跳过 GT 统计。")
        
        # --- 核心推理 ---
        result = inference_detector(model, img_path)
        thr = 0.05
        valid_idx = result.pred_instances.scores > thr
        pred_instances = result.pred_instances[valid_idx]
        pred_count = len(pred_instances)
        
        # --- 💥 功能 2：记录个数与差值 ---
        diff = pred_count - gt_count
        f_count.write(f"{img_name:<30} | {gt_count:<12} | {pred_count:<14} | {diff:<10}\n")
        
        # --- 💥 功能 3：让官方工具画出你最喜欢的那种完美图片 ---
        visualizer.add_datasample(
            name=img_name,
            image=img_rgb,
            data_sample=result,
            draw_gt=False,
            draw_pred=True,
            show=False,
            wait_time=0,
            out_file=os.path.join(vis_out_dir, img_name),
            pred_score_thr=thr
        )
        
        # 准备生成新的 LabelMe JSON 的数据结构
        export_json_data = {
            "version": "5.2.1",
            "flags": {},
            "imagePath": img_name,
            "imageHeight": img_h,
            "imageWidth": img_w,
            "imageData": None,
            "shapes": []
        }
        
        if hasattr(pred_instances, 'masks'):
            masks = pred_instances.masks.cpu().numpy() 
            scores = pred_instances.scores.cpu().numpy()
            
            for i in range(len(masks)):
                # --- 💥 功能 4：精确计算像素面积 ---
                area_pixels = int(np.count_nonzero(masks[i]))
                f_area.write(f"{img_name},{i},{scores[i]:.4f},{area_pixels}\n")
                
                # --- 💥 功能 5：提取多边形轮廓存入新的 LabelMe JSON ---
                mask_uint8 = (masks[i] * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) > 0:
                    c = max(contours, key=cv2.contourArea)
                    epsilon = 0.001 * cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, epsilon, True)
                    
                    if len(approx) >= 3:
                        points = approx.reshape(-1, 2).tolist()
                        export_json_data["shapes"].append({
                            "label": "stone",
                            "points": points,
                            "group_id": None,
                            "description": f"score: {scores[i]:.4f}, area: {area_pixels}",
                            "shape_type": "polygon",
                            "flags": {}
                        })

        # 保存生成的 JSON 文件
        json_save_path = os.path.join(json_out_dir, os.path.splitext(img_name)[0] + '.json')
        with open(json_save_path, 'w', encoding='utf-8') as f_out_json:
            json.dump(export_json_data, f_out_json, indent=2, ensure_ascii=False)
            
        print(f"✅ {img_name} [真实: {gt_count} | 预测: {pred_count} | 差值: {diff}]")

print(f"\n✨ 大功告成！所有输出都在 A-Predict/ultimate_results 文件夹内！")
