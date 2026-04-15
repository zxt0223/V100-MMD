import os
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

out_dir = 'A-Predict/perfect_results'
vis_out_dir = os.path.join(out_dir, 'vis')          # 存放绝美图片
json_out_dir = os.path.join(out_dir, 'jsons')       # 存放带轮廓的 JSON
report_path = os.path.join(out_dir, 'area_report.csv') # 存放面积表格

os.makedirs(vis_out_dir, exist_ok=True)
os.makedirs(json_out_dir, exist_ok=True)

# 1. 初始化模型
print("🚀 正在加载模型至 GPU 1...")
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 2. 召唤官方的绝美可视化器
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]

with open(report_path, 'w', encoding='utf-8') as f_report:
    f_report.write("图片名称,石头编号,置信度得分,Mask像素面积\n")
    print(f"📊 开始执行推理：画图 + 算面积 + 存JSON ...")
    
    for img_name in images:
        img_path = os.path.join(img_dir, img_name)
        img = mmcv.imread(img_path)
        img_rgb = mmcv.imconvert(img, 'bgr', 'rgb')
        img_h, img_w = img.shape[:2]
        
        # --- 核心推理 ---
        result = inference_detector(model, img_path)
        
        # 过滤低分
        thr = 0.05
        valid_idx = result.pred_instances.scores > thr
        pred_instances = result.pred_instances[valid_idx]
        
        # --- 需求 1：让官方工具画出你最喜欢的那种图 ---
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
        
        # 准备 JSON 数据结构
        export_json_data = {
            "imagePath": img_name,
            "imageHeight": img_h,
            "imageWidth": img_w,
            "shapes": []
        }
        
        if hasattr(pred_instances, 'masks'):
            masks = pred_instances.masks.cpu().numpy() 
            scores = pred_instances.scores.cpu().numpy()
            
            for i in range(len(masks)):
                # --- 需求 2：精确计算像素面积 ---
                area_pixels = int(np.count_nonzero(masks[i]))
                f_report.write(f"{img_name},{i},{scores[i]:.4f},{area_pixels}\n")
                
                # --- 需求 3：提取多边形轮廓存入 JSON ---
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
                            "score": float(f"{scores[i]:.4f}"),
                            "area_pixels": area_pixels,
                            "points": points
                        })

        # 保存 JSON 文件
        json_save_path = os.path.join(json_out_dir, os.path.splitext(img_name)[0] + '.json')
        with open(json_save_path, 'w', encoding='utf-8') as f_json:
            json.dump(export_json_data, f_json, indent=2, ensure_ascii=False)
            
        print(f"✅ 完成: {img_name} -> 已生成图片、JSON和面积数据")

print(f"\n✨ 大功告成！去 A-Predict/perfect_results 文件夹里收货吧！")
