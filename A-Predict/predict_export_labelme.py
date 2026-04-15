import os
import json
import torch
import cv2
import mmcv
import numpy as np
from mmdet.apis import init_detector, inference_detector

# ================= 配置区 =================
config_file = 'A_Cascade/cascade_mask_rcnn_boundary_stone.py'
checkpoint_file = '/mnt/old_home/chenjinming/MMD1/mmdetection/A-Out/weights/cascade/cascade_boundary_workdir/epoch_120.pth'
img_dir = '/mnt/old_home/chenjinming/Datas/test1'

# 输出文件夹
out_dir = 'A-Predict/cascade_final_results'
report_path = os.path.join(out_dir, 'area_report.txt')
json_out_dir = os.path.join(out_dir, 'labelme_jsons') # 存放预测生成的 JSON
vis_out_dir = os.path.join(out_dir, 'vis')

os.makedirs(json_out_dir, exist_ok=True)
os.makedirs(vis_out_dir, exist_ok=True)

# 1. 初始化模型
print("🚀 正在加载终极模型至 GPU 1...")
model = init_detector(config_file, checkpoint_file, device='cuda:0')

img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
images = [f for f in os.listdir(img_dir) if f.lower().endswith(img_extensions)]

with open(report_path, 'w', encoding='utf-8') as f_report:
    # 写入表头
    f_report.write("图片名称,石头编号,置信度得分,Mask像素面积\n")
    print(f"📊 开始提取 Mask 面积并生成 LabelMe JSON...")
    
    for img_name in images:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        
        # --- 推理 ---
        result = inference_detector(model, img_path)
        
        thr = 0.05
        valid_idx = result.pred_instances.scores > thr
        pred_instances = result.pred_instances[valid_idx]
        
        # 准备 LabelMe 格式的字典结构
        labelme_dict = {
            "version": "5.2.1",
            "flags": {},
            "shapes": [],
            "imagePath": img_name,
            "imageData": None,
            "imageHeight": img_h,
            "imageWidth": img_w
        }
        
        if hasattr(pred_instances, 'masks'):
            # 转为 numpy
            masks = pred_instances.masks.cpu().numpy() 
            scores = pred_instances.scores.cpu().numpy()
            
            for i in range(len(masks)):
                # 1. 修复面积为 0 的 Bug：使用 count_nonzero 绝对准确地统计 True 的像素点
                area_pixels = int(np.count_nonzero(masks[i]))
                
                # 写入你要求的 CSV/TXT 格式
                f_report.write(f"{img_name},{i},{scores[i]:.4f},{area_pixels}\n")
                
                # 2. 生成 LabelMe 多边形 (Polygon)
                # 将布尔型 Mask 转为 uint8 格式供 OpenCV 提取轮廓
                mask_uint8 = (masks[i] * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) > 0:
                    # 找到面积最大的那个轮廓（防止噪点产生碎片）
                    c = max(contours, key=cv2.contourArea)
                    
                    # 多边形拟合，减少 JSON 文件体积（使边缘平滑且节点不至于多到卡死软件）
                    epsilon = 0.001 * cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, epsilon, True)
                    
                    if len(approx) >= 3:
                        points = approx.reshape(-1, 2).tolist()
                        
                        # 组装 Shape 字典
                        shape = {
                            "label": "stone",
                            "points": points,
                            "group_id": None,
                            "description": f"score: {scores[i]:.3f}",
                            "shape_type": "polygon",
                            "flags": {}
                        }
                        labelme_dict["shapes"].append(shape)
                        
                        # 在图上画出来看看 (可选，只画轮廓)
                        cv2.polylines(img, [approx], True, (0, 255, 0), 2)
                        # 写上置信度
                        cv2.putText(img, f"{scores[i]:.2f}", tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 保存生成的 LabelMe JSON 文件
        json_save_path = os.path.join(json_out_dir, os.path.splitext(img_name)[0] + '.json')
        with open(json_save_path, 'w', encoding='utf-8') as f_json:
            json.dump(labelme_dict, f_json, indent=2, ensure_ascii=False)
            
        # 保存可视化图片
        cv2.imwrite(os.path.join(vis_out_dir, img_name), img)
        
        print(f"✅ 完成: {img_name} [发现 {len(pred_instances)} 块石头, JSON 已生成]")

print(f"\n✨ 任务彻底完成！")
print(f"📄 面积统计报告: {report_path}")
print(f"📁 LabelMe JSON 文件: {json_out_dir}/")
