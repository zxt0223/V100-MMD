import os
import json
import cv2
import numpy as np
import csv
import argparse
from mmdet.apis import DetInferencer

# 将 LabelMe 的多边形点位转换为二值化 Mask 掩膜
def poly2mask(points, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask

# 计算两个 Mask 的交并比 (IoU)
def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--img-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--thr', type=float, default=0.5)
    parser.add_argument('--iou-thr', type=float, default=0.85) # 85% 重叠度硬约束
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = os.path.join(args.out_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, 'prediction_vs_actual_report.csv')
    
    print(f"\n🚀 [1/3] 正在加载模型权重...")
    inferencer = DetInferencer(model=args.config, weights=args.checkpoint)
    
    files = os.listdir(args.img_dir)
    img_files = [f for f in files if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    report_data = []
    
    print(f"🔍 [2/3] 开始逐图预测与 JSON 标注核对 (IoU 阈值: {args.iou_thr})...")
    for img_name in img_files:
        img_path = os.path.join(args.img_dir, img_name)
        json_path = os.path.join(args.img_dir, os.path.splitext(img_name)[0] + '.json')
        
        # 1. 模型预测提取 Mask (修复：增加了 s，并增加鲁棒性解析)
        result = inferencer(img_path, return_datasamples=True)
        
        # 兼容 MMEngine 不同版本的返回格式 (字典或列表)
        if isinstance(result, dict) and 'predictions' in result:
            data_sample = result['predictions'][0]
        else:
            data_sample = result[0]
        
        pred_masks = []
        if hasattr(data_sample, 'pred_instances'):
            pred_instances = data_sample.pred_instances
            scores = pred_instances.scores.cpu().numpy()
            valid_inds = scores > args.thr
            if hasattr(pred_instances, 'masks'):
                pred_masks = pred_instances.masks[valid_inds].cpu().numpy()
        
        pred_count = len(pred_masks)
        gt_count = 0
        gt_masks = []
        
        # 2. 读取真实标注 JSON
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                shapes = data.get('shapes', [])
                gt_count = len(shapes)
                for shape in shapes:
                    gt_masks.append(poly2mask(shape['points'], h, w))
        else:
            print(f"⚠️ 警告: 图片 {img_name} 缺少同名的 JSON 标注文件，无法核对数量与面积。")
            continue
            
        # 3. 严格匹配 (贪心算法) 与计算差值
        matched_gt = set()
        matched_pred = set()
        area_diffs = []
        
        for p_idx, p_mask in enumerate(pred_masks):
            best_iou = 0
            best_g_idx = -1
            for g_idx, g_mask in enumerate(gt_masks):
                if g_idx in matched_gt: continue
                iou = compute_iou(p_mask, g_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx
            
            if best_iou >= args.iou_thr:
                matched_gt.add(best_g_idx)
                matched_pred.add(p_idx)
                # 面积差值 = 预测面积 - 真实面积 (正数说明预测偏大，负数说明预测偏小)
                p_area = p_mask.sum()
                g_area = gt_masks[best_g_idx].sum()
                area_diffs.append(p_area - g_area)
        
        # 4. 核心指标统计
        count_diff = pred_count - gt_count
        correct_matches = len(matched_pred)
        false_positives = pred_count - correct_matches # 错把背景或别人当成目标的数量
        false_negatives = gt_count - correct_matches   # 没预测出来，漏检的数量
        avg_area_diff = np.mean(area_diffs) if area_diffs else 0
        
        report_data.append([
            img_name, gt_count, pred_count, count_diff, 
            correct_matches, false_positives, false_negatives, round(avg_area_diff, 2)
        ])
        
        # 保存基础的可视化图片供人眼复核
        inferencer(img_path, out_dir=vis_dir, pred_score_thr=args.thr)

    # 5. 写入报告文件
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Image Name', 'Actual Count', 'Predicted Count', 'Count Difference', 
                         'Correct Matches (IoU>85%)', 'False Positives (Mis-predicted)', 
                         'False Negatives (Missed)', 'Avg Area Difference (Pixels)'])
        writer.writerows(report_data)
        
    print(f"\n✅ [3/3] 评估完成！")
    print(f"📊 详细核对表已生成: {csv_path}")
    print(f"🖼️ 可视化图片已保存: {vis_dir}")

if __name__ == '__main__':
    main()
