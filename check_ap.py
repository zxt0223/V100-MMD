import re
import glob
import os

def parse_log(log_path):
    metrics = {'bbox': [0,0,0], 'segm': [0,0,0]} # AP, AP50, AP75
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 提取画框 AP
                if 'coco/bbox_mAP:' in line: metrics['bbox'][0] = max(metrics['bbox'][0], float(re.search(r'coco/bbox_mAP:\s+([\d\.]+)', line).group(1)))
                if 'coco/bbox_mAP_50:' in line: metrics['bbox'][1] = max(metrics['bbox'][1], float(re.search(r'coco/bbox_mAP_50:\s+([\d\.]+)', line).group(1)))
                if 'coco/bbox_mAP_75:' in line: metrics['bbox'][2] = max(metrics['bbox'][2], float(re.search(r'coco/bbox_mAP_75:\s+([\d\.]+)', line).group(1)))
                # 提取掩码 AP
                if 'coco/segm_mAP:' in line: metrics['segm'][0] = max(metrics['segm'][0], float(re.search(r'coco/segm_mAP:\s+([\d\.]+)', line).group(1)))
                if 'coco/segm_mAP_50:' in line: metrics['segm'][1] = max(metrics['segm'][1], float(re.search(r'coco/segm_mAP_50:\s+([\d\.]+)', line).group(1)))
                if 'coco/segm_mAP_75:' in line: metrics['segm'][2] = max(metrics['segm'][2], float(re.search(r'coco/segm_mAP_75:\s+([\d\.]+)', line).group(1)))
    except Exception as e:
        pass
    return metrics

print("\n" + "="*50)
print(f"{'模型评估指标对比 (Max mAP)':^48}")
print("="*50)

log_files = glob.glob('A-Out/terminal_logs/*.log')
for log in log_files:
    model_name = "LEGNet + 双流" if "LEGNet" in log else "ResNet50 + 双流"
    m = parse_log(log)
    print(f"\n【{model_name}】")
    print(f"📄 日志源: {os.path.basename(log)}")
    print("-" * 30)
    print(f"🎯 目标检测 (BBox):")
    print(f"   AP   (综合): {m['bbox'][0]:.3f}")
    print(f"   AP50 (宽松): {m['bbox'][1]:.3f}")
    print(f"   AP75 (严格): {m['bbox'][2]:.3f}")
    print(f"🧩 实例分割 (Segm / Mask):")
    print(f"   AP   (综合): {m['segm'][0]:.3f}")
    print(f"   AP50 (宽松): {m['segm'][1]:.3f}")
    print(f"   AP75 (严格): {m['segm'][2]:.3f}")
print("\n" + "="*50)
