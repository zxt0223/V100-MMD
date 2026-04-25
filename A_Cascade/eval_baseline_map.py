import os
import re
import subprocess
import csv

# ================= 配置文件与权重路径 =================
model_name = "Baseline_Vanilla_Cascade"
# 这是你一开始告诉我的一直在用的原版母版配置
config = "A_Cascade/cascade_mask_rcnn_stone.py"
# 你刚刚提供的原版权重绝对路径
ckpt_path = "/mnt/old_home/chenjinming/MMD1/mmdetection/A-Out/weights/cascade/cascade_workdir/epoch_120.pth"

# 目标 CSV 表格路径
out_dir = "A-Predict/mAP_Comparison_Report"
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "mAP_AP50_AP75_comparison.csv")

metrics_to_extract = [
    'coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75',
    'coco/segm_mAP', 'coco/segm_mAP_50', 'coco/segm_mAP_75'
]

print(f"\n{'='*60}")
print(f"🚀 正在启动官方 COCO 评估: {model_name}")
print(f"{'='*60}")

if not os.path.exists(ckpt_path):
    print(f"❌ 错误: 找不到基线权重文件: {ckpt_path}")
    exit(1)

# 执行测试
cmd = ["python", "tools/test.py", config, ckpt_path]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

output_log = ""
for line in process.stdout:
    print(line, end='')  # 保持终端进度条滚动
    output_log += line
    
process.wait()

# 解析数据
model_metrics = {"Model Name": model_name, "Weights Evaluated": os.path.basename(ckpt_path)}
for metric in metrics_to_extract:
    matches = re.findall(rf"{metric}:\s*([0-9.]+)", output_log)
    if matches:
        model_metrics[metric] = float(matches[-1])
    else:
        model_metrics[metric] = "N/A"

# ================= 追加到现有 CSV 报告 =================
file_exists = os.path.isfile(csv_path)
with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
    fieldnames = ["Model Name", "Weights Evaluated"] + metrics_to_extract
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    # 如果文件碰巧不在，先写表头
    if not file_exists:
        writer.writeheader()
    # 追加基线数据行
    writer.writerow(model_metrics)

print(f"\n🎉 Baseline (原版 Cascade) 评估完成！")
print(f"📊 数据已成功追加到对比表格的最后一行: {csv_path}\n")
