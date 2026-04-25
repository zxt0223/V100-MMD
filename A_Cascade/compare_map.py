import os
import re
import subprocess
import csv
import glob

# ================= 配置文件与权重路径 =================
models = {
    "Scheme1_FeatureCat": {
        "config": "A_Cascade/configs/run_scheme1_feature_cat.py",
        "ckpt_dir": "/mnt/old_home/chenjinming/MMD1/mmdetection/A-Out/weights/cascade/cascade_feature_cat_workdir",
        "target_epoch": "epoch_120.pth"
    },
    "Scheme2_LateFusion": {
        "config": "A_Cascade/configs/run_scheme2_late_fusion.py",
        "ckpt_dir": "/mnt/old_home/chenjinming/MMD1/mmdetection/A-Out/weights/cascade/cascade_late_fusion_workdir",
        "target_epoch": "epoch_120.pth"
    }
}

out_dir = "A-Predict/mAP_Comparison_Report"
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "mAP_AP50_AP75_comparison.csv")

# 你需要抓取的核心指标
metrics_to_extract = [
    'coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75',
    'coco/segm_mAP', 'coco/segm_mAP_50', 'coco/segm_mAP_75'
]

results = []

for name, info in models.items():
    print(f"\n{'='*60}")
    print(f"�� 正在启动官方 COCO 评估: {name}")
    print(f"{'='*60}")
    
    # 智能寻找权重：优先找 epoch_120.pth，如果不存在（可能还没跑完），就抓取最新生成的权重
    ckpt_path = os.path.join(info['ckpt_dir'], info['target_epoch'])
    if not os.path.exists(ckpt_path):
        pths = glob.glob(os.path.join(info['ckpt_dir'], '*.pth'))
        if pths:
            ckpt_path = max(pths, key=os.path.getmtime)
            print(f"⚠️ 提示: 未找到 120 轮权重，自动替换为该方案目前最新的权重: {os.path.basename(ckpt_path)}")
        else:
            print(f"❌ 错误: 找不到 {name} 的任何权重文件，已跳过。")
            continue

    # 构造并执行 MMDetection 官方测试命令
    cmd = ["python", "tools/test.py", info['config'], ckpt_path]
    
    # 使用 subprocess 捕获输出，并实时打印到终端
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    output_log = ""
    for line in process.stdout:
        print(line, end='')  # 保持终端进度条滚动
        output_log += line
        
    process.wait()
    
    # 使用正则表达式从繁杂的日志中精准提取各项 mAP 数据
    model_metrics = {"Model Name": name, "Weights Evaluated": os.path.basename(ckpt_path)}
    for metric in metrics_to_extract:
        # 匹配日志中类似 "coco/segm_mAP_50: 0.9480" 的结构
        matches = re.findall(rf"{metric}:\s*([0-9.]+)", output_log)
        if matches:
            model_metrics[metric] = float(matches[-1])
        else:
            model_metrics[metric] = "N/A"
            
    results.append(model_metrics)

# ================= 写入 CSV 报告 =================
with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    # 增加一列表明到底评估的是哪一个权重文件
    fieldnames = ["Model Name", "Weights Evaluated"] + metrics_to_extract
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\n🎉 评估全部完成！")
print(f"📊 模型能力横向对比数据已保存至: {csv_path}\n")
