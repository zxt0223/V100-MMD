import os, re, subprocess, csv

models = {
    "Scheme3_EA_FPN": {
        "config": "/mnt/old_home/chenjinming/MMD1/mmdetection/A-Out/cascade_scheme1_EA_FPN/run_scheme1_feature_cat.py",
        "ckpt_path": "/mnt/old_home/chenjinming/MMD1/mmdetection/A-Out/weights/cascade/cascade_scheme1_EA_FPN/epoch_120.pth"
    },
    "Scheme4_SmoothEdge": {
        "config": "/mnt/old_home/chenjinming/MMD1/mmdetection/A-Out/cascade_scheme2_SmoothEdge/run_scheme2_late_fusion.py",
        "ckpt_path": "/mnt/old_home/chenjinming/MMD1/mmdetection/A-Out/weights/cascade/cascade_scheme2_SmoothEdge/epoch_120.pth"
    }
}

csv_path = "A-Predict/mAP_Comparison_Report/mAP_AP50_AP75_comparison.csv"
metrics_to_extract = ['coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75', 'coco/segm_mAP', 'coco/segm_mAP_50', 'coco/segm_mAP_75']

for name, info in models.items():
    print(f"\n{'='*60}\n🚀 正在评估新架构: {name}\n{'='*60}")
    cmd = ["python", "tools/test.py", info['config'], info['ckpt_path']]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    output_log = ""
    for line in process.stdout:
        print(line, end='')
        output_log += line
    process.wait()
    
    model_metrics = {"Model Name": name, "Weights Evaluated": "epoch_120.pth"}
    for metric in metrics_to_extract:
        matches = re.findall(rf"{metric}:\s*([0-9.]+)", output_log)
        model_metrics[metric] = float(matches[-1]) if matches else "N/A"
        
    with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=["Model Name", "Weights Evaluated"] + metrics_to_extract)
        writer.writerow(model_metrics)

print(f"\n🎉 新模型 mAP 数据追加完成！请查看: {csv_path}")
