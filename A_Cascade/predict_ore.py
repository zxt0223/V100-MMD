import os
import argparse
from mmdet.apis import DetInferencer

def main():
    parser = argparse.ArgumentParser(description='OreSegNet Inference Script')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint file path')
    parser.add_argument('--img-dir', required=True, help='Directory containing images to predict')
    parser.add_argument('--out-dir', required=True, help='Directory to save visualization results')
    parser.add_argument('--thr', type=float, default=0.5, help='Prediction score threshold')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n🚀 [1/3] 加载模型与权重: {args.checkpoint}...")
    
    # 实例化推理器 (MMDet 3.x 高级 API，自动处理可视化)
    inferencer = DetInferencer(model=args.config, weights=args.checkpoint)
    
    print(f"🔍 [2/3] 开始对 {args.img_dir} 中的图片进行预测...")
    # 执行推理，结果（包含画好 bbox 和 mask 的图片）会自动保存
    inferencer(args.img_dir, out_dir=args.out_dir, pred_score_thr=args.thr)
    
    print(f"✅ [3/3] 预测完成！可视化结果已保存至: {args.out_dir}\n")

if __name__ == '__main__':
    main()
