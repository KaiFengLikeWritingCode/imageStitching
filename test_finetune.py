"""
微调模型测试评估程序
"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import *
from components.funiegan_enhancer import FUnIE_GAN_Enhancer
from utils.metrics import compute_metrics


def test_finetuned_model(args):
    """测试微调后的FUnIE-GAN模型"""

    # 参数处理
    model_path = args.model_path or FUNIEGAN_FINETUNED_PATH
    test_dir = args.test_dir
    base_model_path = args.base_model_path or FUNIEGAN_MODEL_PATH
    output_dir = args.output_dir or os.path.join(RESULTS_DIR, f"finetune_test_{os.path.basename(model_path)}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    print(f"加载基础模型: {base_model_path}")
    base_enhancer = FUnIE_GAN_Enhancer(base_model_path)

    print(f"加载微调模型: {model_path}")
    finetuned_enhancer = FUnIE_GAN_Enhancer(model_path)

    # 查找测试图像
    test_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"找到 {len(test_files)} 个测试图像")

    # 评估结果
    base_metrics = {}
    finetuned_metrics = {}

    # 处理测试图像
    for filename in tqdm(test_files, desc="测试增强效果"):
        img_path = os.path.join(test_dir, filename)

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 基础模型增强
        base_enhanced = base_enhancer.enhance(img)

        # 微调模型增强
        finetuned_enhanced = finetuned_enhancer.enhance(img)

        # 计算指标
        base_metric = compute_metrics(base_enhanced)
        finetuned_metric = compute_metrics(finetuned_enhanced)

        # 收集指标
        for key, value in base_metric.items():
            if key not in base_metrics:
                base_metrics[key] = []
            base_metrics[key].append(value)

        for key, value in finetuned_metric.items():
            if key not in finetuned_metrics:
                finetuned_metrics[key] = []
            finetuned_metrics[key].append(value)

        # 保存可视化结果
        result_vis = create_comparison_visualization(
            img, base_enhanced, finetuned_enhanced,
            base_metric, finetuned_metric,
            filename
        )

        output_path = os.path.join(output_dir, f"compare_{filename}")
        cv2.imwrite(output_path, result_vis)

    # 计算平均指标
    avg_base = {k: np.mean(v) for k, v in base_metrics.items()}
    avg_finetuned = {k: np.mean(v) for k, v in finetuned_metrics.items()}

    # 保存比较报告
    create_comparison_report(avg_base, avg_finetuned, output_dir)

    print(f"测试完成！结果保存在 {output_dir}")


def create_comparison_visualization(original, base_enhanced, finetuned_enhanced,
                                    base_metrics, finetuned_metrics, filename):
    """创建对比可视化"""

    # 创建画布
    h, w = original.shape[:2]
    canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

    # 放置图像
    canvas[:h, :w] = original
    canvas[:h, w:w * 2] = base_enhanced
    canvas[h:h * 2, :w] = finetuned_enhanced

    # 计算差异图
    diff = cv2.absdiff(base_enhanced, finetuned_enhanced)
    diff = cv2.applyColorMap(diff, cv2.COLORMAP_JET)  # 应用热图彩色映射
    canvas[h:h * 2, w:w * 2] = diff

    # 添加标题
    cv2.putText(canvas, "原始图像", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(canvas, "基础模型增强", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(canvas, "微调模型增强", (10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(canvas, "差异图", (w + 10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 添加文件名
    cv2.putText(canvas, filename, (10, h * 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 添加指标信息
    y_offset = 60
    for metric, value in base_metrics.items():
        text = f"{metric}: {value:.3f}"
        cv2.putText(canvas, text, (w + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30

    y_offset = h + 60
    for metric, value in finetuned_metrics.items():
        text = f"{metric}: {value:.3f}"
        cv2.putText(canvas, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30

    return canvas


def create_comparison_report(base_metrics, finetuned_metrics, output_dir):
    """创建比较报告"""

    # 保存指标信息
    with open(os.path.join(output_dir, "comparison_report.txt"), "w") as f:
        f.write("指标比较: 基础模型 vs 微调模型\n")
        f.write("=" * 40 + "\n\n")

        for key in base_metrics.keys():
            base_val = base_metrics[key]
            finetuned_val = finetuned_metrics[key]

            # 计算改进比例
            if key == 'ce':  # CE越低越好
                improvement = (base_val - finetuned_val) / base_val * 100
                better = "改进" if improvement > 0 else "退步"
            else:  # 其他指标越高越好
                improvement = (finetuned_val - base_val) / base_val * 100
                better = "改进" if improvement > 0 else "退步"

            f.write(f"{key}:\n")
            f.write(f"  基础模型: {base_val:.4f}\n")
            f.write(f"  微调模型: {finetuned_val:.4f}\n")
            f.write(f"  变化率: {abs(improvement):.2f}% ({better})\n\n")

    # 创建对比图表
    metrics = list(base_metrics.keys())
    base_values = [base_metrics[m] for m in metrics]
    finetuned_values = [finetuned_metrics[m] for m in metrics]

    # 计算改进百分比
    improvement = []
    for i, m in enumerate(metrics):
        if m == 'ce':  # CE值越低越好
            imp = (base_values[i] - finetuned_values[i]) / base_values[i] * 100
        else:
            imp = (finetuned_values[i] - base_values[i]) / base_values[i] * 100
        improvement.append(imp)

    # 绘制对比图表
    plt.figure(figsize=(12, 8))

    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(x - width / 2, base_values, width, label='基础模型')
    plt.bar(x + width / 2, finetuned_values, width, label='微调模型')

    plt.title('基础模型 vs 微调模型性能对比')
    plt.xticks(x, metrics)
    plt.legend()

    # 添加改进百分比标签
    for i, imp in enumerate(improvement):
        color = 'green' if imp > 0 else 'red'
        if metrics[i] == 'ce':  # CE值越低越好，所以改进方向相反
            color = 'green' if imp > 0 else 'red'

        plt.annotate(f"{imp:.1f}%",
                     xy=(x[i], max(base_values[i], finetuned_values[i]) + 0.02),
                     ha='center', va='bottom',
                     color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"))
    plt.close()

    # 创建饼图展示总体改进情况
    improved_count = sum(1 for imp in improvement if imp > 0)
    total_count = len(improvement)

    plt.figure(figsize=(8, 8))
    plt.pie([improved_count, total_count - improved_count],
            labels=['已改进', '未改进'],
            autopct='%1.1f%%',
            colors=['green', 'red'])
    plt.title('微调模型改进比例')
    plt.savefig(os.path.join(output_dir, "improvement_ratio.png"))
    plt.close()


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="测试微调后的FUnIE-GAN模型")
    parser.add_argument('--model_path', help="微调模型路径")
    parser.add_argument('--base_model_path', help="基础模型路径")
    parser.add_argument('--test_dir', required=True, help="测试图像目录")
    parser.add_argument('--output_dir', help="输出目录")

    args = parser.parse_args()

    # 执行测试
    test_finetuned_model(args)