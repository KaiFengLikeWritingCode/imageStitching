"""
FUnIE-GAN微调训练程序
"""

import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import *
from components.funiegan_finetuner import SimpleFinetuner
from components.funiegan_enhancer import FUnIE_GAN_Enhancer
from utils.metrics import compute_metrics
from utils.visualization import visualize_metrics


def train_and_validate_finetune(args):
    """FUnIE-GAN微调训练与验证"""

    # 参数处理
    input_dir = args.input_dir or FINETUNE_CONFIG['input_dir']
    target_dir = args.target_dir or FINETUNE_CONFIG['target_dir']
    val_dir = args.val_dir
    output_model_path = args.output_model_path or FUNIEGAN_FINETUNED_PATH
    base_model_path = args.base_model_path or FUNIEGAN_MODEL_PATH

    # 创建日志目录
    log_dir = os.path.join(LOGS_DIR, f"finetune_{os.path.basename(output_model_path)}")
    os.makedirs(log_dir, exist_ok=True)

    # 保存训练配置
    with open(os.path.join(log_dir, "training_config.txt"), "w") as f:
        f.write(f"基础模型: {base_model_path}\n")
        f.write(f"训练数据: {input_dir}, {target_dir}\n")
        f.write(f"验证数据: {val_dir}\n")
        f.write(f"输出模型: {output_model_path}\n")
        f.write(f"训练轮次: {args.epochs}\n")
        f.write(f"批次大小: {args.batch_size}\n")
        f.write(f"学习率: {args.lr}\n")
        f.write(f"保存间隔: {args.save_interval}\n")

    # 初始化微调器
    print("初始化FUnIE-GAN微调器...")
    finetuner = SimpleFinetuner(
        model_path=base_model_path,
        lr=args.lr
    )

    # 训练前评估基础模型
    if val_dir:
        print("\n评估基础模型性能...")
        orig_enhancer = FUnIE_GAN_Enhancer(base_model_path)
        base_metrics = evaluate_enhancer(orig_enhancer, val_dir, os.path.join(log_dir, "base_eval"))

    # 执行微调
    print("\n开始FUnIE-GAN微调...")
    finetuner.fine_tune(
        input_dir=input_dir,
        target_dir=target_dir,
        output_model_path=output_model_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_interval=args.save_interval
    )

    # 训练后评估微调模型
    if val_dir:
        print("\n评估微调后模型性能...")
        finetuned_enhancer = FUnIE_GAN_Enhancer(output_model_path)
        finetuned_metrics = evaluate_enhancer(finetuned_enhancer, val_dir, os.path.join(log_dir, "finetuned_eval"))

        # 比较性能
        compare_performance(base_metrics, finetuned_metrics, os.path.join(log_dir, "performance_comparison.png"))

    print(f"FUnIE-GAN微调训练完成！结果保存在 {log_dir}")


def evaluate_enhancer(enhancer, val_dir, output_dir):
    """评估增强器性能"""
    os.makedirs(output_dir, exist_ok=True)

    # 查找验证图像
    val_files = [f for f in os.listdir(val_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"找到 {len(val_files)} 个验证图像")

    # 评估结果
    all_metrics = {}

    # 处理验证图像
    for i, filename in enumerate(tqdm(val_files, desc="评估增强效果")):
        img_path = os.path.join(val_dir, filename)

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 增强图像
        enhanced = enhancer.enhance(img)

        # 计算指标
        metrics = compute_metrics(enhanced)
        for key, value in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = []
            all_metrics[key].append(value)

        # 保存可视化结果 (每10张图保存一次)
        if i % 10 == 0:
            comparison = np.hstack([img, enhanced])
            output_path = os.path.join(output_dir, f"enhanced_{filename}")
            cv2.imwrite(output_path, comparison)

    # 计算平均指标
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}

    # 保存指标信息
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        for key, value in avg_metrics.items():
            f.write(f"{key}: {value}\n")

    # 可视化整体指标
    plt.figure(figsize=(10, 6))
    plt.bar(avg_metrics.keys(), avg_metrics.values())
    plt.title("增强性能指标")
    plt.ylabel("分数")
    plt.savefig(os.path.join(output_dir, "metrics.png"))
    plt.close()

    return avg_metrics


def compare_performance(base_metrics, finetuned_metrics, output_path):
    """比较基础模型和微调模型的性能"""

    # 准备数据
    metrics = list(base_metrics.keys())
    base_values = [base_metrics[m] for m in metrics]
    finetuned_values = [finetuned_metrics[m] for m in metrics]

    # 计算改进百分比
    improvement = []
    for i, m in enumerate(metrics):
        # 对于CE，越低越好，其他指标越高越好
        if m == 'ce':
            imp = (base_values[i] - finetuned_values[i]) / base_values[i] * 100
        else:
            imp = (finetuned_values[i] - base_values[i]) / base_values[i] * 100
        improvement.append(imp)

    # 创建图表
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # 绘制指标对比
    x = np.arange(len(metrics))
    width = 0.35

    bar1 = ax1.bar(x - width / 2, base_values, width, label='基础模型')
    bar2 = ax1.bar(x + width / 2, finetuned_values, width, label='微调模型')

    ax1.set_title('基础模型 vs 微调模型性能对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()

    # 添加改进百分比标签
    for i, imp in enumerate(improvement):
        color = 'green' if imp > 0 else 'red'
        if metrics[i] == 'ce':  # CE值越低越好，所以改进方向相反
            color = 'green' if imp > 0 else 'red'

        ax1.annotate(f"{imp:.1f}%",
                     xy=(x[i], max(base_values[i], finetuned_values[i]) + 0.02),
                     ha='center', va='bottom',
                     color=color, fontweight='bold')

    # 保存图表
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="FUnIE-GAN微调训练")
    parser.add_argument('--input_dir', help="训练输入图像目录")
    parser.add_argument('--target_dir', help="训练目标图像目录")
    parser.add_argument('--val_dir', help="验证图像目录")
    parser.add_argument('--base_model_path', help="基础模型路径")
    parser.add_argument('--output_model_path', help="输出模型路径")
    parser.add_argument('--batch_size', type=int, default=4, help="批次大小")
    parser.add_argument('--epochs', type=int, default=10, help="训练轮次")
    parser.add_argument('--lr', type=float, default=0.00001, help="学习率")
    parser.add_argument('--save_interval', type=int, default=2, help="保存间隔")

    args = parser.parse_args()

    # 执行训练与验证
    train_and_validate_finetune(args)