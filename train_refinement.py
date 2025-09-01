"""
无监督拼接优化网络训练与测试脚本
"""

import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json
from tqdm import tqdm

# 设置TensorFlow内存增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 导入配置
from config import *

# 导入组件
from components.unsupervised_refinement import (
    UnsupervisedRefinementNetwork,
    UnsupervisedTrainer,
    create_dataset_from_stitched_images,
    StitchingMaskGenerator
)

# 导入工具函数
from utils.metrics import compute_metrics


class RefinementTrainer:
    """无监督拼接优化网络训练器"""

    def __init__(self, config):
        """初始化训练器"""
        self.config = config
        self.setup_components()

    def setup_components(self):
        """设置组件"""
        # 初始化网络
        print("初始化拼接优化网络...")
        self.refinement_network = UnsupervisedRefinementNetwork()

        # 初始化蒙版生成器
        self.mask_generator = StitchingMaskGenerator(dilate_kernel_size=7)

    def train(self, data_dir, output_dir, epochs=20, batch_size=4, image_size=(256, 256), learning_rate=0.0002):
        """
        训练优化网络

        参数:
            data_dir: 拼接图像数据目录
            output_dir: 输出目录
            epochs: 训练轮次
            batch_size: 批次大小
            image_size: 图像尺寸
            learning_rate: 学习率
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 保存训练配置
        config_path = os.path.join(output_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'data_dir': data_dir,
                'epochs': epochs,
                'batch_size': batch_size,
                'image_size': image_size,
                'learning_rate': learning_rate,
                'training_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=4)

        # 创建数据集
        print(f"从 {data_dir} 创建数据集...")
        try:
            dataset = create_dataset_from_stitched_images(
                data_dir=data_dir,
                batch_size=batch_size,
                image_size=image_size
            )
        except Exception as e:
            print(f"创建数据集失败: {e}")
            return False

        # 初始化训练器
        print("初始化训练器...")
        trainer = UnsupervisedTrainer(
            model=self.refinement_network,
            learning_rate=learning_rate
        )

        # 执行训练
        print(f"开始训练，轮次: {epochs}, 批次大小: {batch_size}...")
        trainer.train(
            dataset=dataset,
            epochs=epochs,
            log_dir=output_dir
        )

        # 保存最终模型
        final_model_path = os.path.join(output_dir, 'final_model')
        self.refinement_network.save_weights(final_model_path + '.h5')
        print(f"保存最终模型到: {final_model_path}.h5")

        # 将最终模型复制到标准位置
        try:
            os.makedirs(os.path.dirname(REFINEMENT_MODEL_PATH), exist_ok=True)
            import shutil
            shutil.copy2(final_model_path + '.h5', REFINEMENT_MODEL_PATH + '.h5')
            print(f"复制最终模型到: {REFINEMENT_MODEL_PATH}.h5")
        except Exception as e:
            print(f"复制模型文件时出错: {e}")

        print("训练完成！")
        return True

    def test(self, model_path, test_dir, output_dir):
        """
        测试优化网络

        参数:
            model_path: 模型路径
            test_dir: 测试图像目录
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 加载模型
        print(f"加载模型: {model_path}")
        self.refinement_network.load_weights(model_path)

        # 查找测试图像
        test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"找到 {len(test_files)} 个测试图像")

        # 评估结果
        results = []

        # 处理测试图像
        for filename in tqdm(test_files, desc="测试优化效果"):
            img_path = os.path.join(test_dir, filename)

            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法读取图像: {img_path}")
                continue

            # 调整图像大小
            orig_h, orig_w = img.shape[:2]
            resized = img

            # 预处理
            img_norm = resized.astype(np.float32) / 255.0

            # 转换为RGB
            if len(img_norm.shape) == 2:
                img_norm = np.stack([img_norm] * 3, axis=2)
            elif img_norm.shape[2] == 1:
                img_norm = np.concatenate([img_norm] * 3, axis=2)

            # 创建输入张量
            img_tensor = tf.convert_to_tensor(img_norm)
            img_tensor = tf.expand_dims(img_tensor, 0)

            # 推理
            start_time = time.time()
            refined_tensor = self.refinement_network(img_tensor)
            inference_time = time.time() - start_time

            # 后处理
            refined = (refined_tensor[0].numpy() * 255.0).astype(np.uint8)

            # 调整回原始尺寸
            if refined.shape[:2] != (orig_h, orig_w):
                refined = cv2.resize(refined, (orig_w, orig_h))

            # 计算质量指标
            original_metrics = compute_metrics(img)
            refined_metrics = compute_metrics(refined)

            # 保存结果
            result_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
            os.makedirs(result_dir, exist_ok=True)

            cv2.imwrite(os.path.join(result_dir, 'input.png'), img)
            cv2.imwrite(os.path.join(result_dir, 'refined.png'), refined)

            # 生成差异图
            diff = cv2.absdiff(img, refined)
            diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(result_dir, 'diff.png'), diff_colored)

            # 生成蒙版可视化
            masks = self.mask_generator.generate_masks(tf.convert_to_tensor(img_norm))

            # 蒙版可视化
            plt.figure(figsize=(15, 10))

            plt.subplot(2, 3, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('原始拼接')
            plt.axis('off')

            plt.subplot(2, 3, 2)
            plt.imshow(cv2.cvtColor(refined, cv2.COLOR_BGR2RGB))
            plt.title('优化结果')
            plt.axis('off')

            plt.subplot(2, 3, 3)
            plt.imshow(cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB))
            plt.title('差异图')
            plt.axis('off')

            plt.subplot(2, 3, 4)
            plt.imshow(masks['valid'][..., 0], cmap='gray')
            plt.title('有效区域')
            plt.axis('off')

            plt.subplot(2, 3, 5)
            plt.imshow(masks['dilated_edge'][..., 0], cmap='hot')
            plt.title('拼接边界区域')
            plt.axis('off')

            plt.subplot(2, 3, 6)
            plt.imshow(masks['non_boundary'][..., 0], cmap='gray')
            plt.title('保留结构区域')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, 'visualization.png'))
            plt.close()

            # 记录指标
            result_info = {
                'filename': filename,
                'inference_time': inference_time,
                'original_metrics': original_metrics,
                'refined_metrics': refined_metrics
            }

            # 保存指标
            with open(os.path.join(result_dir, 'metrics.json'), 'w') as f:
                json.dump(
                    {
                        'filename': filename,
                        'inference_time': inference_time,
                        'original_metrics': {k: float(v) for k, v in original_metrics.items()},
                        'refined_metrics': {k: float(v) for k, v in refined_metrics.items()}
                    },
                    f, indent=4
                )

            results.append(result_info)

        # 生成总体报告
        if results:
            self._generate_test_report(results, output_dir)

        print(f"测试完成！结果保存在: {output_dir}")
        return True

    def _generate_test_report(self, results, output_dir):
        """生成测试报告"""
        report_path = os.path.join(output_dir, 'test_report.html')

        # 计算总体指标
        metrics_improvement = {}
        inference_times = []

        for result in results:
            inference_times.append(result['inference_time'])

            for metric in result['original_metrics']:
                if metric not in metrics_improvement:
                    metrics_improvement[metric] = []

                original_value = result['original_metrics'][metric]
                refined_value = result['refined_metrics'][metric]

                # 对于CE，越低越好；对于其他指标，越高越好
                if metric == 'ce':
                    improvement = (original_value - refined_value) / original_value * 100
                else:
                    improvement = (refined_value - original_value) / original_value * 100

                metrics_improvement[metric].append(improvement)

        # 创建性能图表
        plt.figure(figsize=(12, 6))

        # 创建箱线图
        plt.boxplot(list(metrics_improvement.values()), labels=list(metrics_improvement.keys()))
        plt.title('优化性能改进百分比')
        plt.ylabel('改进百分比 (%)')
        plt.grid(True, linestyle='--', alpha=0.7)

        # 添加均值线
        for i, metric in enumerate(metrics_improvement.keys()):
            mean_improvement = np.mean(metrics_improvement[metric])
            plt.plot([i + 1, i + 1], [mean_improvement, mean_improvement], 'r-', linewidth=2)
            plt.text(i + 1, mean_improvement, f'{mean_improvement:.1f}%',
                     horizontalalignment='center', verticalalignment='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'improvement_boxplot.png'))
        plt.close()

        # 创建推理时间直方图
        plt.figure(figsize=(10, 5))
        plt.hist(inference_times, bins=20, alpha=0.7, color='blue')
        plt.axvline(np.mean(inference_times), color='red', linestyle='dashed', linewidth=2)
        plt.text(np.mean(inference_times), plt.ylim()[1] * 0.9,
                 f'平均: {np.mean(inference_times) * 1000:.1f} ms',
                 color='red', horizontalalignment='center')
        plt.title('推理时间分布')
        plt.xlabel('时间 (秒)')
        plt.ylabel('频率')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'inference_time_histogram.png'))
        plt.close()

        # 创建HTML报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>无监督拼接优化网络测试报告</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    .summary {{ margin: 20px 0; }}
                    .chart {{ margin: 20px 0; text-align: center; }}
                    .chart img {{ max-width: 100%; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .improved {{ color: green; }}
                    .worsened {{ color: red; }}
                    .gallery {{ display: flex; flex-wrap: wrap; justify-content: space-around; }}
                    .example {{ margin: 10px; border: 1px solid #ddd; padding: 10px; max-width: 600px; }}
                    .example img {{ max-width: 100%; }}
                </style>
            </head>
            <body>
                <h1>无监督拼接优化网络测试报告</h1>
                <p>测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>测试样本数: {len(results)}</p>

                <h2>性能摘要</h2>
                <div class="summary">
                    <p>平均推理时间: {np.mean(inference_times) * 1000:.2f} ms</p>
                    <p>指标改进:</p>
                    <ul>
            """)

            for metric, improvements in metrics_improvement.items():
                mean_improvement = np.mean(improvements)
                color_class = "improved" if mean_improvement > 0 else "worsened"
                f.write(f"""
                        <li class="{color_class}">{metric.upper()}: {mean_improvement:.2f}% 
                        ({np.sum([imp > 0 for imp in improvements])}/{len(improvements)} 样本改进)</li>
                """)

            f.write("""
                    </ul>
                </div>

                <h2>性能图表</h2>
                <div class="chart">
                    <img src="improvement_boxplot.png" alt="性能改进箱线图">
                    <p>各指标改进百分比分布</p>
                </div>

                <div class="chart">
                    <img src="inference_time_histogram.png" alt="推理时间直方图">
                    <p>推理时间分布</p>
                </div>

                <h2>详细结果</h2>
                <table>
                    <tr>
                        <th>文件名</th>
                        <th>推理时间(ms)</th>
            """)

            # 添加指标列
            metrics = list(results[0]['original_metrics'].keys())
            for metric in metrics:
                f.write(f"<th>{metric.upper()} (原始)</th>")
                f.write(f"<th>{metric.upper()} (优化)</th>")
                f.write(f"<th>{metric.upper()} (改进%)</th>")

            f.write("</tr>")

            # 添加每个测试样本的结果
            for result in results:
                f.write(f"""
                    <tr>
                        <td>{result['filename']}</td>
                        <td>{result['inference_time'] * 1000:.2f}</td>
                """)

                for metric in metrics:
                    orig_value = result['original_metrics'][metric]
                    refined_value = result['refined_metrics'][metric]

                    # 计算改进百分比
                    if metric == 'ce':
                        improvement = (orig_value - refined_value) / orig_value * 100
                    else:
                        improvement = (refined_value - orig_value) / orig_value * 100

                    color_class = "improved" if improvement > 0 else "worsened"

                    f.write(f"<td>{orig_value:.4f}</td>")
                    f.write(f"<td>{refined_value:.4f}</td>")
                    f.write(f"<td class='{color_class}'>{improvement:.2f}%</td>")

                f.write("</tr>")

            f.write("""
                </table>

                <h2>样例展示</h2>
                <div class="gallery">
            """)

            # 选择最大改进的3个样本展示
            best_samples = []
            for result in results:
                # 使用UIQM或PSNR作为主要指标
                main_metric = 'uiqm' if 'uiqm' in result['original_metrics'] else 'psnr'

                if main_metric in result['original_metrics']:
                    orig_value = result['original_metrics'][main_metric]
                    refined_value = result['refined_metrics'][main_metric]

                    improvement = (refined_value - orig_value) / orig_value * 100
                    if main_metric == 'ce':  # 对于CE，越低越好
                        improvement = (orig_value - refined_value) / orig_value * 100

                    best_samples.append((result['filename'], improvement))

            # 排序并选择前3个
            best_samples.sort(key=lambda x: x[1], reverse=True)
            for filename, improvement in best_samples[:3]:
                sample_dir = os.path.join(output_dir, os.path.splitext(filename)[0])

                f.write(f"""
                    <div class="example">
                        <h3>{filename} (改进 {improvement:.2f}%)</h3>
                        <img src="{os.path.splitext(filename)[0]}/visualization.png" alt="{filename} 可视化">
                    </div>
                """)

            f.write("""
                </div>
            </body>
            </html>
            """)

        print(f"测试报告已生成: {report_path}")


def main():
    """主函数"""
    # 直接在代码中配置参数
    config = {
        'mode': 'train',  # 'train'=仅训练, 'test'=仅测试, 'both'=训练后测试
        'data_dir': "F:/UDWIS/FUnIE-GAN-master/data/v0_UWIS_dataset/train/GT",  # 拼接图像数据目录
        'test_dir': "F:/UDWIS/FUnIE-GAN-master/data/v0_UWIS_dataset/train/test",  # 测试图像目录
        'model_path': REFINEMENT_MODEL_PATH + '.h5',  # 模型路径(用于测试)
        'output_dir': os.path.join(RESULTS_DIR, f"refinement_{time.strftime('%Y%m%d_%H%M%S')}"),  # 输出目录
        'epochs': 20,  # 训练轮次
        'batch_size': 4,  # 批次大小
        'learning_rate': 0.0002  # 学习率
    }

    # 如果需要修改默认配置，直接在这里修改上面的值
    # 比如：config['mode'] = 'both'
    # 比如：config['epochs'] = 50

    # 初始化训练器
    trainer = RefinementTrainer(REFINEMENT_TRAIN_CONFIG)

    # 执行训练和/或测试
    if config['mode'] in ['train', 'both']:
        print("\n============ 训练无监督拼接优化网络 ============")
        train_output_dir = os.path.join(config['output_dir'], 'training')
        trainer.train(
            data_dir=config['data_dir'],
            output_dir=train_output_dir,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate']
        )

        # 更新模型路径为刚训练的模型
        config['model_path'] = os.path.join(train_output_dir, 'final_model.h5')

    if config['mode'] in ['test', 'both']:
        print("\n============ 测试无监督拼接优化网络 ============")
        test_output_dir = os.path.join(config['output_dir'], 'testing')

        # 确保模型文件存在
        if not os.path.exists(config['model_path']):
            print(f"错误: 模型文件不存在: {config['model_path']}")
            print("请先训练模型或提供正确的模型路径")
            return

        trainer.test(
            model_path=config['model_path'],
            test_dir=config['test_dir'],
            output_dir=test_output_dir
        )


if __name__ == "__main__":
    main()