"""
批量文件夹测试脚本 - 处理两个目录中的配对图像
"""
import os
# 设置环境变量解决OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import json
from tqdm import tqdm
import pandas as pd
import shutil

# 导入配置
from config import *

# 导入组件
from components.feature_matching import DynamicMatcherSelector
from components.dynamic_decision import DynamicDecisionModule
from components.ransac_stitcher import RansacStitcher
from components.funiegan_enhancer import FUnIE_GAN_Enhancer
from components.unsupervised_refinement import UnsupervisedRefinementNetwork

# 导入工具函数
from utils.metrics import compute_metrics
from utils.visualization import (
    visualize_ddm_decision, visualize_feature_matching,
    visualize_stitching_progress, visualize_metrics
)


class BatchTester:
    """批量图像对测试器"""

    def __init__(self, config):
        """初始化批量测试器"""
        self.config = config
        self.setup_components()

    def setup_components(self):
        """设置组件"""
        # 加载增强器
        print("加载FUnIE-GAN增强器...")
        self.enhancer = FUnIE_GAN_Enhancer(FUNIEGAN_MODEL_PATH)

        # 初始化匹配器
        print("初始化动态匹配器选择器...")
        self.matcher = DynamicMatcherSelector()

        # 初始化DDM
        print("初始化动态决策模块...")
        self.ddm = DynamicDecisionModule(lambda_weight=0.7, threshold=0.1)

        # 初始化拼接器
        print("初始化RANSAC拼接器...")
        self.stitcher = RansacStitcher(
            confidence_threshold=PROCESSING_CONFIG['confidence_threshold']
        )

        # 加载优化网络
        print("加载拼接优化网络...")
        self.refinement_network = UnsupervisedRefinementNetwork()

        # 如果存在优化网络权重，则加载
        refinement_weights = REFINEMENT_MODEL_PATH + ".h5"
        if os.path.exists(refinement_weights):
            try:
                self.refinement_network.load_weights(refinement_weights)
                print(f"成功加载拼接优化网络权重: {refinement_weights}")
            except Exception as e:
                print(f"加载拼接优化网络权重失败: {e}")
                self.refinement_network = None
        else:
            print(f"优化网络权重文件不存在: {refinement_weights}")
            self.refinement_network = None

    def process_image_pair(self, img1, img2, output_dir=None, pair_name=None):
        """
        处理单对图像

        参数:
            img1, img2: 输入图像
            output_dir: 输出目录
            pair_name: 图像对名称

        返回:
            处理结果字典
        """
        # 创建输出目录
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 调整图像大小 (如果配置了目标尺寸)
        if PROCESSING_CONFIG['target_image_size']:
            img1 = cv2.resize(img1, PROCESSING_CONFIG['target_image_size'])
            img2 = cv2.resize(img2, PROCESSING_CONFIG['target_image_size'])

        # 结果字典
        result = {
            'input': {'img1': img1, 'img2': img2},
            'name': pair_name
        }
        metrics_results = {}

        # 1. 图像增强
        if PROCESSING_CONFIG['use_enhancement'] and self.enhancer.model is not None:
            print(f"增强图像: {pair_name}")
            enhanced_img1 = self.enhancer.enhance(img1)
            enhanced_img2 = self.enhancer.enhance(img2)

            result['enhanced'] = {'img1': enhanced_img1, 'img2': enhanced_img2}

            # 评估增强质量
            metrics_results['orig_img1'] = compute_metrics(img1)
            metrics_results['orig_img2'] = compute_metrics(img2)
            metrics_results['enh_img1'] = compute_metrics(enhanced_img1)
            metrics_results['enh_img2'] = compute_metrics(enhanced_img2)

            # 可视化增强效果
            if output_dir and VISUALIZATION_CONFIG['save_intermediate_results']:
                cv2.imwrite(os.path.join(output_dir, 'enhanced_img1.png'), enhanced_img1)
                cv2.imwrite(os.path.join(output_dir, 'enhanced_img2.png'), enhanced_img2)

                # 并排显示原始和增强图像
                comparison1 = np.hstack((img1, enhanced_img1))
                comparison2 = np.hstack((img2, enhanced_img2))
                cv2.imwrite(os.path.join(output_dir, 'enhancement_comparison1.png'), comparison1)
                cv2.imwrite(os.path.join(output_dir, 'enhancement_comparison2.png'), comparison2)
        else:
            enhanced_img1 = None
            enhanced_img2 = None

        # 2. 动态决策选择输入图像
        if PROCESSING_CONFIG['use_ddm'] and enhanced_img1 is not None and enhanced_img2 is not None:
            print(f"DDM选择最佳输入: {pair_name}")
            (selected_img1, selected_img2), ddm_details = self.ddm.decide(
                img1, img2, enhanced_img1, enhanced_img2, self.matcher
            )

            # 记录选择
            choice = "enhanced" if selected_img1 is enhanced_img1 else "original"
            result['ddm_choice'] = choice
            result['ddm_details'] = ddm_details

            # 可视化DDM决策
            if output_dir and VISUALIZATION_CONFIG['show_ddm_selection']:
                ddm_vis_path = os.path.join(output_dir, 'ddm_decision.png')
                visualize_ddm_decision(
                    [img1, img2],
                    [enhanced_img1, enhanced_img2],
                    ddm_details,
                    ddm_vis_path
                )
        else:
            # 如果不使用DDM，基于配置选择输入
            if PROCESSING_CONFIG['use_enhancement'] and enhanced_img1 is not None and enhanced_img2 is not None:
                selected_img1, selected_img2 = enhanced_img1, enhanced_img2
                result['ddm_choice'] = "enhanced"
            else:
                selected_img1, selected_img2 = img1, img2
                result['ddm_choice'] = "original"

        result['selected'] = {'img1': selected_img1, 'img2': selected_img2}

        # 3. 特征匹配
        print(f"执行特征匹配: {pair_name}")
        matches, matcher_used = self.matcher.select_and_match(selected_img1, selected_img2)
        result['matching'] = {'matches': matches, 'matcher_used': matcher_used}

        # 可视化匹配结果
        if output_dir and VISUALIZATION_CONFIG['show_feature_matches']:
            match_vis_path = os.path.join(output_dir, 'feature_matches.png')
            visualize_feature_matching(
                selected_img1, selected_img2,
                matches, matcher_used,
                match_vis_path
            )

        # 4. 拼接
        print(f"估计单应性矩阵与拼接: {pair_name}")
        H = self.stitcher.estimate_homography(matches)

        if H is not None:
            stitched_img = self.stitcher.warp_and_blend(selected_img1, selected_img2, H)
            result['stitched'] = {'img': stitched_img, 'H': H}

            # 计算拼接质量
            metrics_results['stitched'] = compute_metrics(stitched_img)

            # 保存拼接结果
            if output_dir:
                cv2.imwrite(os.path.join(output_dir, 'stitched.png'), stitched_img)

                # 保存拼接结果用于训练集合
                stitched_dataset_dir = os.path.join(STITCHED_IMAGES_DIR)
                os.makedirs(stitched_dataset_dir, exist_ok=True)
                if pair_name:
                    cv2.imwrite(os.path.join(stitched_dataset_dir, f'{pair_name}_stitched.png'), stitched_img)

            # 5. 拼接优化
            if PROCESSING_CONFIG['use_refinement'] and self.refinement_network is not None:
                print(f"优化拼接结果: {pair_name}")

                # 预处理
                stitched_norm = stitched_img.astype(np.float32) / 255.0
                stitched_tensor = tf.convert_to_tensor(stitched_norm)
                stitched_tensor = tf.expand_dims(stitched_tensor, 0)

                # 推理
                refined_tensor = self.refinement_network(stitched_tensor)
                refined_img = (refined_tensor[0].numpy() * 255.0).astype(np.uint8)

                result['refined'] = {'img': refined_img}

                # 计算优化后质量
                metrics_results['refined'] = compute_metrics(refined_img)

                # 保存优化结果
                if output_dir:
                    cv2.imwrite(os.path.join(output_dir, 'refined.png'), refined_img)

                # 可视化拼接与优化效果
                if output_dir:
                    vis_path = os.path.join(output_dir, 'stitching_refinement.png')
                    visualize_stitching_progress(
                        selected_img1, selected_img2,
                        stitched_img, refined_img,
                        vis_path
                    )
            else:
                # 只显示拼接结果
                if output_dir:
                    vis_path = os.path.join(output_dir, 'stitching.png')
                    visualize_stitching_progress(
                        selected_img1, selected_img2,
                        stitched_img, None,
                        vis_path
                    )
        else:
            print(f"单应性估计失败: {pair_name}")
            result['error'] = "单应性估计失败"

        # 6. 可视化评估指标
        if metrics_results and output_dir:
            metrics_vis_path = os.path.join(output_dir, 'metrics.png')
            visualize_metrics(metrics_results, metrics_vis_path)

            # 保存指标数据
            metrics_json_path = os.path.join(output_dir, 'metrics.json')
            with open(metrics_json_path, 'w') as f:
                # 将numpy类型转换为Python基本类型
                serializable_metrics = {}
                for stage, metrics in metrics_results.items():
                    serializable_metrics[stage] = {
                        k: float(v) for k, v in metrics.items()
                    }
                json.dump(serializable_metrics, f, indent=4)

        result['metrics'] = metrics_results
        return result

    def batch_process_folders(self, folder_a, folder_b, output_dir, name_pattern=None):
        """
        批量处理两个文件夹中的图像

        参数:
            folder_a: 第一个文件夹路径（左图像）
            folder_b: 第二个文件夹路径（右图像）
            output_dir: 输出目录
            name_pattern: 文件名匹配模式，如果为None则使用相同文件名
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        results_summary_path = os.path.join(output_dir, 'results_summary.csv')

        # 获取文件列表
        files_a = sorted([f for f in os.listdir(folder_a) if f.endswith(('.jpg', '.png', '.jpeg'))])
        files_b = sorted([f for f in os.listdir(folder_b) if f.endswith(('.jpg', '.png', '.jpeg'))])

        # 匹配文件对
        if name_pattern:
            # 使用提供的模式匹配文件
            image_pairs = []
            for file_a in files_a:
                # 从模式中提取共享部分
                pair_id = self._extract_pair_id(file_a, name_pattern[0])
                if not pair_id:
                    continue

                # 搜索匹配的B文件
                for file_b in files_b:
                    if self._extract_pair_id(file_b, name_pattern[1]) == pair_id:
                        image_pairs.append((file_a, file_b, pair_id))
                        break
        else:
            # 使用相同文件名匹配
            common_files = set(files_a).intersection(set(files_b))
            image_pairs = [(f, f, os.path.splitext(f)[0]) for f in common_files]

        print(f"找到 {len(image_pairs)} 对匹配图像")

        # 处理结果汇总
        summary_data = []

        # 批量处理匹配的图像对
        success_count = 0
        for idx, (file_a, file_b, pair_id) in enumerate(tqdm(image_pairs, desc="处理图像对")):
            try:
                # 加载图像
                img_a_path = os.path.join(folder_a, file_a)
                img_b_path = os.path.join(folder_b, file_b)

                img_a = cv2.imread(img_a_path)
                img_b = cv2.imread(img_b_path)

                if img_a is None or img_b is None:
                    print(f"无法加载图像: {img_a_path} 或 {img_b_path}")
                    continue

                # 创建输出目录
                pair_output_dir = os.path.join(output_dir, f"pair_{pair_id}")

                # 处理图像对
                result = self.process_image_pair(img_a, img_b, pair_output_dir, pair_id)

                if result and 'error' not in result:
                    success_count += 1

                    # 收集指标数据
                    row_data = {
                        'pair_id': pair_id,
                        'status': 'success',
                        'matcher_used': result.get('matching', {}).get('matcher_used', ''),
                        'ddm_choice': result.get('ddm_choice', ''),
                        'feature_count': len(result.get('matching', {}).get('matches', {}).get('mkpts0', []))
                    }

                    # 添加各阶段的指标
                    for stage, metrics in result.get('metrics', {}).items():
                        for metric_name, metric_value in metrics.items():
                            row_data[f'{stage}_{metric_name}'] = float(metric_value)

                    summary_data.append(row_data)
                else:
                    # 记录失败
                    summary_data.append({
                        'pair_id': pair_id,
                        'status': 'failed',
                        'error': result.get('error', '未知错误') if result else '处理失败'
                    })
            except Exception as e:
                print(f"处理图像对 {pair_id} 时出错: {e}")
                summary_data.append({
                    'pair_id': pair_id,
                    'status': 'error',
                    'error': str(e)
                })

        # 保存结果摘要
        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(results_summary_path, index=False)

            # 创建汇总可视化
            self._create_summary_visualization(df, output_dir)

        print(f"批量处理完成！成功拼接 {success_count}/{len(image_pairs)} 对图像")
        print(f"结果保存在: {output_dir}")
        print(f"摘要文件: {results_summary_path}")

        return success_count, len(image_pairs)

    def _extract_pair_id(self, filename, pattern):
        """从文件名中提取配对ID"""
        import re
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
        return None

    def _create_summary_visualization(self, df, output_dir):
        """创建结果汇总可视化"""
        # 确保DataFrame不为空且有成功的结果
        if df.empty or 'status' not in df.columns or not any(df['status'] == 'success'):
            return

        # 创建结果目录
        summary_dir = os.path.join(output_dir, 'summary')
        os.makedirs(summary_dir, exist_ok=True)

        # 1. 成功率饼图
        plt.figure(figsize=(8, 8))
        status_counts = df['status'].value_counts()
        plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%')
        plt.title('处理成功率')
        plt.savefig(os.path.join(summary_dir, 'success_rate.png'))
        plt.close()

        # 2. 匹配器使用情况
        if 'matcher_used' in df.columns:
            plt.figure(figsize=(8, 8))
            matcher_counts = df['matcher_used'].value_counts()
            plt.pie(matcher_counts, labels=matcher_counts.index, autopct='%1.1f%%')
            plt.title('匹配器使用情况')
            plt.savefig(os.path.join(summary_dir, 'matcher_usage.png'))
            plt.close()

        # 3. DDM选择情况
        if 'ddm_choice' in df.columns:
            plt.figure(figsize=(8, 8))
            ddm_counts = df['ddm_choice'].value_counts()
            plt.pie(ddm_counts, labels=ddm_counts.index, autopct='%1.1f%%')
            plt.title('DDM选择情况')
            plt.savefig(os.path.join(summary_dir, 'ddm_choices.png'))
            plt.close()

        # 4. 整体指标箱线图
        success_df = df[df['status'] == 'success']

        # 找出所有指标列
        metric_columns = [col for col in success_df.columns if
                          any(col.startswith(prefix) for prefix in
                              ['orig_img', 'enh_img', 'stitched_', 'refined_'])]

        if metric_columns:
            for metric in ['psnr', 'ssim', 'ce', 'uiqm']:
                metric_cols = [col for col in metric_columns if col.endswith(metric)]
                if metric_cols:
                    plt.figure(figsize=(10, 6))
                    success_df[metric_cols].boxplot()
                    plt.title(f'{metric.upper()} 各阶段对比')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(os.path.join(summary_dir, f'{metric}_comparison.png'))
                    plt.close()

        # 5. 创建HTML摘要报告
        try:
            self._create_html_report(df, output_dir, summary_dir)
        except Exception as e:
            print(f"创建HTML报告时出错: {e}")

    def _create_html_report(self, df, output_dir, summary_dir):
        """创建HTML摘要报告"""
        success_df = df[df['status'] == 'success'].copy()

        # 如果没有成功记录，退出
        if success_df.empty:
            return

        html_path = os.path.join(output_dir, 'report.html')

        # 创建报告标题和样式
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>水下图像拼接批量处理报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #2c3e50; }
                .summary { margin: 20px 0; }
                .summary img { max-width: 500px; display: inline-block; margin: 10px; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .success { color: green; }
                .failed { color: red; }
                .gallery { display: flex; flex-wrap: wrap; gap: 10px; }
                .gallery-item { margin-bottom: 20px; border: 1px solid #ddd; padding: 10px; width: 300px; }
                .gallery-item img { width: 100%; height: auto; }
                .metrics { font-size: 12px; }
            </style>
        </head>
        <body>
            <h1>水下图像拼接批量处理报告</h1>
            <p>处理时间: """ + time.strftime('%Y-%m-%d %H:%M:%S') + """</p>

            <h2>处理摘要</h2>
            <p>总图像对数: """ + str(len(df)) + """</p>
            <p>成功拼接: """ + str(len(success_df)) + """ (""" + f"{len(success_df) / len(df) * 100:.1f}%" + """)</p>

            <div class="summary">
                <h3>摘要图表</h3>
        """

        # 添加摘要图表
        summary_images = [f for f in os.listdir(summary_dir) if f.endswith('.png')]
        for img in summary_images:
            img_rel_path = os.path.join('summary', img)
            html_content += f'<img src="{img_rel_path}" alt="{img}" title="{img}">\n'

        html_content += """
            </div>

            <h2>处理详情</h2>
            <table>
                <tr>
                    <th>序号</th>
                    <th>图像对ID</th>
                    <th>状态</th>
                    <th>匹配器</th>
                    <th>DDM选择</th>
                    <th>特征点数</th>
                    <th>原始UIQM</th>
                    <th>增强UIQM</th>
                    <th>拼接PSNR</th>
                    <th>优化PSNR</th>
                </tr>
        """

        # 添加每行数据
        for i, (_, row) in enumerate(df.iterrows()):
            html_content += f"""
                <tr>
                    <td>{i + 1}</td>
                    <td>{row.get('pair_id', '')}</td>
                    <td class="{'success' if row.get('status') == 'success' else 'failed'}">{row.get('status', '')}</td>
                    <td>{row.get('matcher_used', '')}</td>
                    <td>{row.get('ddm_choice', '')}</td>
                    <td>{row.get('feature_count', '')}</td>
                    <td>{row.get('orig_img1_uiqm', ''):.2f if not pd.isna(row.get('orig_img1_uiqm', '')) else ''}</td>
                    <td>{row.get('enh_img1_uiqm', ''):.2f if not pd.isna(row.get('enh_img1_uiqm', '')) else ''}</td>
                    <td>{row.get('stitched_psnr', ''):.2f if not pd.isna(row.get('stitched_psnr', '')) else ''}</td>
                    <td>{row.get('refined_psnr', ''):.2f if not pd.isna(row.get('refined_psnr', '')) else ''}</td>
                </tr>
            """

        html_content += """
            </table>

            <h2>成功样例</h2>
            <div class="gallery">
        """

        # 添加一些成功样例的图片预览
        top_samples = success_df.head(10)
        for _, row in top_samples.iterrows():
            pair_dir = os.path.join(output_dir, f"pair_{row['pair_id']}")
            stitched_path = os.path.join(pair_dir, 'stitched.png')
            refined_path = os.path.join(pair_dir, 'refined.png')

            if os.path.exists(stitched_path):
                stitched_rel_path = f"pair_{row['pair_id']}/stitched.png"

                # 使用优化结果或拼接结果
                result_path = refined_path if os.path.exists(refined_path) else stitched_path
                result_rel_path = f"pair_{row['pair_id']}/{'refined' if os.path.exists(refined_path) else 'stitched'}.png"

                html_content += f"""
                    <div class="gallery-item">
                        <h4>图像对 {row['pair_id']}</h4>
                        <img src="{result_rel_path}" alt="拼接结果">
                        <div class="metrics">
                            <p>匹配器: {row.get('matcher_used', '')}, 特征点: {row.get('feature_count', '')}</p>
                            <p>DDM选择: {row.get('ddm_choice', '')}</p>
                            <p>UIQM: {row.get('refined_uiqm', row.get('stitched_uiqm', '')):.2f if not pd.isna(row.get('refined_uiqm', row.get('stitched_uiqm', ''))) else ''}</p>
                        </div>
                    </div>
                """

        html_content += """
            </div>
        </body>
        </html>
        """

        # 写入HTML文件
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"HTML报告已生成: {html_path}")


# 移除 argparse 导入
# import argparse  # 已不需要

def main():
    """主函数"""
    # 直接设置路径和参数（在这里修改为您的实际路径）
    folder_a = "F:/UDWIS/FUnIE-GAN-master/data/v0_UWIS_dataset/test/imageA"  # 第一个文件夹(左图像)
    folder_b = "F:/UDWIS/FUnIE-GAN-master/data/v0_UWIS_dataset/test/imageB"  # 第二个文件夹(右图像)
    output_dir = os.path.join(RESULTS_DIR, f"batch_process_{time.strftime('%Y%m%d_%H%M%S')}")  # 输出目录

    # 可选：设置文件名匹配模式
    # 如果A、B文件夹中的文件名不完全匹配，但有共同部分，可以设置这个
    pattern_a = None  # 例如 "(.*)_A" 表示提取"_A"前面的部分作为共享ID
    pattern_b = None  # 例如 "(.*)_B" 表示提取"_B"前面的部分作为共享ID
    pattern_a = "(.*)_A"
    pattern_b = "(.*)_B"

    # 设置匹配模式
    if pattern_a and pattern_b:
        name_pattern = (pattern_a, pattern_b)
    else:
        name_pattern = None

    # 初始化批量测试器
    tester = BatchTester(PROCESSING_CONFIG)

    # 执行批量处理
    tester.batch_process_folders(
        folder_a,
        folder_b,
        output_dir,
        name_pattern
    )


if __name__ == "__main__":
    main()