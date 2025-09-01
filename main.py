"""
水下图像拼接系统主入口
"""
import os
# 设置环境变量解决OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import os
import sys
import argparse
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import json
from tqdm import tqdm

# 导入配置
from config import *

# 导入组件
from components.feature_matching import DynamicMatcherSelector
from components.dynamic_decision import DynamicDecisionModule
from components.ransac_stitcher import RansacStitcher
from components.funiegan_enhancer import FUnIE_GAN_Enhancer
from components.funiegan_finetuner import SimpleFinetuner
from components.unsupervised_refinement import (
    UnsupervisedRefinementNetwork, UnsupervisedTrainer, create_dataset_from_stitched_images
)

# 导入工具函数
from utils.metrics import compute_metrics
from utils.visualization import (
    visualize_ddm_decision, visualize_feature_matching,
    visualize_stitching_progress, visualize_metrics
)


class UnderwaterStitchingSystem:
    """水下图像拼接系统"""

    def __init__(self, config):
        """
        初始化系统

        参数:
            config: 配置字典
        """
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
        if os.path.exists(REFINEMENT_MODEL_PATH + ".h5"):
            try:
                self.refinement_network.load_weights(REFINEMENT_MODEL_PATH + ".h5")
                print(f"成功加载拼接优化网络权重: {REFINEMENT_MODEL_PATH}")
            except Exception as e:
                print(f"加载拼接优化网络权重失败: {e}")

    def finetune_enhancer(self):
        """微调FUnIE-GAN增强器"""
        print("\n============ 微调FUnIE-GAN增强器 ============")

        # 检查数据集
        if not os.path.exists(FINETUNE_CONFIG['input_dir']) or not os.path.exists(FINETUNE_CONFIG['target_dir']):
            print(f"微调数据集不存在: {FINETUNE_CONFIG['input_dir']} 或 {FINETUNE_CONFIG['target_dir']}")
            return False

        # 初始化微调器
        print("初始化FUnIE-GAN微调器...")
        finetuner = SimpleFinetuner(
            model_path=FUNIEGAN_MODEL_PATH,
            lr=FINETUNE_CONFIG['learning_rate']
        )

        # 执行微调
        print("开始FUnIE-GAN微调...")
        finetuner.fine_tune(
            input_dir=FINETUNE_CONFIG['input_dir'],
            target_dir=FINETUNE_CONFIG['target_dir'],
            output_model_path=FUNIEGAN_FINETUNED_PATH,
            batch_size=FINETUNE_CONFIG['batch_size'],
            epochs=FINETUNE_CONFIG['epochs'],
            save_interval=FINETUNE_CONFIG['save_interval']
        )

        # 更新增强器使用微调后的模型
        print("重新加载微调后的增强器...")
        self.enhancer = FUnIE_GAN_Enhancer(FUNIEGAN_FINETUNED_PATH)

        print("FUnIE-GAN微调完成！")
        return True

    def train_refinement_network(self):
        """训练拼接优化网络"""
        print("\n============ 训练拼接优化网络 ============")

        # 检查拼接图像数据集
        if not os.path.exists(STITCHED_IMAGES_DIR):
            print(f"拼接图像数据集不存在: {STITCHED_IMAGES_DIR}")
            return False

        # 创建数据集
        try:
            print("创建拼接图像数据集...")
            dataset = create_dataset_from_stitched_images(
                STITCHED_IMAGES_DIR,
                batch_size=REFINEMENT_TRAIN_CONFIG['batch_size'],
                image_size=REFINEMENT_TRAIN_CONFIG['image_size']
            )
        except Exception as e:
            print(f"创建数据集失败: {e}")
            return False

        # 初始化训练器
        print("初始化拼接优化网络训练器...")
        trainer = UnsupervisedTrainer(
            model=self.refinement_network,
            learning_rate=REFINEMENT_TRAIN_CONFIG['learning_rate']
        )

        # 设置日志目录
        log_dir = os.path.join(LOGS_DIR, f"refinement_training_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(log_dir, exist_ok=True)

        # 执行训练
        print("开始拼接优化网络训练...")
        trainer.train(
            dataset=dataset,
            epochs=REFINEMENT_TRAIN_CONFIG['epochs'],
            log_dir=log_dir
        )

        # 保存最终模型
        self.refinement_network.save_weights(REFINEMENT_MODEL_PATH + ".h5")
        print(f"保存拼接优化网络权重到 {REFINEMENT_MODEL_PATH}")

        print("拼接优化网络训练完成！")
        return True

    def process_image_pair(self, img1_path, img2_path, output_dir=None, gt_path=None):
        """
        处理单对图像

        参数:
            img1_path: 第一幅图像路径
            img2_path: 第二幅图像路径
            output_dir: 输出目录
            gt_path: 真值图像路径 (可选)

        返回:
            处理结果字典
        """
        # 创建输出目录
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 加载图像
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            print(f"加载图像失败: {img1_path} 或 {img2_path}")
            return None

        # 加载真值图像 (如果存在)
        gt_img = None
        if gt_path and os.path.exists(gt_path):
            gt_img = cv2.imread(gt_path)

        # 调整图像大小 (如果配置了目标尺寸)
        if PROCESSING_CONFIG['target_image_size']:
            img1 = cv2.resize(img1, PROCESSING_CONFIG['target_image_size'])
            img2 = cv2.resize(img2, PROCESSING_CONFIG['target_image_size'])

        # 结果字典
        result = {
            'input': {'img1': img1, 'img2': img2}
        }
        metrics_results = {}

        # 1. 图像增强
        if PROCESSING_CONFIG['use_enhancement'] and self.enhancer.model is not None:
            print("增强图像...")
            enhanced_img1 = self.enhancer.enhance(img1)
            enhanced_img2 = self.enhancer.enhance(img2)

            result['enhanced'] = {'img1': enhanced_img1, 'img2': enhanced_img2}

            # 评估增强质量
            metrics_results['orig_img1'] = compute_metrics(img1, gt_img)
            metrics_results['orig_img2'] = compute_metrics(img2, gt_img)
            metrics_results['enh_img1'] = compute_metrics(enhanced_img1, gt_img)
            metrics_results['enh_img2'] = compute_metrics(enhanced_img2, gt_img)

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
            print("DDM选择最佳输入...")
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
        print("执行特征匹配...")
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
        print("估计单应性矩阵与拼接...")
        H = self.stitcher.estimate_homography(matches)

        if H is not None:
            stitched_img = self.stitcher.warp_and_blend(selected_img1, selected_img2, H)
            result['stitched'] = {'img': stitched_img, 'H': H}

            # 计算拼接质量
            metrics_results['stitched'] = compute_metrics(stitched_img, gt_img)

            # 保存拼接结果
            if output_dir:
                cv2.imwrite(os.path.join(output_dir, 'stitched.png'), stitched_img)

            # 5. 拼接优化
            if PROCESSING_CONFIG['use_refinement'] and self.refinement_network is not None:
                print("优化拼接结果...")

                # 预处理
                stitched_norm = stitched_img.astype(np.float32) / 255.0
                stitched_tensor = tf.convert_to_tensor(stitched_norm)
                stitched_tensor = tf.expand_dims(stitched_tensor, 0)

                # 推理
                refined_tensor = self.refinement_network(stitched_tensor)
                refined_img = (refined_tensor[0].numpy() * 255.0).astype(np.uint8)

                result['refined'] = {'img': refined_img}

                # 计算优化后质量
                metrics_results['refined'] = compute_metrics(refined_img, gt_img)

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
            print("单应性估计失败，无法拼接图像")
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

    def batch_process(self, input_dir, output_base_dir):
        """
        批量处理图像对

        参数:
            input_dir: 输入目录，包含图像对
            output_base_dir: 输出基目录
        """
        print("\n============ 批量处理图像对 ============")

        # 创建输出目录
        os.makedirs(output_base_dir, exist_ok=True)

        # 查找图像对
        image_files = sorted([f for f in os.listdir(input_dir)
                              if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

        # 按照前缀分组
        image_pairs = {}
        for filename in image_files:
            # 假设文件名格式为 "prefix_A.jpg" 和 "prefix_B.jpg"
            parts = filename.split('_')
            if len(parts) >= 2:
                prefix = '_'.join(parts[:-1])
                suffix = parts[-1].split('.')[0]

                if suffix in ['A', 'B']:
                    if prefix not in image_pairs:
                        image_pairs[prefix] = {}
                    image_pairs[prefix][suffix] = filename

        # 处理每对图像
        print(f"找到 {len(image_pairs)} 个图像对")

        success_count = 0
        for prefix, pair in tqdm(image_pairs.items(), desc="处理图像对"):
            if 'A' in pair and 'B' in pair:
                img1_path = os.path.join(input_dir, pair['A'])
                img2_path = os.path.join(input_dir, pair['B'])

                # 检查真值图像是否存在
                gt_path = os.path.join(GT_IMAGES_DIR, f"{prefix}_GT.jpg")
                if not os.path.exists(gt_path):
                    gt_path = os.path.join(GT_IMAGES_DIR, f"{prefix}_GT.png")
                    if not os.path.exists(gt_path):
                        gt_path = None

                # 创建输出目录
                pair_output_dir = os.path.join(output_base_dir, prefix)

                # 处理图像对
                result = self.process_image_pair(img1_path, img2_path, pair_output_dir, gt_path)

                if result and 'error' not in result:
                    success_count += 1

        print(f"批量处理完成！成功拼接 {success_count}/{len(image_pairs)} 对图像")

    def interactive_demo(self):
        """交互式演示"""
        print("\n============ 交互式水下图像拼接演示 ============")
        print("请选择操作:")
        print("1. 选择两张图像进行拼接")
        print("2. 退出")

        choice = input("请输入选项 (1-2): ")

        if choice == '1':
            # 请求用户输入图像路径
            img1_path = input("请输入第一张图像路径: ")
            img2_path = input("请输入第二张图像路径: ")

            if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                print("图像文件不存在!")
                return

            # 创建输出目录
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(RESULTS_DIR, f"interactive_demo_{timestamp}")

            # 处理图像对
            print("开始处理图像...")
            result = self.process_image_pair(img1_path, img2_path, output_dir)

            if result and 'error' not in result:
                print(f"处理完成！结果保存在: {output_dir}")

                # 如果在Linux/MacOS上，尝试打开结果文件夹
                if sys.platform == "linux" or sys.platform == "darwin":
                    os.system(f"xdg-open {output_dir} 2>/dev/null || open {output_dir}")
                # 如果在Windows上
                elif sys.platform == "win32":
                    os.system(f"explorer {output_dir}")
            else:
                print("处理失败！")

        elif choice == '2':
            return
        else:
            print("无效选项!")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="水下图像拼接系统")
    parser.add_argument('--mode', choices=['finetune', 'train_refinement', 'process', 'batch', 'interactive', 'all'],
                        default='interactive', help="运行模式")
    parser.add_argument('--img1', help="第一幅输入图像路径")
    parser.add_argument('--img2', help="第二幅输入图像路径")
    parser.add_argument('--input_dir', help="批量处理的输入目录")
    parser.add_argument('--output_dir', help="输出目录")

    args = parser.parse_args()

    # 初始化系统
    system = UnderwaterStitchingSystem(PROCESSING_CONFIG)

    # 根据模式执行相应操作
    if args.mode == 'finetune':
        system.finetune_enhancer()

    elif args.mode == 'train_refinement':
        system.train_refinement_network()

    elif args.mode == 'process':
        if not args.img1 or not args.img2:
            print("请提供两个输入图像路径 (--img1 和 --img2)")
            return

        output_dir = args.output_dir or os.path.join(RESULTS_DIR, f"single_process_{time.strftime('%Y%m%d_%H%M%S')}")
        system.process_image_pair(args.img1, args.img2, output_dir)

    elif args.mode == 'batch':
        if not args.input_dir:
            print("请提供批量处理的输入目录 (--input_dir)")
            return

        output_dir = args.output_dir or os.path.join(RESULTS_DIR, f"batch_process_{time.strftime('%Y%m%d_%H%M%S')}")
        system.batch_process(args.input_dir, output_dir)

    elif args.mode == 'interactive':
        system.interactive_demo()

    elif args.mode == 'all':
        # 执行全部流程
        system.finetune_enhancer()
        system.train_refinement_network()

        if args.input_dir:
            output_dir = args.output_dir or os.path.join(RESULTS_DIR, f"batch_process_{time.strftime('%Y%m%d_%H%M%S')}")
            system.batch_process(args.input_dir, output_dir)
        else:
            system.interactive_demo()


if __name__ == "__main__":
    main()