"""
动态决策模块，选择使用原始图像还是增强图像
"""

import cv2
import numpy as np


class DynamicDecisionModule:
    """动态抉择模块 (DDM) - 自动选择最佳输入组合"""

    def __init__(self, lambda_weight=0.7, threshold=0.1):
        """
        初始化动态抉择模块

        参数:
            lambda_weight: 特征匹配质量与增强质量之间的权重平衡
            threshold: 最小切换阈值，防止微小差异导致不必要的切换
        """
        self.lambda_weight = lambda_weight
        self.threshold = threshold

    def evaluate_matchability(self, img1, img2, matcher):
        """
        评估两幅图像之间的可匹配性

        参数:
            img1, img2: 输入图像
            matcher: 特征匹配器

        返回:
            匹配质量分数 (基于匹配点数量和置信度)
        """
        matches, _ = matcher.select_and_match(img1, img2)

        if matches is None or 'mkpts0' not in matches or len(matches['mkpts0']) == 0:
            return 0.0

        # 计算匹配质量分数
        match_count = len(matches['mkpts0'])
        avg_conf = np.mean(matches['mconf']) if len(matches['mconf']) > 0 else 0

        # 综合匹配点数量和平均置信度
        matchability = min(1.0, match_count / 100.0) * 0.7 + avg_conf * 0.3

        return matchability

    def evaluate_color_quality(self, img):
        """
        评估图像的颜色质量

        参数:
            img: 输入图像

        返回:
            颜色质量分数
        """
        # 确保输入是uint8格式
        if img.dtype != np.uint8:
            img = np.clip(img * 255 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)

        # 转换为HSV空间进行颜色评估
        if len(img.shape) == 3 and img.shape[2] == 3:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # 计算饱和度和亮度的均值和标准差
            s_mean, s_std = np.mean(hsv[:, :, 1]), np.std(hsv[:, :, 1])
            v_mean, v_std = np.mean(hsv[:, :, 2]), np.std(hsv[:, :, 2])

            # 计算对比度 (使用亮度通道)
            contrast = v_std / (v_mean + 1e-6)

            # 计算颜色均衡性 (饱和度分布)
            color_balance = s_mean * (1 - abs(0.5 - s_mean) * 2)

            # 组合为最终的颜色质量分数
            color_quality = 0.6 * contrast + 0.4 * color_balance

            return min(1.0, color_quality)
        else:
            return 0.5  # 灰度图像或无效图像返回中等分数

    def decide(self, orig_img1, orig_img2, enhanced_img1, enhanced_img2, matcher):
        """
        决定使用原始图像对还是增强图像对

        参数:
            orig_img1, orig_img2: 原始图像对
            enhanced_img1, enhanced_img2: 增强图像对
            matcher: 特征匹配器

        返回:
            选择的图像对和决策信息
        """
        # 确保输入图像有效
        if orig_img1 is None or orig_img2 is None:
            return (enhanced_img1, enhanced_img2), {"message": "使用增强图像 (原始图像无效)"}

        if enhanced_img1 is None or enhanced_img2 is None:
            return (orig_img1, orig_img2), {"message": "使用原始图像 (增强图像无效)"}

        # 评估原始图像对
        orig_matchability = self.evaluate_matchability(orig_img1, orig_img2, matcher)
        orig_color_qual1 = self.evaluate_color_quality(orig_img1)
        orig_color_qual2 = self.evaluate_color_quality(orig_img2)
        orig_color_qual = (orig_color_qual1 + orig_color_qual2) / 2

        # 评估增强图像对
        enh_matchability = self.evaluate_matchability(enhanced_img1, enhanced_img2, matcher)
        enh_color_qual1 = self.evaluate_color_quality(enhanced_img1)
        enh_color_qual2 = self.evaluate_color_quality(enhanced_img2)
        enh_color_qual = (enh_color_qual1 + enh_color_qual2) / 2

        # 计算总体质量分数
        orig_quality = self.lambda_weight * orig_matchability + (1 - self.lambda_weight) * orig_color_qual
        enh_quality = self.lambda_weight * enh_matchability + (1 - self.lambda_weight) * enh_color_qual

        # 记录详细评估结果
        details = {
            "original": {
                "matchability": orig_matchability,
                "color_quality": orig_color_qual,
                "total_score": orig_quality
            },
            "enhanced": {
                "matchability": enh_matchability,
                "color_quality": enh_color_qual,
                "total_score": enh_quality
            }
        }

        # 决策逻辑: 只有当增强质量明显优于原始质量时才选择增强图像
        if enh_quality > orig_quality + self.threshold:
            return (enhanced_img1, enhanced_img2), details
        else:
            return (orig_img1, orig_img2), details