"""
特征匹配模块，包含LoFTR匹配器和动态匹配器选择器
"""

import cv2
import numpy as np
import torch
import kornia
import os


class LoFTRMatcher:
    """LoFTR特征匹配器"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """初始化LoFTR匹配器"""
        self.device = device

        # 加载LoFTR模型
        try:
            self.loftr = kornia.feature.LoFTR(pretrained='outdoor')
            self.loftr = self.loftr.to(device).eval()
            print(f"LoFTR模型已加载到{device}")
        except Exception as e:
            print(f"加载LoFTR模型失败: {e}")
            self.loftr = None

    def match_images(self, img1, img2):
        """
        匹配两幅图像，增强错误处理

        参数:
            img1, img2: 输入图像 (numpy数组，BGR格式)

        返回:
            字典包含匹配点、置信度等
        """
        if self.loftr is None:
            print("LoFTR模型未加载，直接使用SIFT")
            return self._match_with_sift(img1, img2)

        # 检查输入图像是否有效
        if img1 is None or img2 is None or img1.size == 0 or img2.size == 0:
            print("无效的输入图像")
            return {'mkpts0': np.array([]), 'mkpts1': np.array([]), 'mconf': np.array([])}

        # 确保图像是uint8类型
        if img1.dtype != np.uint8:
            img1 = np.clip(img1 * 255 if img1.max() <= 1.0 else img1, 0, 255).astype(np.uint8)
        if img2.dtype != np.uint8:
            img2 = np.clip(img2 * 255 if img2.max() <= 1.0 else img2, 0, 255).astype(np.uint8)

        # 转换为灰度图
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1.copy()

        if len(img2.shape) == 3:
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img2_gray = img2.copy()

        # 确保图像不为空
        if img1_gray.shape[0] == 0 or img1_gray.shape[1] == 0 or img2_gray.shape[0] == 0 or img2_gray.shape[1] == 0:
            print("图像尺寸为零")
            return {'mkpts0': np.array([]), 'mkpts1': np.array([]), 'mconf': np.array([])}

        # 确保图像尺寸能被8整除（LoFTR要求）
        h1, w1 = img1_gray.shape
        h2, w2 = img2_gray.shape

        # 添加最小尺寸检查
        if h1 < 32 or w1 < 32 or h2 < 32 or w2 < 32:
            print("图像尺寸太小，无法进行特征匹配")
            return {'mkpts0': np.array([]), 'mkpts1': np.array([]), 'mconf': np.array([])}

        # 调整为能被8整除的尺寸
        new_h1, new_w1 = (h1 // 8) * 8, (w1 // 8) * 8
        new_h2, new_w2 = (h2 // 8) * 8, (w2 // 8) * 8

        # 确保尺寸不为零
        if new_h1 == 0 or new_w1 == 0 or new_h2 == 0 or new_w2 == 0:
            print("调整后的图像尺寸为零")
            return {'mkpts0': np.array([]), 'mkpts1': np.array([]), 'mconf': np.array([])}

        img1_gray = cv2.resize(img1_gray, (new_w1, new_h1))
        img2_gray = cv2.resize(img2_gray, (new_w2, new_h2))

        # 转换为PyTorch张量
        try:
            img1_tensor = torch.from_numpy(img1_gray).float()[None][None].to(self.device) / 255.
            img2_tensor = torch.from_numpy(img2_gray).float()[None][None].to(self.device) / 255.
            batch = {'image0': img1_tensor, 'image1': img2_tensor}

            # 使用LoFTR进行匹配
            with torch.no_grad():
                self.loftr(batch)

                # 检查'keypoints0'是否在batch中
                if 'keypoints0' in batch and 'keypoints1' in batch:
                    mkpts0 = batch['keypoints0'].cpu().numpy()
                    mkpts1 = batch['keypoints1'].cpu().numpy()
                    mconf = batch.get('confidence', torch.ones(len(mkpts0))).cpu().numpy()

                    return {
                        'mkpts0': mkpts0,
                        'mkpts1': mkpts1,
                        'mconf': mconf
                    }
                else:
                    print("LoFTR没有生成匹配点，尝试SIFT")
                    return self._match_with_sift(img1_gray, img2_gray)
        except Exception as e:
            print(f"LoFTR匹配失败: {e}")
            # 回退到SIFT
            return self._match_with_sift(img1_gray, img2_gray)

    def _match_with_sift(self, img1_gray, img2_gray):
        """使用SIFT进行特征匹配（备选方法）"""
        try:
            # 确保输入图像是uint8类型
            if img1_gray.dtype != np.uint8:
                img1_gray = np.clip(img1_gray * 255 if img1_gray.max() <= 1.0 else img1_gray, 0, 255).astype(np.uint8)
            if img2_gray.dtype != np.uint8:
                img2_gray = np.clip(img2_gray * 255 if img2_gray.max() <= 1.0 else img2_gray, 0, 255).astype(np.uint8)

            # 创建SIFT检测器
            sift = cv2.SIFT_create()

            # 检测关键点和描述符
            kp1, des1 = sift.detectAndCompute(img1_gray, None)
            kp2, des2 = sift.detectAndCompute(img2_gray, None)

            if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                # 特征匹配
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)

                good_matches = []
                for m_n in matches:
                    if len(m_n) == 2:  # 确保有两个匹配点
                        m, n = m_n
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)

                if len(good_matches) > 0:
                    # 提取匹配点坐标
                    mkpts0 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                    mkpts1 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
                    mconf = np.ones(len(mkpts0))

                    print(f"SIFT找到 {len(good_matches)} 个匹配点")
                    return {
                        'mkpts0': mkpts0,
                        'mkpts1': mkpts1,
                        'mconf': mconf
                    }
        except Exception as e:
            print(f"SIFT匹配失败: {e}")

        # 如果仍然失败，返回空结果
        print("所有特征匹配方法都失败，返回空结果")
        return {
            'mkpts0': np.array([]),
            'mkpts1': np.array([]),
            'mconf': np.array([])
        }


class DynamicMatcherSelector:
    """动态特征匹配器选择器，智能选择LoFTR或SIFT"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """初始化动态匹配器选择器"""
        self.loftr_matcher = LoFTRMatcher(device=device)
        self.min_match_count = 10
        self.edge_density_weight = 0.7
        self.contrast_weight = 0.3

    def select_and_match(self, img1, img2):
        """
        分析图像特性并选择最佳匹配算法

        参数:
            img1, img2: 输入图像 (numpy数组，BGR格式)

        返回:
            matches: 匹配结果
            matcher_used: 使用的匹配器名称
        """
        # 分析图像特征
        img1_features = self._analyze_image(img1)
        img2_features = self._analyze_image(img2)

        # 计算匹配器兼容性得分
        loftr_score = self._compute_loftr_compatibility(img1_features, img2_features)
        sift_score = self._compute_sift_compatibility(img1_features, img2_features)

        # 选择得分更高的匹配器，失败时自动切换
        if loftr_score >= sift_score:
            matches = self.loftr_matcher.match_images(img1, img2)
            matcher_used = "loftr"

            if matches is None or len(matches['mkpts0']) < self.min_match_count:
                print("LoFTR匹配不足，切换到SIFT")
                matches = self.loftr_matcher._match_with_sift(img1, img2)
                matcher_used = "sift"
        else:
            matches = self.loftr_matcher._match_with_sift(img1, img2)
            matcher_used = "sift"

            if matches is None or len(matches['mkpts0']) < self.min_match_count:
                print("SIFT匹配不足，切换到LoFTR")
                matches = self.loftr_matcher.match_images(img1, img2)
                matcher_used = "loftr"

        return matches, matcher_used

    def _analyze_image(self, img):
        """分析图像特征以评估适合的匹配算法"""
        # 确保输入是uint8格式
        if img.dtype != np.uint8:
            img = np.clip(img * 255 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)

        # 转换为灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # 计算边缘密度
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)

        # 计算对比度
        contrast = np.std(gray.astype(float)) / (np.mean(gray.astype(float)) + 1e-6)

        # 计算纹理复杂度
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        texture_complexity = np.std(gradient_mag)

        # 评估亮度分布
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / (hist.sum() + 1e-6)
        entropy = -np.sum(hist_norm * np.log(hist_norm + 1e-6))

        return {
            'edge_density': edge_density,
            'contrast': contrast,
            'texture_complexity': texture_complexity,
            'entropy': entropy
        }

    def _compute_loftr_compatibility(self, img1_features, img2_features):
        """计算LoFTR兼容性得分"""
        # LoFTR在高纹理复杂度和中等对比度的场景表现更好
        score1 = 0.5 * img1_features['texture_complexity'] + 0.3 * img1_features['entropy'] + 0.2 * img1_features[
            'contrast']
        score2 = 0.5 * img2_features['texture_complexity'] + 0.3 * img2_features['entropy'] + 0.2 * img2_features[
            'contrast']

        # 水下场景可能对比度低，特别考虑
        if img1_features['contrast'] < 0.15 or img2_features['contrast'] < 0.15:
            return (score1 + score2) * 0.7  # 对低对比度场景惩罚LoFTR评分

        return (score1 + score2) / 2.0

    def _compute_sift_compatibility(self, img1_features, img2_features):
        """计算SIFT兼容性得分"""
        # SIFT在边缘显著和高对比度的场景表现更好
        score1 = 0.5 * img1_features['edge_density'] + 0.4 * img1_features['contrast'] + 0.1 * img1_features['entropy']
        score2 = 0.5 * img2_features['edge_density'] + 0.4 * img2_features['contrast'] + 0.1 * img2_features['entropy']

        return (score1 + score2) / 2.0