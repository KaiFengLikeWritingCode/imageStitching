"""
RANSAC拼接器，处理单应性估计和图像拼接
"""

import cv2
import numpy as np


class RansacStitcher:
    """RANSAC拼接器，处理单应性估计和图像拼接"""

    def __init__(self, confidence_threshold=0.5):
        """
        初始化RANSAC拼接器

        参数:
            confidence_threshold: 匹配点置信度阈值
        """
        self.confidence_threshold = confidence_threshold

    def estimate_homography(self, matches):
        """
        使用RANSAC估计单应性矩阵

        参数:
            matches: 包含mkpts0, mkpts1, mconf的字典

        返回:
            H: 单应性矩阵，如果失败则为None
        """
        if matches is None:
            return None

        mkpts0 = matches['mkpts0']
        mkpts1 = matches['mkpts1']
        mconf = matches['mconf']

        # 筛选高置信度匹配点
        if len(mconf) > 0:
            mask = mconf > self.confidence_threshold
            mkpts0 = mkpts0[mask]
            mkpts1 = mkpts1[mask]

        # 确保有足够的匹配点
        if len(mkpts0) >= 4:
            try:
                H, _ = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 3.0)
                return H
            except Exception as e:
                print(f"单应性估计失败: {e}")

        return None

    def warp_and_blend(self, img1, img2, H):
        """
        使用单应性矩阵拼接图像

        参数:
            img1, img2: 输入图像
            H: 单应性矩阵

        返回:
            result: 拼接结果，失败则返回None
        """
        if H is None:
            return None

        # 获取图像尺寸
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # 计算变换后的四个角点
        corners1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32).reshape(-1, 1, 2)
        corners1_transformed = cv2.perspectiveTransform(corners1, H)

        # 计算全局边界
        corners2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32).reshape(-1, 1, 2)
        all_corners = np.vstack((corners1_transformed, corners2))

        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        # 平移量
        trans_x, trans_y = -xmin, -ymin

        # 构建平移矩阵
        H_translation = np.array([
            [1, 0, trans_x],
            [0, 1, trans_y],
            [0, 0, 1]
        ])

        # 结合单应性变换和平移
        H_final = H_translation.dot(H)

        # 创建输出画布
        output_w, output_h = xmax - xmin, ymax - ymin
        result = cv2.warpPerspective(img1, H_final, (output_w, output_h))

        # 在结果图像中放置第二幅图像
        result[trans_y:trans_y + h2, trans_x:trans_x + w2] = img2

        return result