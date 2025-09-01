"""
FUnIE-GAN增强器，用于水下图像增强
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json


class FUnIE_GAN_Enhancer:
    """FUnIE-GAN增强器，用于水下图像增强"""

    def __init__(self, model_path):
        """
        初始化FUnIE-GAN增强器

        参数:
            model_path: 模型路径（不包含后缀）
        """
        self.model = None
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        """加载FUnIE-GAN模型"""
        model_json_path = self.model_path + ".json"
        model_h5_path = self.model_path + ".h5"

        # 检查文件是否存在
        if not os.path.exists(model_json_path) or not os.path.exists(model_h5_path):
            print(f"模型文件不存在: {model_json_path} 或 {model_h5_path}")
            return

        try:
            # 加载模型架构
            with open(model_json_path, "r") as json_file:
                loaded_model_json = json_file.read()

            # 创建模型
            self.model = model_from_json(loaded_model_json)

            # 加载权重
            self.model.load_weights(model_h5_path)
            print(f"成功加载FUnIE-GAN模型: {model_json_path}")
        except Exception as e:
            print(f"加载FUnIE-GAN模型失败: {e}")
            self.model = None

    def enhance(self, image):
        """
        增强水下图像

        参数:
            image: 输入图像 (numpy数组，BGR格式)

        返回:
            增强后的图像
        """
        if self.model is None:
            print("模型未加载，返回原始图像")
            return image

        # 确保图像是正确格式
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)

        orig_shape = image.shape

        # 保存原始图像尺寸
        orig_h, orig_w = orig_shape[:2]

        # 调整图像尺寸为模型所需的 256x256
        resized_img = cv2.resize(image, (256, 256))

        # 预处理
        img = resized_img.astype(np.float32) / 127.5 - 1.0

        # 处理输入维度
        if len(resized_img.shape) == 2:  # 灰度图像
            img = np.stack([img, img, img], axis=2)

        # 添加批次维度
        img = np.expand_dims(img, axis=0)

        # 使用模型增强
        enhanced = self.model.predict(img)

        # 后处理
        enhanced = ((enhanced[0] + 1.0) * 127.5).astype(np.uint8)

        # 调整回原始尺寸
        enhanced = cv2.resize(enhanced, (orig_w, orig_h))

        # 确保输出与输入具有相同的通道数
        if len(orig_shape) == 2:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)

        return enhanced