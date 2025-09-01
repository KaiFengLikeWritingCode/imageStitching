"""
无监督拼接优化网络，用于改进拼接结果
"""

import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from pathlib import Path

# 设置TensorFlow内存增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class ResidualBlock(tf.keras.layers.Layer):
    """简单的残差块"""

    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters, 3, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, 3, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.add = layers.Add()
        self.relu = layers.Activation('relu')

    def call(self, inputs, training=None):
        shortcut = inputs
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.add([x, shortcut])
        x = self.relu(x)
        return x


class UnsupervisedRefinementNetwork(tf.keras.Model):
    """无监督拼接优化网络"""

    def __init__(self):
        super(UnsupervisedRefinementNetwork, self).__init__()

        # 轻量级编码器
        self.encoder = [
            layers.Conv2D(16, 5, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization()
        ]

        # 残差块
        self.residual_blocks = [
            ResidualBlock(32),
            ResidualBlock(32)
        ]

        # 轻量级解码器
        self.decoder = [
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(3, 3, padding='same', activation='tanh')  # tanh用于生成-1到1的修正
        ]

    def call(self, inputs, training=None):
        # 编码器
        x = inputs
        for layer in self.encoder:
            x = layer(x, training=training)

        # 残差块
        for block in self.residual_blocks:
            x = block(x, training=training)

        # 解码器
        for layer in self.decoder:
            x = layer(x, training=training)

        # 残差学习 - 限制修正幅度
        correction = x * 0.1  # 限制修正幅度为原始值的±10%
        refined = inputs + correction

        # 确保值在0-1范围内
        refined = tf.clip_by_value(refined, 0.0, 1.0)
        return refined


class StitchingMaskGenerator:
    """拼接蒙版生成器 - 用于识别拼接区域和边界"""

    def __init__(self, dilate_kernel_size=7):
        self.dilate_kernel_size = dilate_kernel_size

        # 边缘检测器
        self.edge_detector = tf.constant([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ], dtype=tf.float32)
        self.edge_detector = tf.reshape(self.edge_detector, [3, 3, 1, 1])

    def generate_masks(self, image):
        """
        生成拼接图像的各种蒙版

        参数:
            image: 拼接图像，形状为 [H, W, C] 或 [B, H, W, C]

        返回:
            包含各种蒙版的字典
        """
        # 确保输入是4D张量
        if len(tf.shape(image)) == 3:
            # 如果是3D张量 [H, W, C]，添加批次维度
            image = tf.expand_dims(image, axis=0)  # 变为 [1, H, W, C]

        # 转换为灰度
        if image.shape[-1] == 3:
            gray = tf.image.rgb_to_grayscale(image)
        else:
            gray = image

        # 创建二值蒙版(用于检测边缘)
        gray_mask = tf.cast(gray > 0.1, tf.float32)

        # 定义有效区域掩码
        valid_mask = gray_mask

        # 检测边缘 - 确保gray_mask是4D
        edge_mask = tf.nn.conv2d(gray_mask, self.edge_detector,
                                 strides=[1, 1, 1, 1], padding='SAME')
        edge_mask = tf.cast(tf.abs(edge_mask) > 0.1, tf.float32)
        edge_mask = tf.abs(edge_mask)
        edge_mask = tf.clip_by_value(edge_mask, 0, 1)

        # 扩展边缘区域
        dilated_edge = tf.nn.max_pool2d(
            edge_mask,
            ksize=self.dilate_kernel_size,
            strides=1,
            padding='SAME'
        )

        # 非边界区域
        non_boundary = valid_mask * (1.0 - dilated_edge)

        # 构建masks字典 - 添加'valid'键以保持兼容性
        masks = {
            'gray_mask': gray_mask,
            'valid_mask': valid_mask,
            'valid': valid_mask,  # 添加这一行确保兼容性
            'edge_mask': edge_mask,
            'dilated_edge': dilated_edge,
            'non_boundary': non_boundary
        }

        # 如果原始输入是3D，移除批次维度以保持一致
        if len(tf.shape(image)) == 4 and image.shape[0] == 1:
            # 移除所有蒙版的批次维度
            masks = {k: v[0] for k, v in masks.items()}

        return masks


def unsupervised_loss(stitched_image, refined_image, mask_generator):
    """无监督拼接优化损失函数"""
    # 生成蒙版
    masks = mask_generator.generate_masks(stitched_image)
    valid_mask = masks['valid']
    dilated_edge = masks['dilated_edge']
    non_boundary = masks['non_boundary']

    # 损失组件字典
    losses = {}

    # 1. 边界平滑度损失
    sobel_refined = tf.image.sobel_edges(refined_image)
    gradient_mag = tf.sqrt(tf.square(sobel_refined[..., 0]) +
                           tf.square(sobel_refined[..., 1]) + 1e-6)

    boundary_smoothness = tf.reduce_mean(dilated_edge * gradient_mag)
    losses['boundary_smoothness'] = boundary_smoothness

    # 2. 结构保持损失
    structure_preservation = tf.reduce_mean(
        non_boundary * tf.abs(refined_image - stitched_image))
    losses['structure_preservation'] = structure_preservation

    # 3. 颜色一致性损失
    avg_pool = tf.nn.avg_pool2d(refined_image, ksize=15, strides=1, padding='SAME')
    color_consistency = tf.reduce_mean(
        dilated_edge * tf.abs(refined_image - avg_pool))
    losses['color_consistency'] = color_consistency

    # 4. 总体变化限制
    overall_change = tf.reduce_mean(valid_mask * tf.abs(refined_image - stitched_image))
    losses['overall_change'] = overall_change

    # 5. 亮度均衡损失 - 特别适用于水下图像
    luminance_stitched = tf.reduce_mean(stitched_image, axis=-1, keepdims=True)
    luminance_refined = tf.reduce_mean(refined_image, axis=-1, keepdims=True)

    # 计算局部区域亮度方差
    local_var_stitched = tf.nn.avg_pool2d(
        tf.square(luminance_stitched - tf.nn.avg_pool2d(luminance_stitched, ksize=7, strides=1, padding='SAME')),
        ksize=7, strides=1, padding='SAME'
    )

    local_var_refined = tf.nn.avg_pool2d(
        tf.square(luminance_refined - tf.nn.avg_pool2d(luminance_refined, ksize=7, strides=1, padding='SAME')),
        ksize=7, strides=1, padding='SAME'
    )

    # 优化后的局部亮度方差应接近或小于原始方差
    brightness_loss = tf.reduce_mean(dilated_edge *
                                     tf.maximum(0.0, local_var_refined - local_var_stitched))
    losses['brightness_balance'] = brightness_loss

    # 组合所有损失
    total_loss = (
            2.0 * boundary_smoothness +
            1.0 * structure_preservation +
            2.0 * color_consistency +
            0.5 * overall_change +
            1.0 * brightness_loss
    )

    losses['total'] = total_loss
    return total_loss, losses


class UnsupervisedTrainer:
    """无监督拼接优化网络训练器"""

    def __init__(self, model, learning_rate=0.0002):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.mask_generator = StitchingMaskGenerator(dilate_kernel_size=7)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.loss_components = {}

    def reset_metrics(self):
        """重置指标"""
        self.train_loss.reset_states()
        self.loss_components = {}

    @tf.function
    def train_step(self, stitched_images):
        """单步训练"""
        with tf.GradientTape() as tape:
            # 前向传播
            refined_images = self.model(stitched_images, training=True)

            # 计算损失
            loss, loss_components = unsupervised_loss(
                stitched_images, refined_images, self.mask_generator)

        # 反向传播
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # 更新指标
        self.train_loss(loss)

        return loss, loss_components

    def train(self, dataset, epochs=10, log_dir=None):
        """训练模型"""
        # 设置日志目录
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_file = open(os.path.join(log_dir, 'training_log.txt'), 'w')

            # 创建可视化输出目录
            vis_dir = os.path.join(log_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
        else:
            log_file = None

        # 创建模型保存目录
        if log_dir:
            checkpoint_dir = os.path.join(log_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)

        # 训练循环
        for epoch in range(epochs):
            # 重置指标
            self.reset_metrics()

            # 训练一个epoch
            for batch_idx, stitched_batch in enumerate(dataset):
                loss, loss_components = self.train_step(stitched_batch)

                # 日志
                if batch_idx % 10 == 0:
                    log_msg = f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss:.6f}"
                    print(log_msg)

                    if log_file:
                        log_file.write(log_msg + '\n')
                        for name, value in loss_components.items():
                            log_file.write(f"  - {name}: {value.numpy():.6f}\n")
                        log_file.flush()

                # 可视化一些样本 - 移除对len(dataset)的依赖
                if log_dir and batch_idx % 50 == 0:
                    self._visualize_results(
                        stitched_batch, epoch, batch_idx, vis_dir)

            # Epoch结束，保存模型
            if log_dir:
                self.model.save_weights(os.path.join(checkpoint_dir, f'model_epoch{epoch + 1}'))

            # 打印epoch摘要
            epoch_loss = self.train_loss.result()
            epoch_summary = f"Epoch {epoch + 1}/{epochs}, Average Loss: {epoch_loss:.6f}"
            print(epoch_summary)

            if log_file:
                log_file.write(epoch_summary + '\n\n')
                log_file.flush()

        # 关闭日志文件
        if log_file:
            log_file.close()

        print("训练完成！")

    def _visualize_results(self, stitched_batch, refined_batch, batch_idx, log_dir, epoch=0):
        """可视化结果"""
        # 确保输入是正确的格式
        # 检查输入类型，并处理不同情况
        if isinstance(refined_batch, (int, float)):
            # 如果refined_batch是标量值(如损失值)，创建一个占位符图像
            sample_stitched = stitched_batch[0]  # 获取批次中第一个样本
            sample_refined = sample_stitched  # 使用输入图像作为占位符
            is_scalar = True
        else:
            # 正常处理张量输入
            sample_stitched = stitched_batch[0]  # 获取批次中第一个样本
            sample_refined = refined_batch[0]
            is_scalar = False

        # 如果样本没有批次维度但需要它，添加一个
        if len(tf.shape(sample_stitched)) == 3:
            sample_stitched = tf.expand_dims(sample_stitched, 0)

        # 生成蒙版
        masks = self.mask_generator.generate_masks(sample_stitched)

        # 创建可视化目录
        vis_dir = os.path.join(log_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        # 绘制结果
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # 拼接输入
        stitched = stitched_batch[0].numpy()
        axes[0, 0].imshow(stitched)
        axes[0, 0].set_title('拼接输入')
        axes[0, 0].axis('off')

        # 优化输出或显示损失值
        if is_scalar:
            # 如果refined_batch是标量，显示损失值
            axes[0, 1].text(0.5, 0.5, f'Loss: {refined_batch:.6f}',
                            horizontalalignment='center', verticalalignment='center',
                            transform=axes[0, 1].transAxes, fontsize=12)
            axes[0, 1].set_title('当前损失值')
            axes[0, 1].axis('off')

            # 没有优化输出，所以差异也无法显示
            axes[0, 2].text(0.5, 0.5, 'No difference data available',
                            horizontalalignment='center', verticalalignment='center',
                            transform=axes[0, 2].transAxes, fontsize=10)
            axes[0, 2].set_title('差异')
            axes[0, 2].axis('off')
        else:
            # 正常处理优化输出
            refined = refined_batch[0].numpy()
            axes[0, 1].imshow(refined)
            axes[0, 1].set_title('优化输出')
            axes[0, 1].axis('off')

            # 差异
            diff = np.abs(refined - stitched) * 5  # 放大差异便于可视化
            axes[0, 2].imshow(diff)
            axes[0, 2].set_title('差异 (x5)')
            axes[0, 2].axis('off')

        # 有效区域蒙版
        axes[1, 0].imshow(masks['valid'][..., 0].numpy(), cmap='gray')
        axes[1, 0].set_title('有效区域')
        axes[1, 0].axis('off')

        # 边界蒙版
        axes[1, 1].imshow(masks['dilated_edge'][..., 0].numpy(), cmap='hot')
        axes[1, 1].set_title('拼接边界区域')
        axes[1, 1].axis('off')

        # 非边界蒙版
        axes[1, 2].imshow(masks['non_boundary'][..., 0].numpy(), cmap='gray')
        axes[1, 2].set_title('保留结构区域')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'epoch{epoch + 1}_batch{batch_idx}.png'))
        plt.close(fig)


def create_dataset_from_stitched_images(data_dir, batch_size=4, image_size=(256, 256)):
    """从拼接图像创建数据集"""
    # 查找所有拼接图像
    stitched_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        stitched_files.extend(list(Path(data_dir).glob(ext)))

    print(f"找到 {len(stitched_files)} 个拼接图像文件")

    if len(stitched_files) == 0:
        raise ValueError(f"在 {data_dir} 中没有找到图像文件")

    # 创建数据集生成器
    def generator():
        for file_path in stitched_files:
            # 读取图像
            img = cv2.imread(str(file_path))
            if img is None:
                print(f"无法读取 {file_path}，跳过")
                continue

            # 转BGR到RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 归一化到0-1
            img = img.astype(np.float32) / 255.0

            # 调整大小
            if image_size:
                img = cv2.resize(img, image_size)

            yield img

    # 创建TensorFlow数据集
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=tf.TensorSpec(shape=(image_size[1], image_size[0], 3), dtype=tf.float32)
    )

    # 批处理和预取
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset