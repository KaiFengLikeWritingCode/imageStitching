"""
FUnIE-GAN微调器，用于微调预训练的FUnIE-GAN模型
"""

import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import model_from_json, clone_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class SimpleFinetuner:
    """简化的FUnIE-GAN超保守微调器"""

    def __init__(self, model_path, lr=0.00001):  # 降低学习率以更保守
        """初始化微调器"""
        # 加载预训练模型
        self.original_model_path = model_path
        model_json_path = model_path + ".json"
        model_h5_path = model_path + ".h5"

        with open(model_json_path, "r") as json_file:
            loaded_model_json = json_file.read()

        # 加载基础模型
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(model_h5_path)

        # 创建参考模型(冻结权重)用于保守性约束
        self.reference_model = clone_model(self.model)
        self.reference_model.set_weights(self.model.get_weights())
        self.reference_model.trainable = False

        # 超保守微调 - 只微调最后极少数层
        fine_tune_layers = max(1, int(len(self.model.layers) * 0.02))  # 只微调最后2%的层或至少1层
        for layer in self.model.layers:
            layer.trainable = False
        for layer in self.model.layers[-fine_tune_layers:]:
            layer.trainable = True

        print(f"总层数: {len(self.model.layers)}, 可训练层数: {fine_tune_layers}")

        # 设置优化器和损失
        self.optimizer = Adam(learning_rate=lr)
        self.mse_loss = MeanSquaredError()

        # 编译模型
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.custom_loss,
            metrics=['mae']
        )

    def custom_loss(self, y_true, y_pred):
        """简化的自定义损失函数，保持高度保守性"""
        # 获取参考模型预测
        reference_pred = self.reference_model(tf.stop_gradient(y_true), training=False)

        # 1. 重建损失 - 使输出接近目标
        reconstruction_loss = self.mse_loss(y_true, y_pred)

        # 2. 保守性损失 - 不要偏离原始模型太远（高权重）
        conservatism_loss = self.mse_loss(y_pred, reference_pred)

        # 简化的总损失 - 强调保守性
        total_loss = reconstruction_loss * 0.1 + conservatism_loss * 0.9

        return total_loss

    def data_generator(self, input_dir, target_dir, batch_size=4, target_size=(256, 256)):
        """创建一个能处理不同尺寸图像的数据生成器"""
        # 获取文件列表
        input_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        target_files = sorted([f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

        # 找到共同的文件名
        common_files = list(set(input_files).intersection(set(target_files)))
        print(f"数据集包含 {len(common_files)} 对图像")

        indices = list(range(len(common_files)))
        steps_per_epoch = max(1, len(common_files) // batch_size)

        while True:
            np.random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:min(i + batch_size, len(indices))]
                batch_inputs = []
                batch_targets = []

                for idx in batch_indices:
                    filename = common_files[idx]

                    # 加载并处理输入图像
                    input_img = load_img(os.path.join(input_dir, filename), target_size=target_size)
                    input_array = img_to_array(input_img) / 255.0
                    batch_inputs.append(input_array)

                    # 加载并处理目标图像
                    target_img = load_img(os.path.join(target_dir, filename), target_size=target_size)
                    target_array = img_to_array(target_img) / 255.0
                    batch_targets.append(target_array)

                yield np.array(batch_inputs), np.array(batch_targets)

    def fine_tune(self, input_dir, target_dir, output_model_path,
                  batch_size=4, epochs=10, save_interval=5):
        """执行轻微微调"""
        print(f"使用现有数据集: 输入目录={input_dir}, 目标目录={target_dir}")

        # 创建数据生成器
        train_gen = self.data_generator(input_dir, target_dir, batch_size)

        # 获取steps_per_epoch
        input_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        target_files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        common_files = list(set(input_files).intersection(set(target_files)))
        steps_per_epoch = max(1, len(common_files) // batch_size)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

        # 单阶段超保守微调
        print(f"开始超保守微调过程...")

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            # 训练一个epoch
            history = self.model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=1,
                verbose=1
            )

            # 保存模型
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
                model_path = f"{output_model_path}_epoch{epoch + 1}"
                model_json = self.model.to_json()
                with open(f"{model_path}.json", "w") as json_file:
                    json_file.write(model_json)
                self.model.save_weights(f"{model_path}.h5")
                print(f"保存模型到 {model_path}")

        # 保存最终模型
        model_json = self.model.to_json()
        with open(f"{output_model_path}.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(f"{output_model_path}.h5")
        print(f"保存最终模型到 {output_model_path}")