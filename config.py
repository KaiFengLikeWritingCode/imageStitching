"""
配置文件，包含所有系统参数
"""

import os

# 路径配置
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 数据目录
INPUT_IMAGES_DIR = os.path.join(DATA_DIR, "input")
ENHANCED_IMAGES_DIR = os.path.join(DATA_DIR, "enhanced")
STITCHED_IMAGES_DIR = os.path.join(DATA_DIR, "stitched")
GT_IMAGES_DIR = os.path.join(DATA_DIR, "ground_truth")  # 可选的真值参考

# 结果目录
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, "visualizations")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

# 模型路径
FUNIEGAN_MODEL_PATH = os.path.join(MODELS_DIR, "F:/UDWIS/FUnIE-GAN-master/TF-Keras/models/gen_p/model_15320_")
FUNIEGAN_FINETUNED_PATH = os.path.join(MODELS_DIR, "F:/UDWIS/models/finetuned2/underwater_funie_gan_conservative2_epoch5")
REFINEMENT_MODEL_PATH = os.path.join(MODELS_DIR, "refinement/refinement_model")

# 创建必要的目录
for dir_path in [
    INPUT_IMAGES_DIR, ENHANCED_IMAGES_DIR, STITCHED_IMAGES_DIR, GT_IMAGES_DIR,
    VISUALIZATION_DIR, METRICS_DIR, LOGS_DIR,
    os.path.dirname(FUNIEGAN_MODEL_PATH),
    os.path.dirname(FUNIEGAN_FINETUNED_PATH),
    os.path.dirname(REFINEMENT_MODEL_PATH)
]:
    os.makedirs(dir_path, exist_ok=True)

# FUnIE-GAN微调配置
FINETUNE_CONFIG = {
    "input_dir": os.path.join(DATA_DIR, "finetune/input"),
    "target_dir": os.path.join(DATA_DIR, "finetune/target"),
    "batch_size": 4,
    "epochs": 10,
    "learning_rate": 0.00001,
    "save_interval": 2
}

# 拼接优化网络训练配置
REFINEMENT_TRAIN_CONFIG = {
    "batch_size": 4,
    "epochs": 20,
    "learning_rate": 0.0002,
    "image_size": (256, 256)
}

# 评估指标配置
METRICS_CONFIG = {
    "compute_psnr": True,
    "compute_ssim": True,
    "compute_ce": True,
    "compute_uiqm": True
}

# 可视化配置
VISUALIZATION_CONFIG = {
    "save_intermediate_results": True,
    "show_feature_matches": True,
    "show_ddm_selection": True
}

# 系统处理配置
PROCESSING_CONFIG = {
    "use_enhancement": True,
    "use_ddm": True,
    "use_refinement": True,
    "target_image_size": (640, 480),  # 处理前调整图像大小
    "confidence_threshold": 0.5,      # 特征匹配置信度阈值
    "ransac_threshold": 3.0           # RANSAC阈值
}