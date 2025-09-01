"""
评估指标计算模块，包括PSNR, SSIM, CE, UIQM等
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage


def compute_psnr(img1, img2):
    """
    计算PSNR (峰值信噪比)

    参数:
        img1, img2: 要比较的图像 (numpy数组，取值范围0-255)

    返回:
        PSNR值 (dB)
    """
    # 确保图像类型为uint8
    if img1.dtype != np.uint8:
        img1 = np.clip(img1 * 255.0 if img1.max() <= 1.0 else img1, 0, 255).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = np.clip(img2 * 255.0 if img2.max() <= 1.0 else img2, 0, 255).astype(np.uint8)

    # 确保尺寸一致
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 计算MSE
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')

    # 计算PSNR
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr


def compute_ssim(img1, img2):
    """
    计算SSIM (结构相似性)

    参数:
        img1, img2: 要比较的图像 (numpy数组，取值范围0-255)

    返回:
        SSIM值 (0-1)
    """
    # 确保图像类型为uint8
    if img1.dtype != np.uint8:
        img1 = np.clip(img1 * 255.0 if img1.max() <= 1.0 else img1, 0, 255).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = np.clip(img2 * 255.0 if img2.max() <= 1.0 else img2, 0, 255).astype(np.uint8)

    # 确保尺寸一致
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 转换为灰度图
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1

    if len(img2.shape) == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2

    # 计算SSIM
    ssim_value = ssim(img1_gray, img2_gray, data_range=255)
    return ssim_value


def compute_ce(img):
    """
    计算CE (对比度熵)，水下图像特有的质量指标

    参数:
        img: 输入图像 (numpy数组，取值范围0-255)

    返回:
        CE值 (越低越好)
    """
    # 确保图像类型为uint8
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)

    # 转换为灰度图
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    # 计算梯度
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度幅值
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # 将梯度幅值归一化到[0,1]
    if np.max(gradient_magnitude) > 0:
        gradient_magnitude = gradient_magnitude / np.max(gradient_magnitude)

    # 计算直方图
    hist, _ = np.histogram(gradient_magnitude, bins=256, range=(0, 1))
    hist = hist / np.sum(hist)

    # 计算熵
    entropy = -np.sum(hist * np.log2(hist + 1e-10))

    return entropy


def compute_uiqm(img):
    """
    计算UIQM (水下图像质量评价指标)

    参数:
        img: 输入图像 (numpy数组，取值范围0-255)

    返回:
        UIQM值 (越高越好)
    """
    # 确保图像类型为uint8
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)

    # 参数
    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753

    # 转换为LAB颜色空间
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # 计算UICM (颜色丰富度)
    l, a, b = cv2.split(img_lab)
    a_mean, a_var = np.mean(a), np.var(a)
    b_mean, b_var = np.mean(b), np.var(b)
    a_kurt = np.mean(((a - a_mean) / np.sqrt(a_var + 1e-10)) ** 4) - 3
    b_kurt = np.mean(((b - b_mean) / np.sqrt(b_var + 1e-10)) ** 4) - 3
    uicm = -0.0268 * np.sqrt(a_var + b_var) + 0.1586 * np.sqrt(np.abs(a_kurt) + np.abs(b_kurt))

    # 计算UISM (锐度)
    sobel_operator = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    def compute_contrast(band):
        gx = ndimage.convolve(band.astype(np.float64), sobel_operator)
        gy = ndimage.convolve(band.astype(np.float64), sobel_operator.T)
        return np.sqrt(gx ** 2 + gy ** 2 + 1e-10)

    l_contrast = compute_contrast(l)
    a_contrast = compute_contrast(a)
    b_contrast = compute_contrast(b)

    l_contrast_mean = np.mean(l_contrast)
    a_contrast_mean = np.mean(a_contrast)
    b_contrast_mean = np.mean(b_contrast)

    uism = 0.299 * l_contrast_mean + 0.587 * a_contrast_mean + 0.114 * b_contrast_mean

    # 计算UIConM (对比度)
    l_norm = l / 255.0
    l_mean = np.mean(l_norm)
    uiconm = np.log(np.abs(np.sum(l_norm - l_mean) ** 2) / (l.size * (1 - l_mean) ** 2 + 1e-10))

    # 计算UIQM
    uiqm = c1 * uicm + c2 * uism + c3 * uiconm

    return uiqm


def compute_metrics(img, reference_img=None):
    """
    计算所有指标

    参数:
        img: 输入图像
        reference_img: 参考图像 (如果提供则计算PSNR和SSIM)

    返回:
        包含所有指标的字典
    """
    metrics = {}

    # 计算无参考指标
    metrics['ce'] = compute_ce(img)
    metrics['uiqm'] = compute_uiqm(img)

    # 计算有参考指标
    if reference_img is not None:
        metrics['psnr'] = compute_psnr(img, reference_img)
        metrics['ssim'] = compute_ssim(img, reference_img)

    return metrics