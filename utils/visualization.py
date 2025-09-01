"""
可视化工具模块，用于生成各种可视化结果
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time


def visualize_ddm_decision(orig_imgs, enhanced_imgs, ddm_details, output_path=None):
    """
    可视化DDM决策过程

    参数:
        orig_imgs: 原始图像对 (列表，[img1, img2])
        enhanced_imgs: 增强图像对 (列表，[img1, img2])
        ddm_details: DDM决策详情
        output_path: 结果保存路径
    """
    # 创建画布
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig)

    # 原始图像
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.imshow(cv2.cvtColor(orig_imgs[0], cv2.COLOR_BGR2RGB))
    ax1.set_title('原始图像 A')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 2:4])
    ax2.imshow(cv2.cvtColor(orig_imgs[1], cv2.COLOR_BGR2RGB))
    ax2.set_title('原始图像 B')
    ax2.axis('off')

    # 增强图像
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax3.imshow(cv2.cvtColor(enhanced_imgs[0], cv2.COLOR_BGR2RGB))
    ax3.set_title('增强图像 A')
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[1, 2:4])
    ax4.imshow(cv2.cvtColor(enhanced_imgs[1], cv2.COLOR_BGR2RGB))
    ax4.set_title('增强图像 B')
    ax4.axis('off')

    # 决策分数可视化
    ax5 = fig.add_subplot(gs[2, 1:3])

    labels = ['匹配性', '颜色质量', '总分']
    x = np.arange(len(labels))
    width = 0.35

    # 准备数据
    orig_scores = [
        ddm_details['original']['matchability'],
        ddm_details['original']['color_quality'],
        ddm_details['original']['total_score']
    ]

    enh_scores = [
        ddm_details['enhanced']['matchability'],
        ddm_details['enhanced']['color_quality'],
        ddm_details['enhanced']['total_score']
    ]

    # 绘制条形图
    rects1 = ax5.bar(x - width / 2, orig_scores, width, label='原始图像')
    rects2 = ax5.bar(x + width / 2, enh_scores, width, label='增强图像')

    # 添加标签
    ax5.set_ylabel('分数')
    ax5.set_title('DDM 决策评分')
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels)
    ax5.legend()

    # 在条形图上添加具体数值
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax5.annotate(f'{height:.2f}',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    # 添加决策结果
    choice = "增强图像" if ddm_details['enhanced']['total_score'] > ddm_details['original'][
        'total_score'] + 0.1 else "原始图像"
    fig.suptitle(f'DDM 决策结果: 选择{choice}', fontsize=16)

    plt.tight_layout()

    # 保存结果
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def visualize_feature_matching(img1, img2, matches, matcher_name, output_path=None):
    """
    可视化特征匹配结果

    参数:
        img1, img2: 输入图像对
        matches: 匹配结果
        matcher_name: 匹配器名称
        output_path: 结果保存路径
    """
    # 确保图像是uint8类型
    if img1.dtype != np.uint8:
        img1 = np.clip(img1 * 255.0 if img1.max() <= 1.0 else img1, 0, 255).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = np.clip(img2 * 255.0 if img2.max() <= 1.0 else img2, 0, 255).astype(np.uint8)

    # 创建匹配点可视化
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 创建连接图像
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1 if len(img1.shape) == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    vis[:h2, w1:w1 + w2] = img2 if len(img2.shape) == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # 绘制匹配点
    mkpts0 = matches['mkpts0']
    mkpts1 = matches['mkpts1']
    mconf = matches['mconf']

    # 根据置信度为线条设置颜色
    for i, (pt1, pt2, conf) in enumerate(zip(mkpts0, mkpts1, mconf)):
        pt1 = tuple(map(int, pt1))
        pt2 = (int(pt2[0] + w1), int(pt2[1]))

        # 根据置信度计算颜色 (低置信度红色，高置信度绿色)
        color = (
            int(255 * (1 - conf)),  # B
            int(255 * conf),  # G
            0  # R
        )

        # 绘制线条
        cv2.line(vis, pt1, pt2, color, 1)

        # 绘制点
        cv2.circle(vis, pt1, 3, color, -1)
        cv2.circle(vis, pt2, 3, color, -1)

    # 添加文本信息
    cv2.putText(vis, f"匹配器: {matcher_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(vis, f"匹配点数量: {len(mkpts0)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if len(mconf) > 0:
        avg_conf = np.mean(mconf)
        cv2.putText(vis, f"平均置信度: {avg_conf:.3f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 保存或显示结果
    if output_path:
        cv2.imwrite(output_path, vis)

    return vis


def visualize_stitching_progress(img1, img2, stitched, refined=None, output_path=None):
    """
    可视化拼接过程

    参数:
        img1, img2: 输入图像对
        stitched: 拼接结果
        refined: 优化后的拼接结果 (可选)
        output_path: 结果保存路径
    """
    # 创建画布
    rows = 2 if refined is not None else 1
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))

    # 如果只有一行，调整axes形状
    if rows == 1:
        axes = [axes]

    # 显示输入图像
    axes[0][0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0][0].set_title('输入图像 A')
    axes[0][0].axis('off')

    axes[0][1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[0][1].set_title('输入图像 B')
    axes[0][1].axis('off')

    # 显示拼接结果
    axes[0][2].imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
    axes[0][2].set_title('拼接结果')
    axes[0][2].axis('off')

    # 如果有优化结果，显示优化前后对比
    if refined is not None:
        # 计算差异图
        diff = cv2.absdiff(stitched, refined)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)

        # 增强差异以便可视化
        diff = diff * 5
        diff = np.clip(diff, 0, 255).astype(np.uint8)

        axes[1][0].imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
        axes[1][0].set_title('拼接结果')
        axes[1][0].axis('off')

        axes[1][1].imshow(cv2.cvtColor(refined, cv2.COLOR_BGR2RGB))
        axes[1][1].set_title('优化结果')
        axes[1][1].axis('off')

        axes[1][2].imshow(diff)
        axes[1][2].set_title('差异 (x5)')
        axes[1][2].axis('off')

    plt.tight_layout()

    # 保存或显示结果
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def visualize_metrics(metrics_dict, output_path=None):
    """
    可视化图像质量指标

    参数:
        metrics_dict: 包含评估指标的字典
        output_path: 结果保存路径
    """
    # 准备数据
    stages = list(metrics_dict.keys())
    psnr_values = [metrics_dict[s].get('psnr', 0) for s in stages if 'psnr' in metrics_dict[s]]
    ssim_values = [metrics_dict[s].get('ssim', 0) for s in stages if 'ssim' in metrics_dict[s]]
    ce_values = [metrics_dict[s].get('ce', 0) for s in stages if 'ce' in metrics_dict[s]]
    uiqm_values = [metrics_dict[s].get('uiqm', 0) for s in stages if 'uiqm' in metrics_dict[s]]

    # 创建图表
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # PSNR图表
    if psnr_values:
        axs[0, 0].bar(stages, psnr_values, color='skyblue')
        axs[0, 0].set_title('PSNR (越高越好)')
        axs[0, 0].set_ylabel('dB')

        # 添加具体值
        for i, v in enumerate(psnr_values):
            axs[0, 0].text(i, v + 0.5, f"{v:.2f}", ha='center')

    # SSIM图表
    if ssim_values:
        axs[0, 1].bar(stages, ssim_values, color='lightgreen')
        axs[0, 1].set_title('SSIM (越高越好)')
        axs[0, 1].set_ylabel('值')

        # 添加具体值
        for i, v in enumerate(ssim_values):
            axs[0, 1].text(i, v + 0.02, f"{v:.3f}", ha='center')

    # CE图表
    if ce_values:
        axs[1, 0].bar(stages, ce_values, color='salmon')
        axs[1, 0].set_title('CE (越低越好)')
        axs[1, 0].set_ylabel('值')

        # 添加具体值
        for i, v in enumerate(ce_values):
            axs[1, 0].text(i, v + 0.1, f"{v:.2f}", ha='center')

    # UIQM图表
    if uiqm_values:
        axs[1, 1].bar(stages, uiqm_values, color='plum')
        axs[1, 1].set_title('UIQM (越高越好)')
        axs[1, 1].set_ylabel('值')

        # 添加具体值
        for i, v in enumerate(uiqm_values):
            axs[1, 1].text(i, v + 0.1, f"{v:.2f}", ha='center')

    plt.tight_layout()

    # 保存或显示结果
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()