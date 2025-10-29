import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def save_prediction_visualization(images, masks, outputs, save_path, slice_idx=None):
    """保存分割预测结果的综合可视化
    
    创建一个多面板的可视化图像，展示模型的分割性能。这种可视化方式能够：
    1. 直观比较预测结果与真实标签
    2. 识别模型的分割错误模式
    3. 评估不同肿瘤区域的分割质量
    4. 为模型调优提供视觉反馈
    
    可视化布局：
    - 第1列：原始FLAIR图像（提供解剖学背景）
    - 第2列：真实分割标签（彩色编码的ground truth）
    - 第3列：模型预测结果（彩色编码的预测）
    - 第4列：预测差异图（高亮显示错误区域）
    
    颜色编码方案：
    - 红色：坏死核心（Necrotic Core, NCR）
    - 绿色：水肿区域（Edema, ED）
    - 蓝色：增强肿瘤（Enhancing Tumor, ET）
    - 背景：透明（显示底层FLAIR图像）
    
    参数:
        images (ndarray/Tensor): 多模态MRI输入，形状为 [B, 4, D, H, W]
            4个通道对应：T1, T1ce, T2, FLAIR
        masks (ndarray/Tensor): 真实分割标签，形状为 [B, D, H, W]
            标签值：0=背景, 1=坏死核心, 2=水肿, 3=增强肿瘤
        outputs (ndarray/Tensor): 模型预测logits，形状为 [B, C, D, H, W]
            C为类别数（通常为4）
        save_path (str): 可视化图像的保存路径（支持.png, .jpg等格式）
        slice_idx (int, 可选): 要可视化的切片索引
            如果为None，自动选择包含最多前景像素的切片
    """
    # ==================== 数据类型转换 ====================
    # 确保所有输入都是CPU上的numpy数组
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
    
    # ==================== 参数提取 ====================
    batch_size = images.shape[0]
    depth = images.shape[2]
    
    # 智能选择可视化切片
    if slice_idx is None:
        # 选择包含最多前景像素的切片，这样的切片通常最有代表性
        foreground_counts = []
        for z in range(depth):
            # 计算每个切片中前景像素的数量
            fg_count = np.sum(masks[0, z] > 0)  # 使用第一个样本
            foreground_counts.append(fg_count)
        
        # 选择前景像素最多的切片
        slice_idx = np.argmax(foreground_counts)
        print(f"自动选择切片 {slice_idx}（前景像素数：{max(foreground_counts)}）")
    
    # 确保切片索引在有效范围内
    slice_idx = max(0, min(slice_idx, depth - 1))
    
    # ==================== 创建可视化布局 ====================
    # 创建子图网格：每行对应一个批次样本，每行4列
    fig, axes = plt.subplots(
        batch_size, 4, 
        figsize=(20, 5 * batch_size),
        facecolor='white'
    )
    
    # 处理单样本情况，确保axes是二维数组
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    # ==================== 为每个样本创建可视化 ====================
    for batch_idx in range(batch_size):
        # 提取当前样本的数据
        sample_images = images[batch_idx]  # [4, D, H, W]
        sample_mask = masks[batch_idx, slice_idx]  # [H, W]
        
        # 提取各个模态的切片
        t1_slice = sample_images[0, slice_idx]      # T1加权
        t1ce_slice = sample_images[1, slice_idx]    # T1对比增强
        t2_slice = sample_images[2, slice_idx]      # T2加权
        flair_slice = sample_images[3, slice_idx]   # FLAIR（用于显示）
        
        # ==================== 处理预测结果 ====================
        sample_outputs = outputs[batch_idx]  # [C, D, H, W]
        
        if sample_outputs.shape[0] > 1:
            # 多类别情况：使用argmax获取预测类别
            pred_slice = np.argmax(sample_outputs, axis=0)[slice_idx]
        else:
            # 二分类情况：使用阈值二值化
            pred_slice = (sample_outputs[0, slice_idx] > 0.5).astype(np.uint8)
        
        # ==================== 创建彩色掩码 ====================
        # 为真实标签和预测结果创建RGB彩色表示
        mask_rgb = np.zeros((*sample_mask.shape, 3), dtype=np.float32)
        pred_rgb = np.zeros((*pred_slice.shape, 3), dtype=np.float32)
        
        # 应用颜色编码
        # 坏死核心（类别1）-> 红色
        mask_rgb[sample_mask == 1] = [1.0, 0.0, 0.0]
        pred_rgb[pred_slice == 1] = [1.0, 0.0, 0.0]
        
        # 水肿区域（类别2）-> 绿色
        mask_rgb[sample_mask == 2] = [0.0, 1.0, 0.0]
        pred_rgb[pred_slice == 2] = [0.0, 1.0, 0.0]
        
        # 增强肿瘤（类别3）-> 蓝色
        mask_rgb[sample_mask == 3] = [0.0, 0.0, 1.0]
        pred_rgb[pred_slice == 3] = [0.0, 0.0, 1.0]
        
        # ==================== 创建各个面板 ====================
        # 面板1：原始FLAIR图像
        axes[batch_idx, 0].imshow(flair_slice, cmap='gray', vmin=0, vmax=1)
        axes[batch_idx, 0].set_title(f'FLAIR (切片 {slice_idx})', fontsize=12, fontweight='bold')
        axes[batch_idx, 0].axis('off')
        
        # 面板2：真实标签叠加
        axes[batch_idx, 1].imshow(flair_slice, cmap='gray', vmin=0, vmax=1)
        axes[batch_idx, 1].imshow(mask_rgb, alpha=0.6)  # 半透明叠加
        axes[batch_idx, 1].set_title('真实标签', fontsize=12, fontweight='bold')
        axes[batch_idx, 1].axis('off')
        
        # 面板3：预测结果叠加
        axes[batch_idx, 2].imshow(flair_slice, cmap='gray', vmin=0, vmax=1)
        axes[batch_idx, 2].imshow(pred_rgb, alpha=0.6)  # 半透明叠加
        axes[batch_idx, 2].set_title('预测结果', fontsize=12, fontweight='bold')
        axes[batch_idx, 2].axis('off')
        
        # 面板4：差异分析
        # 计算预测错误的像素
        diff_mask = (sample_mask != pred_slice).astype(np.float32)
        
        # 创建差异可视化（错误区域显示为红色）
        diff_rgb = np.zeros((*diff_mask.shape, 3), dtype=np.float32)
        diff_rgb[diff_mask == 1] = [1.0, 0.0, 0.0]  # 错误区域为红色
        
        axes[batch_idx, 3].imshow(flair_slice, cmap='gray', vmin=0, vmax=1)
        axes[batch_idx, 3].imshow(diff_rgb, alpha=0.7)
        
        # 计算错误率
        total_pixels = sample_mask.size
        error_pixels = np.sum(diff_mask)
        error_rate = (error_pixels / total_pixels) * 100
        
        axes[batch_idx, 3].set_title(f'预测差异 (错误率: {error_rate:.1f}%)', 
                                   fontsize=12, fontweight='bold')
        axes[batch_idx, 3].axis('off')
    
    # ==================== 添加颜色图例 ====================
    # 在图像底部添加颜色编码说明
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.6, label='坏死核心'),
        plt.Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.6, label='水肿区域'),
        plt.Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.6, label='增强肿瘤')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
              bbox_to_anchor=(0.5, -0.02), fontsize=12)
    
    # ==================== 保存和清理 ====================
    # 调整布局并保存
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)  # 为图例留出空间
    
    # 创建保存目录（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存高质量图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # 释放内存
    plt.close(fig)
    
    print(f"可视化结果已保存到: {save_path}")


def visualize_3d_volume(volume, save_dir, prefix='slice', cmap='gray'):
    """将3D体积可视化为一系列2D切片
    
    Args:
        volume: 3D体积数据
        save_dir: 保存目录
        prefix: 文件名前缀
        cmap: 颜色映射
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 确保输入是numpy数组
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
    
    # 获取体积的深度
    depth = volume.shape[0]
    
    # 保存每个切片
    for z in range(depth):
        plt.figure(figsize=(5, 5))
        plt.imshow(volume[z], cmap=cmap)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_{z:03d}.png'))
        plt.close()


def create_3d_gif(volume, save_path, duration=100, cmap='viridis', title_prefix="切片"):
    """创建3D体积的GIF动画可视化
    
    将3D医学图像体积转换为逐切片播放的GIF动画，便于观察整个体积的3D结构。
    这种可视化方式特别适合：
    1. 展示分割结果的3D连续性
    2. 检查分割质量在不同切片上的一致性
    3. 识别可能的分割错误或异常
    4. 为临床医生提供直观的3D视图
    
    动画特点：
    - 按切片顺序播放，模拟3D浏览体验
    - 支持自定义颜色映射，适应不同数据类型
    - 可调节播放速度，平衡观察效果和文件大小
    - 自动处理不同的输入数据格式
    
    参数:
        volume (ndarray/Tensor): 3D体积数据，形状为 [D, H, W]
            - 对于分割掩码：值为离散的类别标签
            - 对于图像数据：值为连续的强度值
        save_path (str): GIF文件的保存路径（必须以.gif结尾）
        duration (int): 每帧的持续时间（毫秒），默认100ms
            - 较小值：播放速度快，适合快速浏览
            - 较大值：播放速度慢，适合仔细观察
        cmap (str): matplotlib颜色映射名称，默认'viridis'
            - 'viridis': 适合连续数据，视觉友好
            - 'tab10': 适合离散标签，颜色区分明显
            - 'gray': 适合灰度图像
        title_prefix (str): 每帧标题的前缀，默认"切片"
    """
    try:
        # 检查必要的依赖库
        import imageio
        from PIL import Image
    except ImportError as e:
        print(f"错误：缺少必要的库 {e}")
        print("请安装：pip install imageio pillow")
        return
    
    # ==================== 数据预处理 ====================
    # 确保输入是numpy数组
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
    
    # 验证输入数据
    if len(volume.shape) != 3:
        raise ValueError(f"输入体积必须是3D数组，当前形状: {volume.shape}")
    
    depth, height, width = volume.shape
    
    if depth == 0:
        raise ValueError("体积深度不能为0")
    
    # 数据归一化（用于更好的可视化效果）
    if volume.dtype in [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32]:
        # 整数类型，可能是分割标签
        volume_normalized = volume.astype(np.float32)
    else:
        # 浮点类型，进行归一化
        volume_min = volume.min()
        volume_max = volume.max()
        if volume_max > volume_min:
            volume_normalized = (volume - volume_min) / (volume_max - volume_min)
        else:
            volume_normalized = volume.astype(np.float32)
    
    # ==================== 创建动画帧 ====================
    print(f"正在创建 {depth} 帧的GIF动画...")
    
    frames = []
    
    # 设置matplotlib参数以获得更好的图像质量
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    for z in range(depth):
        # 创建当前切片的图像
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        
        # 显示当前切片
        im = ax.imshow(
            volume_normalized[z], 
            cmap=cmap,
            interpolation='nearest',  # 保持像素的锐利边缘
            aspect='equal'           # 保持像素的正方形形状
        )
        
        # 设置标题和样式
        ax.set_title(f'{title_prefix} {z+1}/{depth}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')  # 隐藏坐标轴
        
        # 添加颜色条（仅对连续数据）
        if not np.array_equal(volume_normalized[z], volume_normalized[z].astype(int)):
            # 连续数据，添加颜色条
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
            cbar.ax.tick_params(labelsize=12)
        
        # 调整布局
        plt.tight_layout()
        
        # ==================== 将图像转换为数组 ====================
        # 将matplotlib图像转换为numpy数组
        fig.canvas.draw()
        
        # 获取图像数据
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        frames.append(buf)
        
        # 关闭当前图像以释放内存
        plt.close(fig)
        
        # 显示进度
        if (z + 1) % max(1, depth // 10) == 0:
            progress = (z + 1) / depth * 100
            print(f"进度: {progress:.1f}% ({z+1}/{depth})")
    
    # ==================== 创建和保存GIF ====================
    print("正在保存GIF文件...")
    
    try:
        # 创建保存目录（如果不存在）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存GIF动画
        imageio.mimsave(
            save_path, 
            frames, 
            duration=duration/1000.0,  # imageio使用秒为单位
            loop=0  # 无限循环
        )
        
        # 计算文件大小
        file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
        
        print(f"GIF动画已成功保存到: {save_path}")
        print(f"文件大小: {file_size:.2f} MB")
        print(f"总帧数: {len(frames)}")
        print(f"播放时长: {len(frames) * duration / 1000:.1f} 秒")
        
    except Exception as e:
        print(f"保存GIF时出错: {e}")
        raise
    
    # ==================== 清理资源 ====================
    # 重置matplotlib参数
    plt.rcdefaults()
    
    print("GIF创建完成！")