import os
import argparse
import numpy as np
import torch
import nibabel as nib  # 用于处理医学影像格式
from tqdm import tqdm  # 进度条显示

# 导入自定义模块
from models.unet3d import UNet3D  # 3D U-Net模型
from utils.data_utils import normalize_scan, resample_volume, crop_brain_region  # 数据预处理工具
from utils.visualization import save_prediction_visualization, create_3d_gif  # 可视化工具


def parse_args():
    """解析命令行参数
    
    定义并解析推理脚本的命令行参数，包括输入输出路径、模型路径和预处理参数等
    
    返回:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description='Inference with 3D U-Net on BraTS2021 dataset')
    
    # 数据参数
    parser.add_argument('--input_dir', type=str, required=True,
                        help='输入目录，包含待预测的MRI图像')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='输出目录，用于保存预测结果')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型检查点路径')
    
    # 预处理参数
    parser.add_argument('--target_shape', type=int, nargs='+', default=[128, 128, 128],
                        help='目标体积形状，用于重采样MRI体积')
    
    # 可视化参数
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='是否生成可视化结果，包括2D切片可视化')
    parser.add_argument('--create_gif', action='store_true',
                        help='是否创建GIF动画，用于3D可视化')
    
    return parser.parse_args()


def load_model(model_path, device):
    """加载预训练的3D U-Net模型
    
    从保存的检查点文件中恢复完整的模型状态，包括网络架构参数和训练好的权重。
    这个函数确保模型能够正确地从训练状态转换到推理状态。
    
    加载流程：
    1. 读取检查点文件
    2. 提取模型配置参数
    3. 重建网络架构
    4. 加载预训练权重
    5. 设置为评估模式
    
    参数:
        model_path (str): 模型检查点文件的完整路径
            通常是训练过程中保存的.pth文件
        device (torch.device): 目标计算设备
            torch.device('cuda') 或 torch.device('cpu')
        
    返回:
        tuple: (加载的模型实例, 训练时的参数配置)
            - model: 已加载权重并设置为评估模式的UNet3D实例
            - args: 包含模型配置的参数对象
            
    异常:
        FileNotFoundError: 当模型文件不存在时
        RuntimeError: 当模型加载失败时
    """
    try:
        print(f"正在加载模型: {model_path}")
        
        # ==================== 加载检查点文件 ====================
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载检查点，map_location确保能在目标设备上运行
        checkpoint = torch.load(model_path, map_location=device)
        
        # 验证检查点文件的完整性
        required_keys = ['model_state', 'args']
        for key in required_keys:
            if key not in checkpoint:
                raise KeyError(f"检查点文件缺少必要的键: {key}")
        
        # 提取训练时的参数配置
        args = checkpoint['args']
        
        print(f"模型配置:")
        print(f"  - 输入通道数: {args.in_channels}")
        print(f"  - 输出通道数: {args.out_channels}")
        print(f"  - 特征通道数: {args.features}")
        print(f"  - 深度监督: {args.deep_supervision}")
        print(f"  - 残差连接: {args.residual}")
        
        # ==================== 重建网络架构 ====================
        model = UNet3D(
            in_channels=args.in_channels,      # 输入通道数（4个MRI模态）
            out_channels=args.out_channels,    # 输出通道数（分割类别数）
            features=args.features,            # 每层的特征通道数
            deep_supervision=args.deep_supervision,  # 是否使用深度监督
            residual=args.residual             # 是否使用残差连接
        )
        
        # ==================== 加载预训练权重 ====================
        # 加载模型状态字典
        model_state = checkpoint['model_state']
        
        # 处理可能的键名不匹配问题
        try:
            model.load_state_dict(model_state, strict=True)
        except RuntimeError as e:
            print(f"严格模式加载失败，尝试非严格模式: {e}")
            model.load_state_dict(model_state, strict=False)
        
        # 将模型移动到目标设备
        model = model.to(device)
        
        # ==================== 设置为评估模式 ====================
        model.eval()  # 禁用dropout和batch normalization的随机性
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"模型加载成功!")
        print(f"  - 总参数数量: {total_params:,}")
        print(f"  - 可训练参数: {trainable_params:,}")
        print(f"  - 运行设备: {device}")
        
        # 如果有训练信息，也打印出来
        if 'epoch' in checkpoint:
            print(f"  - 训练轮数: {checkpoint['epoch']}")
        if 'best_dice' in checkpoint:
            print(f"  - 最佳Dice分数: {checkpoint['best_dice']:.4f}")
        
        return model, args
        
    except Exception as e:
        error_msg = f"加载模型时发生错误: {str(e)}"
        print(f"错误: {error_msg}")
        raise RuntimeError(error_msg)


def preprocess_scan(t1_path, t1ce_path, t2_path, flair_path, target_shape):
    """多模态MRI扫描的标准化预处理
    
    执行与训练时相同的预处理流程，确保推理数据与训练数据的一致性。
    这个函数是推理管道的关键组件，直接影响模型的预测质量。
    
    预处理流程：
    1. 读取四种MRI模态的NIfTI文件
    2. 提取图像数据和元信息
    3. 强度归一化（Min-Max归一化到[0,1]）
    4. 脑部区域智能裁剪
    5. 重采样到统一的目标尺寸
    6. 多模态数据堆叠
    7. 转换为PyTorch张量格式
    
    数据一致性保证：
    - 使用与训练时相同的归一化方法
    - 应用相同的裁剪策略
    - 保持相同的模态顺序：[T1, T1ce, T2, FLAIR]
    
    参数:
        t1_path (str): T1加权MRI文件路径
        t1ce_path (str): T1对比增强MRI文件路径
        t2_path (str): T2加权MRI文件路径
        flair_path (str): FLAIR序列MRI文件路径
        target_shape (list/tuple): 目标重采样尺寸，如 [128, 128, 128]
        
    返回:
        tuple: (预处理后的图像张量, 原始仿射矩阵, 原始头信息)
            - image: 形状为 [1, 4, D, H, W] 的PyTorch张量
            - affine: 原始图像的仿射变换矩阵
            - header: 原始图像的NIfTI头信息
            
    异常:
        FileNotFoundError: 当任何输入文件不存在时
        RuntimeError: 当预处理过程中发生错误时
    """
    try:
        print("开始预处理MRI扫描数据...")
        
        # ==================== 验证输入文件 ====================
        file_paths = {
            'T1': t1_path,
            'T1ce': t1ce_path,
            'T2': t2_path,
            'FLAIR': flair_path
        }
        
        for modality, path in file_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"{modality} 文件不存在: {path}")
        
        # ==================== 读取NIfTI文件 ====================
        print("正在读取NIfTI文件...")
        
        try:
            t1_nii = nib.load(t1_path)
            t1ce_nii = nib.load(t1ce_path)
            t2_nii = nib.load(t2_path)
            flair_nii = nib.load(flair_path)
        except Exception as e:
            raise RuntimeError(f"读取NIfTI文件失败: {str(e)}")
        
        # 提取图像数据
        t1 = t1_nii.get_fdata().astype(np.float32)
        t1ce = t1ce_nii.get_fdata().astype(np.float32)
        t2 = t2_nii.get_fdata().astype(np.float32)
        flair = flair_nii.get_fdata().astype(np.float32)
        
        # 保存原始元信息用于后处理
        original_shape = flair.shape
        affine = flair_nii.affine.copy()
        header = flair_nii.header.copy()
        
        print(f"原始图像尺寸: {original_shape}")
        print(f"目标图像尺寸: {target_shape}")
        
        # ==================== 强度归一化 ====================
        print("正在进行强度归一化...")
        
        # 对每个模态独立进行Min-Max归一化
        modalities = {'T1': t1, 'T1ce': t1ce, 'T2': t2, 'FLAIR': flair}
        normalized_modalities = {}
        
        for name, data in modalities.items():
            normalized_data = normalize_scan(data)
            normalized_modalities[name] = normalized_data
            
            # 打印归一化统计信息
            print(f"  {name}: [{data.min():.2f}, {data.max():.2f}] -> "
                  f"[{normalized_data.min():.2f}, {normalized_data.max():.2f}]")
        
        # ==================== 脑部区域裁剪 ====================
        print("正在进行脑部区域裁剪...")
        
        # 使用FLAIR作为参考进行裁剪，因为它通常有最好的脑组织对比度
        reference_modality = normalized_modalities['FLAIR']
        
        # 对所有模态应用相同的裁剪区域
        cropped_modalities = {}
        for name, data in normalized_modalities.items():
            if name == 'FLAIR':
                # 对参考模态进行裁剪
                cropped_data, _ = crop_brain_region(data)
            else:
                # 对其他模态应用相同的裁剪
                cropped_data, _ = crop_brain_region(data)
            
            cropped_modalities[name] = cropped_data
        
        cropped_shape = cropped_modalities['FLAIR'].shape
        print(f"裁剪后尺寸: {cropped_shape}")
        
        # ==================== 重采样到目标尺寸 ====================
        print("正在重采样到目标尺寸...")
        
        resampled_modalities = {}
        for name, data in cropped_modalities.items():
            resampled_data = resample_volume(
                data, target_shape, interpolation='linear'
            )
            resampled_modalities[name] = resampled_data
        
        # ==================== 多模态数据堆叠 ====================
        print("正在组合多模态数据...")
        
        # 按固定顺序堆叠：[T1, T1ce, T2, FLAIR]
        # 这个顺序必须与训练时保持一致
        image = np.stack([
            resampled_modalities['T1'],
            resampled_modalities['T1ce'],
            resampled_modalities['T2'],
            resampled_modalities['FLAIR']
        ], axis=0)  # 形状: (4, D, H, W)
        
        # ==================== 转换为PyTorch张量 ====================
        # 添加批次维度并转换为PyTorch张量
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # 形状: (1, 4, D, H, W)
        
        print(f"预处理完成!")
        print(f"  - 输出张量形状: {image_tensor.shape}")
        print(f"  - 数据类型: {image_tensor.dtype}")
        print(f"  - 数值范围: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
        
        return image_tensor, affine, header
        
    except Exception as e:
        error_msg = f"预处理过程中发生错误: {str(e)}"
        print(f"错误: {error_msg}")
        raise RuntimeError(error_msg)


def postprocess_prediction(pred, original_shape, affine, header):
    """后处理预测结果
    
    将模型输出的预测结果转换为NIfTI格式，并重采样回原始形状
    
    参数:
        pred (torch.Tensor): 模型预测输出
        original_shape (tuple): 原始图像形状
        affine (numpy.ndarray): 原始图像的仿射矩阵
        header (nibabel.nifti1.Nifti1Header): 原始图像的头信息
        
    返回:
        nibabel.nifti1.Nifti1Image: 后处理后的NIfTI格式预测结果
    """
    # 将预测结果转换为numpy数组
    if pred.shape[1] > 1:  # 多通道输出（多分类情况）
        pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()  # 取最大概率的类别
    else:  # 单通道输出（二分类情况）
        pred = (torch.sigmoid(pred) > 0.5).squeeze().cpu().numpy().astype(np.uint8)  # 二值化
    
    # 重采样回原始形状，使用最近邻插值保持分割标签的整数值
    pred_original = resample_volume(pred, original_shape, interpolation='nearest')
    
    # 创建NIfTI对象，保持与原始图像相同的仿射矩阵和头信息
    pred_nii = nib.Nifti1Image(pred_original, affine, header)
    
    return pred_nii


def main():
    """主函数
    
    推理脚本的主要执行流程，包括参数解析、模型加载、数据预处理、模型推理和结果保存
    """
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)  # 创建预测结果保存目录
    if args.visualize:
        os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)  # 创建可视化结果保存目录
    
    # 设置计算设备，优先使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载预训练模型
    model, model_args = load_model(args.model_path, device)
    
    # 获取输入目录中的所有BraTS病例
    cases = []
    for case_folder in os.listdir(args.input_dir):
        if os.path.isdir(os.path.join(args.input_dir, case_folder)) and case_folder.startswith('BraTS'):
            cases.append(case_folder)
    
    # 对每个病例进行预测，使用tqdm显示进度
    for case in tqdm(cases, desc='Processing cases'):
        case_path = os.path.join(args.input_dir, case)
        
        # 构建四种模态的文件路径
        t1_path = os.path.join(case_path, f"{case}_t1.nii.gz")  # T1模态
        t1ce_path = os.path.join(case_path, f"{case}_t1ce.nii.gz")  # T1增强模态
        t2_path = os.path.join(case_path, f"{case}_t2.nii.gz")  # T2模态
        flair_path = os.path.join(case_path, f"{case}_flair.nii.gz")  # FLAIR模态
        
        # 预处理扫描图像，包括归一化、裁剪和重采样
        image, affine, header = preprocess_scan(t1_path, t1ce_path, t2_path, flair_path, args.target_shape)
        
        # 获取原始图像形状，用于后处理时的重采样
        original_shape = nib.load(flair_path).shape
        
        # 使用模型进行预测，禁用梯度计算提高推理速度
        with torch.no_grad():
            image = image.to(device)  # 将图像数据移至计算设备
            outputs = model(image)  # 模型前向传播
            if isinstance(outputs, tuple):  # 如果使用深度监督，输出为元组
                outputs = outputs[0]  # 只使用主输出（最终层的输出）
        
        # 后处理预测结果，转换为NIfTI格式
        pred_nii = postprocess_prediction(outputs, original_shape, affine, header)
        
        # 保存预测结果为NIfTI文件
        nib.save(pred_nii, os.path.join(args.output_dir, f"{case}_pred.nii.gz"))
        
        # 可视化处理部分
        if args.visualize:
            # 加载真实分割掩码（如果存在）用于对比显示
            seg_path = os.path.join(case_path, f"{case}_seg.nii.gz")
            if os.path.exists(seg_path):
                # 读取真实分割掩码
                seg_nii = nib.load(seg_path)
                seg = seg_nii.get_fdata()
                
                # 处理分割标签，将BraTS标准标签转换为连续的类别索引
                mask = np.zeros_like(seg)
                mask[seg == 1] = 1  # 坏死核心 (Necrotic core)
                mask[seg == 2] = 2  # 水肿区域 (Edema)
                mask[seg == 4] = 3  # 增强肿瘤 (Enhancing tumor)
                
                # 重采样掩码到与模型输入相同的目标形状
                mask = resample_volume(mask, args.target_shape, interpolation='nearest')
                mask = torch.from_numpy(mask).long().unsqueeze(0)  # 转换为张量并添加批次维度
                
                # 保存可视化结果，包含原始图像、真实掩码和预测结果的对比
                save_prediction_visualization(
                    image, mask, outputs,
                    os.path.join(args.output_dir, 'visualizations', f"{case}_pred.png")
                )
            else:
                # 如果没有真实掩码，只可视化原始图像和预测结果
                dummy_mask = torch.zeros_like(outputs[:, 0:1]).long()  # 创建空白掩码
                save_prediction_visualization(
                    image, dummy_mask, outputs,
                    os.path.join(args.output_dir, 'visualizations', f"{case}_pred.png")
                )
            
            # 创建3D GIF动画用于立体可视化
            if args.create_gif:
                # 将模型输出转换为分割掩码
                if outputs.shape[1] > 1:  # 多通道输出（多分类情况）
                    pred = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()  # 取最大概率的类别
                else:  # 单通道输出（二分类情况）
                    pred = (torch.sigmoid(outputs) > 0.5).squeeze().cpu().numpy().astype(np.uint8)  # 二值化
                
                # 创建并保存3D旋转GIF动画，duration控制帧间隔（毫秒）
                create_3d_gif(
                    pred,
                    os.path.join(args.output_dir, 'visualizations', f"{case}_pred.gif"),
                    duration=100  # 帧间隔时间（毫秒）
                )


if __name__ == '__main__':
    # 程序入口点，调用主函数
    main()