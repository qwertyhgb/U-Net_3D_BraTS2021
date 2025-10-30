import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast  # 用于混合精度训练
import torch.distributed as dist  # 分布式训练
from torch.nn.parallel import DistributedDataParallel as DDP  # 分布式数据并行
from torch.utils.data.distributed import DistributedSampler  # 分布式采样器
from torch.utils.tensorboard import SummaryWriter  # TensorBoard可视化
from tqdm import tqdm  # 进度条

# 导入自定义模块
from models.unet3d import UNet3D  # 3D U-Net模型
from utils.data_utils import get_data_loaders  # 数据加载工具
from utils.metrics import multiclass_dice_coefficient, multiclass_hausdorff_distance  # 评估指标
from utils.visualization import save_prediction_visualization  # 可视化工具


def parse_args():
    """解析命令行参数
    
    定义并解析训练脚本的命令行参数，包括数据路径、模型配置、训练参数等
    
    返回:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description='Train 3D U-Net on BraTS2021 dataset')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data/BraTS2021_Training_Data',
                        help='BraTS2021数据集根目录')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录，用于保存模型和日志')
    
    # 模型参数
    parser.add_argument('--in_channels', type=int, default=4,
                        help='输入通道数，对应四种模态(T1、T1ce、T2、FLAIR)')
    parser.add_argument('--out_channels', type=int, default=4,
                        help='输出通道数，对应分割类别数(背景、坏死核心、水肿、增强肿瘤)')
    parser.add_argument('--features', type=int, nargs='+', default=[16, 32, 64, 128, 256],
                        help='每层的特征图数量，从浅层到深层')
    parser.add_argument('--deep_supervision', action='store_true',
                        help='是否使用深度监督，从多个深度层级输出预测结果')
    parser.add_argument('--residual', action='store_true', default=True,
                        help='是否使用残差连接，有助于训练更深的网络')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批量大小，受GPU内存限制')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数，总训练次数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='初始学习率，Adam优化器的学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减，用于L2正则化')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停耐心值，验证集性能不再提升的最大轮数')
    parser.add_argument('--target_shape', type=int, nargs='+', default=[128, 128, 128],
                        help='目标体积形状，用于重采样MRI体积')
    
    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'],
                        help='优化器选择：adam, adamw, sgd')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD优化器的动量参数')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Adam优化器的beta1参数')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam优化器的beta2参数')
    
    # 学习率调度器参数
    parser.add_argument('--scheduler', type=str, default='plateau', 
                        choices=['plateau', 'cosine', 'step', 'exponential'],
                        help='学习率调度器类型')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='学习率衰减因子')
    parser.add_argument('--lr_patience', type=int, default=5,
                        help='学习率调度器的耐心值')
    parser.add_argument('--min_lr', type=float, default=1e-7,
                        help='最小学习率')
    
    # 损失函数参数
    parser.add_argument('--loss_weights', type=float, nargs='+', default=[1.0, 1.0],
                        help='损失函数权重：[交叉熵权重, Dice权重]')
    parser.add_argument('--focal_loss', action='store_true',
                        help='是否使用Focal Loss处理类别不平衡')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Focal Loss的alpha参数')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss的gamma参数')
    
    # 数据增强参数
    parser.add_argument('--augmentation_prob', type=float, default=0.5,
                        help='数据增强的应用概率')
    parser.add_argument('--rotation_angle', type=float, default=15,
                        help='随机旋转的最大角度')
    parser.add_argument('--elastic_alpha', type=float, default=15,
                        help='弹性变形的强度参数')
    
    # 分布式训练参数
    parser.add_argument('--distributed', action='store_true',
                        help='是否使用分布式训练，多GPU训练')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='分布式训练的本地排名，由torch.distributed.launch设置')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='数据加载器的工作进程数，用于并行数据加载')
    
    # 混合精度训练
    parser.add_argument('--amp', action='store_true', default=True,
                        help='是否使用混合精度训练，可加速训练并减少内存使用')
    
    # 梯度相关参数
    parser.add_argument('--gradient_clipping', type=float, default=1.0,
                        help='梯度裁剪的最大范数，0表示不使用梯度裁剪')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='梯度累积步数，用于模拟更大的批次大小')
    
    # 可视化和日志参数
    parser.add_argument('--vis_freq', type=int, default=10,
                        help='可视化频率（每n个epoch保存一次可视化结果）')
    parser.add_argument('--log_freq', type=int, default=10,
                        help='日志记录频率（每n个batch记录一次详细信息）')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='模型保存频率（每n个epoch保存一次检查点）')
    
    # 恢复训练参数
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='预训练模型路径')
    
    # 验证和测试参数
    parser.add_argument('--val_freq', type=int, default=1,
                        help='验证频率（每n个epoch进行一次验证）')
    parser.add_argument('--test_only', action='store_true',
                        help='仅进行测试，不训练')
    
    return parser.parse_args()


def setup_distributed(args):
    """设置分布式训练环境
    
    初始化分布式训练环境，设置当前设备和进程组
    
    参数:
        args: 命令行参数对象，包含分布式训练相关配置
    """
    if args.distributed:
        # 设置当前设备
        torch.cuda.set_device(args.local_rank)
        # 初始化进程组，使用NCCL后端（适用于GPU）
        dist.init_process_group(backend='nccl')
        # 获取总进程数和当前进程排名
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
    else:
        # 非分布式训练时，设置为单进程
        args.world_size = 1
        args.rank = 0


def get_model(args):
    """创建模型实例
    
    根据参数创建3D U-Net模型实例
    
    参数:
        args: 命令行参数对象，包含模型配置
        
    返回:
        UNet3D: 创建的模型实例
    """
    model = UNet3D(
        in_channels=args.in_channels,  # 输入通道数（四种模态）
        out_channels=args.out_channels,  # 输出通道数（分割类别数）
        features=args.features,  # 每层特征图数量
        deep_supervision=args.deep_supervision,  # 是否使用深度监督
        residual=args.residual  # 是否使用残差连接
    )
    
    return model


def get_optimizer(model, args):
    """创建优化器
    
    根据参数配置创建不同类型的优化器，支持Adam、AdamW、SGD等
    
    参数:
        model: 要优化的模型
        args: 命令行参数
        
    返回:
        optimizer: 配置好的优化器
    """
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            eps=1e-8
        )
        print(f"使用Adam优化器，lr={args.lr}, betas=({args.beta1}, {args.beta2}), weight_decay={args.weight_decay}")
        
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            eps=1e-8
        )
        print(f"使用AdamW优化器，lr={args.lr}, betas=({args.beta1}, {args.beta2}), weight_decay={args.weight_decay}")
        
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True
        )
        print(f"使用SGD优化器，lr={args.lr}, momentum={args.momentum}, weight_decay={args.weight_decay}")
        
    else:
        raise ValueError(f"不支持的优化器类型: {args.optimizer}")
    
    return optimizer


def get_scheduler(optimizer, args):
    """创建学习率调度器
    
    根据参数配置创建不同类型的学习率调度器
    
    参数:
        optimizer: 优化器
        args: 命令行参数
        
    返回:
        scheduler: 配置好的学习率调度器
    """
    if args.scheduler.lower() == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # 监控指标越大越好（如Dice系数）
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
            verbose=True
        )
        print(f"使用ReduceLROnPlateau调度器，factor={args.lr_factor}, patience={args.lr_patience}")
        
    elif args.scheduler.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr
        )
        print(f"使用CosineAnnealingLR调度器，T_max={args.epochs}, eta_min={args.min_lr}")
        
    elif args.scheduler.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_patience * 2,  # 每隔patience*2个epoch衰减一次
            gamma=args.lr_factor
        )
        print(f"使用StepLR调度器，step_size={args.lr_patience * 2}, gamma={args.lr_factor}")
        
    elif args.scheduler.lower() == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=args.lr_factor
        )
        print(f"使用ExponentialLR调度器，gamma={args.lr_factor}")
        
    else:
        raise ValueError(f"不支持的调度器类型: {args.scheduler}")
    
    return scheduler


class FocalLoss(nn.Module):
    """Focal Loss实现
    
    Focal Loss专门设计用于处理类别不平衡问题，通过降低易分类样本的权重，
    让模型更专注于困难样本的学习。
    
    公式：FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)
    其中：
    - p_t: 正确类别的预测概率
    - α_t: 类别权重因子
    - γ: 聚焦参数，控制困难样本的权重
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_function(args):
    """构建优化的组合损失函数
    
    根据参数配置构建损失函数，支持多种损失类型的组合：
    1. 交叉熵损失 / Focal Loss
    2. Dice损失
    3. 可配置的权重平衡
    
    参数:
        args: 命令行参数，包含损失函数配置
        
    返回:
        function: 组合损失函数
    """
    
    def dice_loss(y_pred, y_true, smooth=1e-6):
        """改进的Dice损失计算
        
        使用更稳定的数值计算方式，避免梯度消失问题
        """
        # 展平张量
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)
        
        # 计算交集和并集
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum()
        
        # 使用改进的Dice计算公式，提高数值稳定性
        dice_coeff = (2. * intersection + smooth) / (union + smooth)
        
        return 1 - dice_coeff
    
    def tversky_loss(y_pred, y_true, alpha=0.3, beta=0.7, smooth=1e-6):
        """Tversky损失函数
        
        Tversky损失是Dice损失的泛化，通过α和β参数可以调节
        对假阳性和假阴性的敏感度，特别适合处理类别不平衡问题
        """
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (y_pred * y_true).sum()
        FP = ((1-y_true) * y_pred).sum()
        FN = (y_true * (1-y_pred)).sum()
        
        tversky_coeff = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        
        return 1 - tversky_coeff
    
    # 根据配置选择基础损失函数
    if args.focal_loss:
        base_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"使用Focal Loss，alpha={args.focal_alpha}, gamma={args.focal_gamma}")
    else:
        base_loss_fn = nn.CrossEntropyLoss()
        print("使用标准交叉熵损失")
    
    # 获取损失权重
    ce_weight, dice_weight = args.loss_weights
    print(f"损失权重配置：交叉熵={ce_weight}, Dice={dice_weight}")
    
    def combined_loss(y_pred, y_true):
        """优化的组合损失函数
        
        参数:
            y_pred (Tensor): 模型预测logits，形状为 [B, C, D, H, W]
            y_true (Tensor): 真实分割标签，形状为 [B, D, H, W]
            
        返回:
            dict: 包含各项损失的字典
        """
        # ==================== 基础损失计算 ====================
        base_loss = base_loss_fn(y_pred, y_true)
        
        # ==================== Dice/Tversky损失计算 ====================
        # 将预测logits转换为概率分布
        y_pred_softmax = torch.softmax(y_pred, dim=1)
        
        # 将真实标签转换为one-hot编码
        num_classes = y_pred.shape[1]
        y_true_one_hot = torch.zeros_like(y_pred_softmax)
        y_true_one_hot = y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1)
        
        # 计算每个前景类别的Dice损失
        dice_losses = []
        for class_idx in range(1, num_classes):  # 跳过背景类别
            pred_class = y_pred_softmax[:, class_idx]
            true_class = y_true_one_hot[:, class_idx]
            
            # 只有当该类别存在时才计算损失
            if true_class.sum() > 0:
                class_dice_loss = dice_loss(pred_class, true_class)
                dice_losses.append(class_dice_loss)
        
        # 计算平均Dice损失
        if dice_losses:
            avg_dice_loss = torch.stack(dice_losses).mean()
        else:
            avg_dice_loss = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        
        # ==================== 组合损失 ====================
        total_loss = ce_weight * base_loss + dice_weight * avg_dice_loss
        
        # 返回详细的损失信息用于监控
        return {
            'total_loss': total_loss,
            'base_loss': base_loss,
            'dice_loss': avg_dice_loss,
            'ce_weight': ce_weight,
            'dice_weight': dice_weight
        }
    
    return combined_loss


def compute_loss_with_deep_supervision(outputs, masks, loss_fn, batch_idx):
    """计算包含深度监督的损失
    
    参数:
        outputs: 模型输出，可能是单个输出或包含深度监督的元组
        masks: 真实标签
        loss_fn: 损失函数
        batch_idx: 当前批次索引
        
    返回:
        dict: 包含各项损失的字典
    """
    if isinstance(outputs, tuple):
        # 深度监督模式：outputs = (主输出, 辅助输出1, 辅助输出2, ...)
        main_output = outputs[0]
        auxiliary_outputs = outputs[1:]
        
        # 计算主输出的损失
        main_loss_dict = loss_fn(main_output, masks)
        total_loss = main_loss_dict['total_loss']
        
        # 添加辅助输出的损失，权重递减
        aux_losses = []
        for i, aux_output in enumerate(auxiliary_outputs):
            # 权重系数：0.5, 0.25, 0.125, ...
            weight = 0.5 ** (i + 1)
            aux_loss_dict = loss_fn(aux_output, masks)
            aux_loss = aux_loss_dict['total_loss']
            total_loss += weight * aux_loss
            aux_losses.append(aux_loss.item())
            
        if batch_idx < 3:
            print(f"批次 {batch_idx + 1} - 深度监督: 主损失={main_loss_dict['total_loss']:.4f}, "
                  f"辅助损失={aux_losses}, 辅助输出数={len(auxiliary_outputs)}")
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss_dict['total_loss'],
            'aux_losses': aux_losses,
            'base_loss': main_loss_dict['base_loss'],
            'dice_loss': main_loss_dict['dice_loss']
        }
    else:
        # 标准模式：只有主输出
        loss_dict = loss_fn(outputs, masks)
        return loss_dict


def train_epoch(model, dataloader, optimizer, loss_fn, device, scaler=None, use_amp=False, epoch_num=0, args=None):
    """执行一个训练周期
    
    在一个epoch中遍历所有训练数据，执行前向传播、损失计算、反向传播和参数更新。
    支持混合精度训练、深度监督机制、梯度累积和梯度裁剪。
    
    训练流程：
    1. 设置模型为训练模式
    2. 遍历数据批次
    3. 前向传播计算预测结果
    4. 计算损失（包括深度监督损失）
    5. 反向传播计算梯度（支持梯度累积）
    6. 梯度裁剪（可选）
    7. 更新模型参数
    8. 累计损失统计和性能监控
    
    参数:
        model (nn.Module): 要训练的3D U-Net模型
        dataloader (DataLoader): 训练数据加载器
        optimizer (Optimizer): 优化器
        loss_fn (function): 损失函数
        device (torch.device): 计算设备（CPU或GPU）
        scaler (GradScaler, 可选): 混合精度训练的梯度缩放器
        use_amp (bool): 是否使用自动混合精度训练
        epoch_num (int): 当前epoch编号
        args: 命令行参数，包含训练配置
        
    返回:
        dict: 包含训练统计信息的字典
    """
    # 设置模型为训练模式，启用dropout和batch normalization的训练行为
    model.train()
    
    # 初始化统计变量
    epoch_stats = {
        'total_loss': 0.0,
        'base_loss': 0.0,
        'dice_loss': 0.0,
        'num_batches': len(dataloader),
        'batch_times': [],
        'data_load_times': [],
        'forward_times': [],
        'backward_times': [],
        'gpu_memory_usage': []
    }
    
    # 梯度累积相关变量
    accumulation_steps = getattr(args, 'accumulation_steps', 1)
    effective_batch_size = args.batch_size * accumulation_steps
    
    print(f"\n{'='*60}")
    print(f"开始训练 Epoch {epoch_num + 1}")
    print(f"总批次数: {epoch_stats['num_batches']}")
    print(f"使用设备: {device}")
    print(f"混合精度训练: {'启用' if use_amp else '禁用'}")
    print(f"梯度累积步数: {accumulation_steps}")
    print(f"有效批次大小: {effective_batch_size}")
    if hasattr(args, 'gradient_clipping') and args.gradient_clipping > 0:
        print(f"梯度裁剪: {args.gradient_clipping}")
    print(f"{'='*60}")
    
    epoch_start_time = time.time()
    
    # 创建训练进度条
    train_pbar = tqdm(
        enumerate(dataloader), 
        total=epoch_stats['num_batches'],
        desc=f"Epoch {epoch_num + 1} - 训练",
        unit="batch",
        ncols=140,
        leave=False
    )
    
    # 遍历训练数据批次
    for batch_idx, batch in train_pbar:
        batch_start_time = time.time()
        
        # ==================== 数据准备 ====================
        data_start_time = time.time()
        # 将数据移动到指定设备（GPU或CPU）
        images = batch['image'].to(device, non_blocking=True)  # [B, 4, D, H, W]
        masks = batch['mask'].to(device, non_blocking=True)    # [B, D, H, W]
        data_load_time = time.time() - data_start_time
        epoch_stats['data_load_times'].append(data_load_time)
        
        # 打印数据形状信息（前几个批次）
        if batch_idx < 3:
            print(f"批次 {batch_idx + 1} - 数据形状: images={list(images.shape)}, masks={list(masks.shape)}")
            print(f"批次 {batch_idx + 1} - 数据加载时间: {data_load_time:.3f}s")
        
        # 梯度累积：只在累积步数的开始清零梯度
        if batch_idx % accumulation_steps == 0:
            optimizer.zero_grad()
        
        # ==================== 前向传播 ====================
        forward_start_time = time.time()
        
        if use_amp:
            # 混合精度训练路径
            with autocast('cuda'):
                # 模型前向传播
                outputs = model(images)
                
                # 处理深度监督输出并计算损失
                loss_dict = compute_loss_with_deep_supervision(outputs, masks, loss_fn, batch_idx)
                loss = loss_dict['total_loss']
                
                # 梯度累积：按累积步数缩放损失
                loss = loss / accumulation_steps
            
            forward_time = time.time() - forward_start_time
            epoch_stats['forward_times'].append(forward_time)
            
            # ==================== 反向传播（混合精度）====================
            backward_start_time = time.time()
            
            # 缩放损失以防止梯度下溢
            scaler.scale(loss).backward()
            
            # 梯度累积：只在累积步数结束时更新参数
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == epoch_stats['num_batches']:
                # 梯度裁剪
                if hasattr(args, 'gradient_clipping') and args.gradient_clipping > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clipping)
                
                # 更新参数
                scaler.step(optimizer)
                scaler.update()
            
            backward_time = time.time() - backward_start_time
            epoch_stats['backward_times'].append(backward_time)
            
        else:
            # ==================== 标准精度训练路径 ====================
            # 模型前向传播
            outputs = model(images)
            
            # 处理深度监督输出并计算损失
            loss_dict = compute_loss_with_deep_supervision(outputs, masks, loss_fn, batch_idx)
            loss = loss_dict['total_loss']
            
            # 梯度累积：按累积步数缩放损失
            loss = loss / accumulation_steps
            
            forward_time = time.time() - forward_start_time
            epoch_stats['forward_times'].append(forward_time)
            
            # ==================== 反向传播（标准精度）====================
            backward_start_time = time.time()
            
            loss.backward()
            
            # 梯度累积：只在累积步数结束时更新参数
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == epoch_stats['num_batches']:
                # 梯度裁剪
                if hasattr(args, 'gradient_clipping') and args.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clipping)
                
                # 更新参数
                optimizer.step()
            
            backward_time = time.time() - backward_start_time
            epoch_stats['backward_times'].append(backward_time)
        
        # ==================== 损失统计和进度显示 ====================
        # 累计当前批次的损失（注意：这里的loss已经被累积步数缩放过）
        actual_loss = loss.item() * accumulation_steps  # 恢复真实损失值用于统计
        epoch_stats['total_loss'] += actual_loss
        
        # 累计详细损失信息
        if 'base_loss' in loss_dict:
            epoch_stats['base_loss'] += loss_dict['base_loss'].item()
        if 'dice_loss' in loss_dict:
            epoch_stats['dice_loss'] += loss_dict['dice_loss'].item()
        
        batch_time = time.time() - batch_start_time
        epoch_stats['batch_times'].append(batch_time)
        
        # GPU内存使用情况
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(device) / 1024**3  # GB
            epoch_stats['gpu_memory_usage'].append(gpu_memory)
            memory_info = f"GPU:{gpu_memory:.1f}GB"
        else:
            memory_info = "CPU"
        
        # 更新进度条
        avg_batch_time = sum(epoch_stats['batch_times']) / len(epoch_stats['batch_times'])
        current_avg_loss = epoch_stats['total_loss'] / (batch_idx + 1)
        
        # 更新进度条描述
        train_pbar.set_postfix({
            'Loss': f'{actual_loss:.4f}',
            'AvgLoss': f'{current_avg_loss:.4f}',
            'Time': f'{batch_time:.2f}s',
            'Memory': memory_info,
            'LR': f'{optimizer.param_groups[0]["lr"]:.1e}',
            'Accum': f'{(batch_idx % accumulation_steps) + 1}/{accumulation_steps}'
        })
        
        # 详细信息输出（前几个批次和关键节点）
        log_freq = getattr(args, 'log_freq', 10)
        if batch_idx < 3 or batch_idx % max(1, epoch_stats['num_batches'] // log_freq) == 0:
            eta_seconds = avg_batch_time * (epoch_stats['num_batches'] - batch_idx - 1)
            eta_minutes = eta_seconds / 60
            
            # 构建详细的损失信息
            loss_info = f"总损失={actual_loss:.6f}"
            if 'base_loss' in loss_dict:
                loss_info += f", 基础损失={loss_dict['base_loss'].item():.6f}"
            if 'dice_loss' in loss_dict:
                loss_info += f", Dice损失={loss_dict['dice_loss'].item():.6f}"
            
            tqdm.write(f"[详细] 批次 {batch_idx+1}: {loss_info}, "
                      f"数据加载={epoch_stats['data_load_times'][-1]:.3f}s, "
                      f"前向={epoch_stats['forward_times'][-1]:.3f}s, "
                      f"反向={epoch_stats['backward_times'][-1]:.3f}s, "
                      f"ETA={eta_minutes:.1f}min")
    
    # 关闭训练进度条
    train_pbar.close()
    
    # Epoch结束统计
    epoch_time = time.time() - epoch_start_time
    num_batches = epoch_stats['num_batches']
    
    # 计算平均损失
    avg_total_loss = epoch_stats['total_loss'] / num_batches
    avg_base_loss = epoch_stats['base_loss'] / num_batches if epoch_stats['base_loss'] > 0 else 0
    avg_dice_loss = epoch_stats['dice_loss'] / num_batches if epoch_stats['dice_loss'] > 0 else 0
    
    # 计算平均时间
    avg_batch_time = sum(epoch_stats['batch_times']) / len(epoch_stats['batch_times'])
    avg_data_time = sum(epoch_stats['data_load_times']) / len(epoch_stats['data_load_times'])
    avg_forward_time = sum(epoch_stats['forward_times']) / len(epoch_stats['forward_times'])
    avg_backward_time = sum(epoch_stats['backward_times']) / len(epoch_stats['backward_times'])
    
    # GPU内存统计
    if epoch_stats['gpu_memory_usage']:
        avg_gpu_memory = sum(epoch_stats['gpu_memory_usage']) / len(epoch_stats['gpu_memory_usage'])
        max_gpu_memory = max(epoch_stats['gpu_memory_usage'])
    else:
        avg_gpu_memory = max_gpu_memory = 0
    
    print(f"\n{'='*80}")
    print(f"Epoch {epoch_num + 1} 训练完成!")
    print(f"{'='*80}")
    print(f"时间统计:")
    print(f"  总时间: {epoch_time:.2f}s ({epoch_time/60:.1f}min)")
    print(f"  平均批次时间: {avg_batch_time:.3f}s")
    print(f"  平均数据加载时间: {avg_data_time:.3f}s ({avg_data_time/avg_batch_time*100:.1f}%)")
    print(f"  平均前向传播时间: {avg_forward_time:.3f}s ({avg_forward_time/avg_batch_time*100:.1f}%)")
    print(f"  平均反向传播时间: {avg_backward_time:.3f}s ({avg_backward_time/avg_batch_time*100:.1f}%)")
    print(f"损失统计:")
    print(f"  平均总损失: {avg_total_loss:.6f}")
    if avg_base_loss > 0:
        print(f"  平均基础损失: {avg_base_loss:.6f}")
    if avg_dice_loss > 0:
        print(f"  平均Dice损失: {avg_dice_loss:.6f}")
    if avg_gpu_memory > 0:
        print(f"GPU内存统计:")
        print(f"  平均使用: {avg_gpu_memory:.1f}GB")
        print(f"  峰值使用: {max_gpu_memory:.1f}GB")
    print(f"{'='*80}\n")
    
    # 返回详细的训练统计信息
    return {
        'avg_total_loss': avg_total_loss,
        'avg_base_loss': avg_base_loss,
        'avg_dice_loss': avg_dice_loss,
        'epoch_time': epoch_time,
        'avg_batch_time': avg_batch_time,
        'avg_gpu_memory': avg_gpu_memory,
        'max_gpu_memory': max_gpu_memory
    }


def validate(model, dataloader, loss_fn, device, args):
    """模型验证评估
    
    在验证集上评估模型性能，计算损失和关键评估指标。
    验证过程不更新模型参数，用于监控训练进度和防止过拟合。
    
    评估指标：
    1. 验证损失：与训练损失相同的损失函数
    2. Dice系数：评估分割重叠度，值越高越好
    3. Hausdorff距离：评估边界准确性，值越小越好
    
    验证流程：
    1. 设置模型为评估模式
    2. 禁用梯度计算以节省内存
    3. 遍历验证数据
    4. 计算预测结果和评估指标
    5. 统计平均性能
    
    参数:
        model (nn.Module): 要评估的模型
        dataloader (DataLoader): 验证数据加载器
        loss_fn (function): 损失函数
        device (torch.device): 计算设备
        args (Namespace): 包含模型配置的参数对象
        
    返回:
        tuple: (平均验证损失, 平均Dice系数, 平均Hausdorff距离)
    """
    # 设置模型为评估模式，禁用dropout和batch normalization的随机性
    model.eval()
    
    # 初始化统计变量
    val_loss = 0.0
    dice_scores = []
    hausdorff_distances = []
    num_batches = len(dataloader)
    
    # 创建验证进度条
    val_pbar = tqdm(
        enumerate(dataloader),
        total=num_batches,
        desc="验证中",
        unit="batch",
        ncols=100,
        leave=False
    )
    
    # 禁用梯度计算，节省内存并加速推理
    with torch.no_grad():
        for batch_idx, batch in val_pbar:
            # ==================== 数据准备 ====================
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            
            # ==================== 模型推理 ====================
            outputs = model(images)
            
            # 处理深度监督输出，只使用主输出进行评估
            if isinstance(outputs, tuple):
                main_output = outputs[0]  # 使用主输出
                outputs = main_output
            
            # ==================== 损失计算 ====================
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()
            
            # ==================== 评估指标计算 ====================
            try:
                # 计算多类别Dice系数
                class_dice_scores, avg_dice = multiclass_dice_coefficient(
                    outputs, masks, num_classes=args.out_channels
                )
                dice_scores.append(avg_dice.item())
                
                # 计算多类别Hausdorff距离
                class_hausdorff_distances, avg_hausdorff = multiclass_hausdorff_distance(
                    outputs, masks, num_classes=args.out_channels
                )
                hausdorff_distances.append(avg_hausdorff)
                
            except Exception as e:
                print(f"警告：批次 {batch_idx} 的指标计算失败: {str(e)}")
                # 如果指标计算失败，使用默认值
                dice_scores.append(0.0)
                hausdorff_distances.append(float('inf'))
            
            # 更新验证进度条
            current_dice = dice_scores[-1] if dice_scores else 0.0
            current_hausdorff = hausdorff_distances[-1] if hausdorff_distances else float('inf')
            
            val_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{current_dice:.4f}',
                'HD': f'{current_hausdorff:.2f}' if current_hausdorff != float('inf') else 'inf'
            })
    
    # 关闭验证进度条
    val_pbar.close()
    
    # ==================== 统计结果 ====================
    # 计算平均验证损失
    avg_val_loss = val_loss / num_batches
    
    # 计算平均Dice系数
    if dice_scores:
        avg_dice = sum(dice_scores) / len(dice_scores)
    else:
        avg_dice = 0.0
        print("警告：没有有效的Dice分数")
    
    # 计算平均Hausdorff距离
    if hausdorff_distances:
        # 过滤掉无穷大值
        valid_hausdorff = [h for h in hausdorff_distances if h != float('inf')]
        if valid_hausdorff:
            avg_hausdorff = sum(valid_hausdorff) / len(valid_hausdorff)
        else:
            avg_hausdorff = float('inf')
            print("警告：所有Hausdorff距离都是无穷大")
    else:
        avg_hausdorff = float('inf')
        print("警告：没有有效的Hausdorff距离")
    
    return avg_val_loss, avg_dice, avg_hausdorff


def save_checkpoint(model, optimizer, scheduler, epoch, best_dice, patience_counter, args, filename):
    """保存检查点
    
    参数:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前epoch
        best_dice: 最佳Dice分数
        patience_counter: 早停计数器
        args: 训练参数
        filename: 保存文件名
    """
    if args.distributed:
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    state = {
        'epoch': epoch,
        'model_state': model_state,
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_dice': best_dice,
        'patience_counter': patience_counter,
        'args': args,
        'pytorch_version': torch.__version__
    }
    
    torch.save(state, filename)
    print(f"检查点已保存: {filename}")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, args):
    """加载检查点恢复训练
    
    参数:
        checkpoint_path: 检查点文件路径
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        args: 训练参数
        
    返回:
        tuple: (start_epoch, best_dice, patience_counter)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    print(f"正在加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载模型状态
    if args.distributed:
        model.module.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint['model_state'])
    
    # 加载优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    # 加载调度器状态
    if 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    # 获取训练状态
    start_epoch = checkpoint['epoch'] + 1
    best_dice = checkpoint.get('best_dice', 0)
    patience_counter = checkpoint.get('patience_counter', 0)
    
    print(f"检查点加载完成 - Epoch: {start_epoch}, Best Dice: {best_dice:.4f}")
    
    return start_epoch, best_dice, patience_counter


def load_pretrained_model(pretrained_path, model, args):
    """加载预训练模型
    
    参数:
        pretrained_path: 预训练模型路径
        model: 模型
        args: 训练参数
    """
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"预训练模型文件不存在: {pretrained_path}")
    
    print(f"正在加载预训练模型: {pretrained_path}")
    
    # 尝试加载检查点格式
    try:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'model_state' in checkpoint:
            pretrained_dict = checkpoint['model_state']
        else:
            pretrained_dict = checkpoint
    except:
        # 直接加载状态字典
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    
    # 获取当前模型的状态字典
    if args.distributed:
        model_dict = model.module.state_dict()
    else:
        model_dict = model.state_dict()
    
    # 过滤掉不匹配的键
    filtered_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered_dict[k] = v
        else:
            print(f"跳过不匹配的参数: {k}")
    
    # 更新模型参数
    model_dict.update(filtered_dict)
    
    if args.distributed:
        model.module.load_state_dict(model_dict)
    else:
        model.load_state_dict(model_dict)
    
    print(f"预训练模型加载完成，成功加载 {len(filtered_dict)}/{len(pretrained_dict)} 个参数")


def save_training_config(args):
    """保存训练配置到文件
    
    参数:
        args: 训练参数
    """
    import json
    
    config_dict = vars(args).copy()
    
    # 转换不能JSON序列化的对象
    for key, value in config_dict.items():
        if isinstance(value, (list, tuple)):
            config_dict[key] = list(value)
        elif not isinstance(value, (str, int, float, bool, type(None))):
            config_dict[key] = str(value)
    
    config_path = os.path.join(args.output_dir, 'training_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"训练配置已保存: {config_path}")


def main():
    args = parse_args()
    
    print("="*100)
    print("3D U-Net BraTS2021 训练开始")
    print("="*100)
    
    # 打印训练配置
    print("训练配置:")
    print(f"  数据目录: {args.data_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  学习率: {args.lr}")
    print(f"  目标形状: {args.target_shape}")
    print(f"  混合精度: {'启用' if args.amp else '禁用'}")
    print(f"  深度监督: {'启用' if args.deep_supervision else '禁用'}")
    print(f"  残差连接: {'启用' if args.residual else '禁用'}")
    print(f"  工作进程: {args.num_workers}")
    print("-"*50)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    print(f"输出目录已创建: {args.output_dir}")
    
    # 保存训练配置
    if args.rank == 0:
        save_training_config(args)
    
    # 设置分布式训练
    setup_distributed(args)
    
    # 设置设备
    if torch.cuda.is_available():
        if args.local_rank >= 0:
            device = torch.device(f'cuda:{args.local_rank}')
        else:
            device = torch.device('cuda')
        
        # 打印GPU信息
        print(f"使用设备: {device}")
        print(f"GPU名称: {torch.cuda.get_device_name(device)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"PyTorch版本: {torch.__version__}")
    else:
        device = torch.device('cpu')
        print("使用设备: CPU")
    print("-"*50)
    
    # 获取数据加载器
    print("正在加载数据...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_shape=args.target_shape
    )
    
    print("数据加载完成:")
    print(f"  训练集: {len(train_loader.dataset)} 个样本, {len(train_loader)} 个批次")
    print(f"  验证集: {len(val_loader.dataset)} 个样本, {len(val_loader)} 个批次")
    print(f"  测试集: {len(test_loader.dataset)} 个样本, {len(test_loader)} 个批次")
    print("-"*50)
    
    # 创建模型
    print("正在创建模型...")
    model = get_model(args).to(device)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("模型信息:")
    print(f"  模型类型: 3D U-Net")
    print(f"  输入通道: {args.in_channels}")
    print(f"  输出通道: {args.out_channels}")
    print(f"  特征层级: {args.features}")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024**2:.1f} MB (FP32)")
    print("-"*50)
    
    # 分布式训练设置
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    # 创建优化器
    optimizer = get_optimizer(model, args)
    
    # 创建学习率调度器
    scheduler = get_scheduler(optimizer, args)
    
    # 获取损失函数
    loss_fn = get_loss_function(args)
    
    # 创建混合精度训练的缩放器
    scaler = GradScaler() if args.amp else None
    
    # 创建TensorBoard写入器
    if args.rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    # 初始化训练状态
    start_epoch = 0
    best_dice = 0
    patience_counter = 0
    
    # 加载预训练模型或恢复训练
    if args.resume:
        start_epoch, best_dice, patience_counter = load_checkpoint(
            args.resume, model, optimizer, scheduler, args
        )
        print(f"从检查点恢复训练: epoch {start_epoch}, best_dice {best_dice:.4f}")
    elif args.pretrained:
        load_pretrained_model(args.pretrained, model, args)
        print(f"加载预训练模型: {args.pretrained}")
    
    # 如果只进行测试
    if args.test_only:
        print("仅进行测试评估...")
        test_loss, test_dice, test_hausdorff = validate(model, test_loader, loss_fn, device, args)
        print(f"测试结果 - 损失: {test_loss:.6f}, Dice: {test_dice:.6f}, Hausdorff: {test_hausdorff:.4f}")
        return
    
    # 创建总体训练进度条
    epoch_pbar = tqdm(
        range(start_epoch, args.epochs),
        desc="总体训练进度",
        unit="epoch",
        ncols=120,
        position=0,
        initial=start_epoch,
        total=args.epochs
    )
    
    for epoch in epoch_pbar:
        # 训练一个epoch
        train_stats = train_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler, args.amp, epoch, args
        )
        train_loss = train_stats['avg_total_loss']
        
        # 验证（根据验证频率）
        if (epoch + 1) % args.val_freq == 0:
            val_loss, val_dice, val_hausdorff = validate(model, val_loader, loss_fn, device, args)
            
            # 更新学习率
            if args.scheduler.lower() == 'plateau':
                scheduler.step(val_dice)
            else:
                scheduler.step()
        else:
            # 如果不进行验证，使用上一次的验证结果
            val_loss = val_loss if 'val_loss' in locals() else float('inf')
            val_dice = val_dice if 'val_dice' in locals() else 0.0
            val_hausdorff = val_hausdorff if 'val_hausdorff' in locals() else float('inf')
            
            # 对于非plateau调度器，仍需要step
            if args.scheduler.lower() != 'plateau':
                scheduler.step()
        
        # 更新总体进度条
        epoch_pbar.set_postfix({
            'TrainLoss': f'{train_loss:.4f}',
            'ValLoss': f'{val_loss:.4f}',
            'ValDice': f'{val_dice:.4f}',
            'BestDice': f'{best_dice:.4f}',
            'Patience': f'{patience_counter}/{args.patience}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.1e}'
        })
        
        # 记录指标
        if args.rank == 0:
            # 使用tqdm.write来避免与进度条冲突
            tqdm.write(f"\n{'*'*80}")
            tqdm.write(f"EPOCH {epoch+1}/{args.epochs} 总结")
            tqdm.write(f"{'*'*80}")
            tqdm.write(f"训练损失:     {train_loss:.6f}")
            tqdm.write(f"验证损失:     {val_loss:.6f}")
            tqdm.write(f"验证Dice系数: {val_dice:.6f}")
            tqdm.write(f"验证Hausdorff: {val_hausdorff:.4f}")
            tqdm.write(f"当前学习率:   {optimizer.param_groups[0]['lr']:.2e}")
            tqdm.write(f"最佳Dice:     {best_dice:.6f}")
            tqdm.write(f"早停计数:     {patience_counter}/{args.patience}")
            
            # 性能趋势分析
            if epoch > 0:
                tqdm.write(f"\n性能变化:")
                if val_dice > best_dice:
                    tqdm.write(f"  ✓ Dice系数提升: {val_dice - best_dice:+.6f}")
                else:
                    tqdm.write(f"  ✗ Dice系数下降: {val_dice - best_dice:+.6f}")
            
            tqdm.write(f"{'*'*80}\n")
            
            # 记录训练指标
            writer.add_scalar('Loss/train_total', train_loss, epoch)
            if 'avg_base_loss' in train_stats and train_stats['avg_base_loss'] > 0:
                writer.add_scalar('Loss/train_base', train_stats['avg_base_loss'], epoch)
            if 'avg_dice_loss' in train_stats and train_stats['avg_dice_loss'] > 0:
                writer.add_scalar('Loss/train_dice', train_stats['avg_dice_loss'], epoch)
            
            # 记录验证指标
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/dice', val_dice, epoch)
            writer.add_scalar('Metrics/hausdorff', val_hausdorff, epoch)
            
            # 记录学习率和性能指标
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Performance/epoch_time', train_stats['epoch_time'], epoch)
            writer.add_scalar('Performance/avg_batch_time', train_stats['avg_batch_time'], epoch)
            
            # 记录GPU内存使用
            if train_stats['avg_gpu_memory'] > 0:
                writer.add_scalar('Memory/avg_gpu_usage', train_stats['avg_gpu_memory'], epoch)
                writer.add_scalar('Memory/max_gpu_usage', train_stats['max_gpu_memory'], epoch)
            
            # 可视化预测结果
            if (epoch + 1) % args.vis_freq == 0:
                with torch.no_grad():
                    # 获取一个批次的验证数据
                    val_batch = next(iter(val_loader))
                    val_images = val_batch['image'].to(device)
                    val_masks = val_batch['mask'].to(device)
                    
                    # 获取预测结果
                    val_outputs = model(val_images)
                    if isinstance(val_outputs, tuple):
                        val_outputs = val_outputs[0]
                    
                    # 保存可视化结果
                    save_prediction_visualization(
                        val_images, val_masks, val_outputs,
                        os.path.join(args.output_dir, 'visualizations', f'epoch_{epoch+1}.png')
                    )
        
        # 保存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            
            if args.rank == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_dice, patience_counter, args,
                    os.path.join(args.output_dir, 'checkpoints', 'best_model.pth')
                )
                tqdm.write(f'✓ 新的最佳模型已保存，Dice: {best_dice:.4f}')
        else:
            patience_counter += 1
        
        # 保存定期检查点
        if args.rank == 0 and (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_dice, patience_counter, args,
                os.path.join(args.output_dir, 'checkpoints', f'model_epoch_{epoch+1}.pth')
            )
        
        # 保存最新检查点（用于意外中断后恢复）
        if args.rank == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_dice, patience_counter, args,
                os.path.join(args.output_dir, 'checkpoints', 'latest.pth')
            )
        
        # 早停
        if patience_counter >= args.patience:
            tqdm.write(f'⚠ 早停触发，在第 {epoch+1} 轮停止训练')
            break
    
    # 关闭进度条和TensorBoard写入器
    epoch_pbar.close()
    if args.rank == 0:
        writer.close()
    
    # 在测试集上评估最佳模型
    if args.rank == 0:
        print('\n' + '='*60)
        print('正在评估最佳模型...')
        print('='*60)
        
        # 加载最佳模型
        checkpoint = torch.load(os.path.join(args.output_dir, 'checkpoints', 'best_model.pth'))
        
        if args.distributed:
            model.module.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint['model_state'])
        
        # 在测试集上评估
        test_loss, test_dice, test_hausdorff = validate(model, test_loader, loss_fn, device, args)
        
        print('\n' + '='*60)
        print('最终测试结果:')
        print('='*60)
        print(f'测试损失:      {test_loss:.6f}')
        print(f'测试Dice系数:  {test_dice:.6f}')
        print(f'测试Hausdorff: {test_hausdorff:.4f}')
        print('='*60)
        print('训练完成! 🎉')
        print('='*60)


if __name__ == '__main__':
    main()