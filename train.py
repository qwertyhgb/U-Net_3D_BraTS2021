import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast  # 用于混合精度训练
import torch.distributed as dist  # 分布式训练
from torch.nn.parallel import DistributedDataParallel as DDP  # 分布式数据并行
from torch.utils.data.distributed import DistributedSampler  # 分布式采样器
from torch.utils.tensorboard import SummaryWriter  # TensorBoard可视化

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
    parser.add_argument('--batch_size', type=int, default=2,
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
    
    # 分布式训练参数
    parser.add_argument('--distributed', action='store_true',
                        help='是否使用分布式训练，多GPU训练')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='分布式训练的本地排名，由torch.distributed.launch设置')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器的工作进程数，用于并行数据加载')
    
    # 混合精度训练
    parser.add_argument('--amp', action='store_true', default=True,
                        help='是否使用混合精度训练，可加速训练并减少内存使用')
    
    # 可视化参数
    parser.add_argument('--vis_freq', type=int, default=10,
                        help='可视化频率（每n个epoch保存一次可视化结果）')
    
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


def get_loss_function():
    """构建组合损失函数
    
    医学图像分割任务中，单一损失函数往往无法很好地处理类别不平衡和边界精确性问题。
    本函数构建了一个组合损失函数，结合了交叉熵损失和Dice损失的优势：
    
    1. 交叉熵损失（Cross-Entropy Loss）：
       - 优势：训练稳定，收敛快，对类别不平衡有一定鲁棒性
       - 劣势：主要关注像素级分类，对分割区域的整体性考虑不足
    
    2. Dice损失（Dice Loss）：
       - 优势：直接优化Dice系数，关注分割区域的重叠度，对小目标友好
       - 劣势：训练初期可能不稳定，梯度可能较小
    
    组合策略：Loss = CrossEntropy + Dice，平衡像素级准确性和区域级完整性
    
    返回:
        function: 组合损失函数，接受预测和真实标签，返回损失值
    """
    
    def dice_loss(y_pred, y_true, smooth=1e-6):
        """计算单个类别的Dice损失
        
        Dice损失 = 1 - Dice系数
        Dice系数 = 2 * |预测∩真实| / (|预测| + |真实|)
        
        参数:
            y_pred (Tensor): 预测概率，形状为 [N]，值范围 [0,1]
            y_true (Tensor): 真实标签，形状为 [N]，值为 0 或 1
            smooth (float): 平滑项，防止分母为0，提高数值稳定性
            
        返回:
            Tensor: Dice损失值
        """
        # 展平张量，便于计算
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)
        
        # 计算交集：预测为正且真实为正的像素数
        intersection = (y_pred * y_true).sum()
        
        # 计算Dice系数，然后转换为损失（1 - Dice）
        dice_coeff = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
        dice_loss_value = 1 - dice_coeff
        
        return dice_loss_value
    
    def combined_loss(y_pred, y_true):
        """组合损失函数：交叉熵 + Dice损失
        
        参数:
            y_pred (Tensor): 模型预测logits，形状为 [B, C, D, H, W]
            y_true (Tensor): 真实分割标签，形状为 [B, D, H, W]
            
        返回:
            Tensor: 组合损失值
        """
        # ==================== 交叉熵损失 ====================
        # 处理类别不平衡，可以考虑加权交叉熵
        # 这里使用标准交叉熵，对所有类别等权重处理
        ce_loss = nn.CrossEntropyLoss()(y_pred, y_true)
        
        # ==================== Dice损失 ====================
        # 将预测logits转换为概率分布
        y_pred_softmax = torch.softmax(y_pred, dim=1)  # [B, C, D, H, W]
        
        # 将真实标签转换为one-hot编码
        num_classes = y_pred.shape[1]
        y_true_one_hot = torch.zeros_like(y_pred_softmax)
        y_true_one_hot = y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1)
        
        # 计算每个前景类别的Dice损失（跳过背景类别）
        dice_loss_total = 0
        num_foreground_classes = 0
        
        for class_idx in range(1, num_classes):  # 从1开始，跳过背景类别（索引0）
            # 提取当前类别的预测概率和真实标签
            pred_class = y_pred_softmax[:, class_idx]  # [B, D, H, W]
            true_class = y_true_one_hot[:, class_idx]  # [B, D, H, W]
            
            # 只有当真实标签中存在该类别时才计算损失
            if true_class.sum() > 0:
                dice_loss_total += dice_loss(pred_class, true_class)
                num_foreground_classes += 1
        
        # 计算平均Dice损失
        if num_foreground_classes > 0:
            avg_dice_loss = dice_loss_total / num_foreground_classes
        else:
            # 如果没有前景类别，Dice损失为0
            avg_dice_loss = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        
        # ==================== 组合损失 ====================
        # 将交叉熵损失和Dice损失相加
        # 可以考虑添加权重系数来平衡两种损失的贡献
        total_loss = ce_loss + avg_dice_loss
        
        return total_loss
    
    return combined_loss


def train_epoch(model, dataloader, optimizer, loss_fn, device, scaler=None, use_amp=False):
    """执行一个训练周期
    
    在一个epoch中遍历所有训练数据，执行前向传播、损失计算、反向传播和参数更新。
    支持混合精度训练和深度监督机制。
    
    训练流程：
    1. 设置模型为训练模式
    2. 遍历数据批次
    3. 前向传播计算预测结果
    4. 计算损失（包括深度监督损失）
    5. 反向传播计算梯度
    6. 更新模型参数
    7. 累计损失统计
    
    参数:
        model (nn.Module): 要训练的3D U-Net模型
        dataloader (DataLoader): 训练数据加载器
        optimizer (Optimizer): 优化器（如Adam）
        loss_fn (function): 损失函数
        device (torch.device): 计算设备（CPU或GPU）
        scaler (GradScaler, 可选): 混合精度训练的梯度缩放器
        use_amp (bool): 是否使用自动混合精度训练
        
    返回:
        float: 当前epoch的平均训练损失
    """
    # 设置模型为训练模式，启用dropout和batch normalization的训练行为
    model.train()
    
    # 初始化损失累计器
    epoch_loss = 0.0
    num_batches = len(dataloader)
    
    # 遍历训练数据批次
    for batch_idx, batch in enumerate(dataloader):
        # ==================== 数据准备 ====================
        # 将数据移动到指定设备（GPU或CPU）
        images = batch['image'].to(device, non_blocking=True)  # [B, 4, D, H, W]
        masks = batch['mask'].to(device, non_blocking=True)    # [B, D, H, W]
        
        # 清零梯度，防止梯度累积
        optimizer.zero_grad()
        
        # ==================== 前向传播 ====================
        if use_amp:
            # 混合精度训练路径
            with autocast():
                # 模型前向传播
                outputs = model(images)
                
                # 处理深度监督输出
                if isinstance(outputs, tuple):
                    # 深度监督模式：outputs = (主输出, 辅助输出1, 辅助输出2, ...)
                    main_output = outputs[0]
                    auxiliary_outputs = outputs[1:]
                    
                    # 计算主输出的损失
                    loss = loss_fn(main_output, masks)
                    
                    # 添加辅助输出的损失，权重递减
                    for i, aux_output in enumerate(auxiliary_outputs):
                        # 权重系数：0.5, 0.25, 0.125, ...
                        weight = 0.5 ** (i + 1)
                        aux_loss = loss_fn(aux_output, masks)
                        loss += weight * aux_loss
                else:
                    # 标准模式：只有主输出
                    loss = loss_fn(outputs, masks)
            
            # ==================== 反向传播（混合精度）====================
            # 缩放损失以防止梯度下溢
            scaler.scale(loss).backward()
            
            # 更新参数
            scaler.step(optimizer)
            
            # 更新缩放因子
            scaler.update()
            
        else:
            # ==================== 标准精度训练路径 ====================
            # 模型前向传播
            outputs = model(images)
            
            # 处理深度监督输出
            if isinstance(outputs, tuple):
                # 深度监督模式
                main_output = outputs[0]
                auxiliary_outputs = outputs[1:]
                
                # 计算主输出的损失
                loss = loss_fn(main_output, masks)
                
                # 添加辅助输出的损失，权重递减
                for i, aux_output in enumerate(auxiliary_outputs):
                    weight = 0.5 ** (i + 1)
                    aux_loss = loss_fn(aux_output, masks)
                    loss += weight * aux_loss
            else:
                # 标准模式
                loss = loss_fn(outputs, masks)
            
            # ==================== 反向传播（标准精度）====================
            loss.backward()
            
            # 可选：梯度裁剪，防止梯度爆炸
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
        
        # ==================== 损失统计 ====================
        # 累计当前批次的损失
        epoch_loss += loss.item()
        
        # 可选：打印批次进度（每10%打印一次）
        if batch_idx % (num_batches // 10 + 1) == 0:
            progress = 100.0 * batch_idx / num_batches
            print(f'训练进度: {progress:.1f}% ({batch_idx}/{num_batches}), '
                  f'当前批次损失: {loss.item():.4f}')
    
    # 计算并返回平均损失
    avg_epoch_loss = epoch_loss / num_batches
    return avg_epoch_loss


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
    
    # 禁用梯度计算，节省内存并加速推理
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
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
            
            # 可选：打印验证进度
            if batch_idx % (num_batches // 5 + 1) == 0:
                progress = 100.0 * batch_idx / num_batches
                current_dice = dice_scores[-1] if dice_scores else 0.0
                print(f'验证进度: {progress:.1f}% ({batch_idx}/{num_batches}), '
                      f'当前Dice: {current_dice:.4f}')
    
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


def save_checkpoint(model, optimizer, epoch, best_dice, args, filename):
    """保存检查点"""
    if args.distributed:
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    state = {
        'epoch': epoch,
        'model_state': model_state,
        'optimizer_state': optimizer.state_dict(),
        'best_dice': best_dice,
        'args': args
    }
    
    torch.save(state, filename)


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    # 设置分布式训练
    setup_distributed(args)
    
    # 设置设备
    device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_shape=args.target_shape
    )
    
    # 创建模型
    model = get_model(args).to(device)
    
    # 分布式训练设置
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # 获取损失函数
    loss_fn = get_loss_function()
    
    # 创建混合精度训练的缩放器
    scaler = GradScaler() if args.amp else None
    
    # 创建TensorBoard写入器
    if args.rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    # 训练循环
    best_dice = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # 训练一个epoch
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler, args.amp
        )
        
        # 验证
        val_loss, val_dice, val_hausdorff = validate(model, val_loader, loss_fn, device, args)
        
        # 更新学习率
        scheduler.step(val_dice)
        
        # 记录指标
        if args.rank == 0:
            print(f'Epoch {epoch+1}/{args.epochs}, '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'Val Dice: {val_dice:.4f}, '
                  f'Val Hausdorff: {val_hausdorff:.4f}')
            
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/dice', val_dice, epoch)
            writer.add_scalar('Metrics/hausdorff', val_hausdorff, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            
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
                    model, optimizer, epoch, best_dice, args,
                    os.path.join(args.output_dir, 'checkpoints', 'best_model.pth')
                )
                print(f'New best model saved with Dice: {best_dice:.4f}')
        else:
            patience_counter += 1
        
        # 保存最新模型
        if args.rank == 0 and (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, best_dice, args,
                os.path.join(args.output_dir, 'checkpoints', f'model_epoch_{epoch+1}.pth')
            )
        
        # 早停
        if patience_counter >= args.patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # 关闭TensorBoard写入器
    if args.rank == 0:
        writer.close()
    
    # 在测试集上评估最佳模型
    if args.rank == 0:
        print('Evaluating best model on test set...')
        
        # 加载最佳模型
        checkpoint = torch.load(os.path.join(args.output_dir, 'checkpoints', 'best_model.pth'))
        
        if args.distributed:
            model.module.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint['model_state'])
        
        # 在测试集上评估
        test_loss, test_dice, test_hausdorff = validate(model, test_loader, loss_fn, device, args)
        
        print(f'Test Loss: {test_loss:.4f}, '
              f'Test Dice: {test_dice:.4f}, '
              f'Test Hausdorff: {test_hausdorff:.4f}')


if __name__ == '__main__':
    main()