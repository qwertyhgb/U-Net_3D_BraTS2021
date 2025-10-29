import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def dice_coefficient(y_pred, y_true, smooth=1e-6):
    """计算Dice系数（Sørensen-Dice系数）
    
    Dice系数是医学图像分割中最重要的评估指标之一，用于衡量预测分割与真实分割的重叠程度。
    该指标对分割区域的大小相对不敏感，特别适合评估医学图像中的小目标分割效果。
    
    数学定义：
        Dice = (2 * |预测 ∩ 真实|) / (|预测| + |真实|)
        
    其中：
        - |预测 ∩ 真实| 表示预测和真实分割的交集大小
        - |预测| 和 |真实| 分别表示预测和真实分割的大小
        
    特点：
        - 值域：[0, 1]，1表示完美分割，0表示完全不重叠
        - 对称性：Dice(A,B) = Dice(B,A)
        - 对小目标友好：相比IoU，对小目标的分割效果更敏感
        - 数值稳定：通过平滑项避免除零错误
    
    参数:
        y_pred (Tensor): 预测的分割概率或二值掩码，形状任意，值范围[0,1]
        y_true (Tensor): 真实的分割掩码，形状与y_pred相同，值为0或1
        smooth (float): 平滑项，防止分母为0，提高数值稳定性，默认1e-6
        
    返回:
        Tensor: Dice系数，标量张量，值范围[0,1]
    """
    # 将多维张量展平为一维，便于计算
    # contiguous()确保内存连续性，view(-1)展平为一维
    y_pred_flat = y_pred.contiguous().view(-1)
    y_true_flat = y_true.contiguous().view(-1)
    
    # 计算交集：预测为正且真实为正的像素数量
    # 对于概率预测，这计算的是加权交集
    intersection = (y_pred_flat * y_true_flat).sum()
    
    # 计算并集的替代：预测正例数 + 真实正例数
    # 这种计算方式等价于 |A| + |B|，用于Dice公式的分母
    union_substitute = y_pred_flat.sum() + y_true_flat.sum()
    
    # 计算Dice系数
    # 分子：2 * 交集 + 平滑项
    # 分母：并集替代 + 平滑项
    dice = (2.0 * intersection + smooth) / (union_substitute + smooth)
    
    return dice


def multiclass_dice_coefficient(y_pred, y_true, num_classes=4, smooth=1e-6):
    """计算多类别分割的Dice系数
    
    在多类别分割任务中，需要为每个类别单独计算Dice系数，然后求平均值。
    这种方法能够：
    1. 评估每个解剖结构的分割质量
    2. 识别模型在特定类别上的性能瓶颈
    3. 提供更细粒度的性能分析
    4. 支持类别不平衡情况下的公平评估
    
    计算流程：
    1. 将多类别预测转换为每个类别的二值预测
    2. 将多类别真实标签转换为每个类别的二值标签
    3. 为每个前景类别计算Dice系数
    4. 计算所有前景类别的平均Dice系数
    
    注意：通常跳过背景类别（索引0），只计算前景类别的Dice系数
    
    参数:
        y_pred (Tensor): 模型预测logits，形状为 [B, C, D, H, W]
            B: 批量大小
            C: 类别数量（包括背景）
            D, H, W: 空间维度
        y_true (Tensor): 真实分割标签，形状为 [B, D, H, W]
            每个像素的值表示其所属的类别索引
        num_classes (int): 总类别数量，默认为4（背景+3个肿瘤区域）
        smooth (float): 平滑项，提高数值稳定性
        
    返回:
        tuple: (每个前景类别的Dice系数列表, 平均Dice系数)
            - 列表长度为 num_classes-1（排除背景）
            - 平均Dice系数为所有前景类别的算术平均
    """
    dice_scores = []
    
    # 将预测logits转换为概率分布
    y_pred_softmax = torch.softmax(y_pred, dim=1)  # [B, C, D, H, W]
    
    # 遍历所有前景类别（跳过背景类别，索引0）
    for class_idx in range(1, num_classes):
        # ==================== 提取当前类别的预测和真实标签 ====================
        # 获取当前类别的预测概率
        y_pred_class = y_pred_softmax[:, class_idx, ...]  # [B, D, H, W]
        
        # 将真实标签转换为当前类别的二值掩码
        y_true_class = (y_true == class_idx).float()  # [B, D, H, W]
        
        # ==================== 计算当前类别的Dice系数 ====================
        dice_class = dice_coefficient(y_pred_class, y_true_class, smooth)
        dice_scores.append(dice_class)
    
    # ==================== 计算平均Dice系数 ====================
    if len(dice_scores) > 0:
        # 计算所有前景类别的平均Dice系数
        avg_dice = torch.stack(dice_scores).mean()
    else:
        # 如果没有前景类别，返回0
        avg_dice = torch.tensor(0.0, device=y_pred.device)
    
    return dice_scores, avg_dice


def hausdorff_distance(y_pred, y_true, percentile=95):
    """计算Hausdorff距离（豪斯多夫距离）
    
    Hausdorff距离是评估分割边界准确性的重要几何指标，特别适用于医学图像分割。
    它衡量两个点集之间的最大最小距离，能够捕捉分割边界的最大偏差。
    
    数学定义：
        H(A,B) = max(h(A,B), h(B,A))
        其中 h(A,B) = max_{a∈A} min_{b∈B} ||a-b||
    
    百分位Hausdorff距离：
        使用第p百分位数代替最大值，减少离群点的影响，提高鲁棒性。
        95%Hausdorff距离是常用的变体，平衡了敏感性和鲁棒性。
    
    应用场景：
        - 评估分割边界的精确度
        - 检测分割结果的最大误差
        - 医学图像中器官边界的质量评估
        - 对分割形状变化敏感的任务
    
    参数:
        y_pred (Tensor): 预测的分割掩码，二值张量
        y_true (Tensor): 真实的分割掩码，二值张量
        percentile (int): 百分位数，用于计算鲁棒的Hausdorff距离，默认95
            - 100: 经典Hausdorff距离，对离群点敏感
            - 95: 95%Hausdorff距离，常用的鲁棒变体
            - 50: 中位数Hausdorff距离，最鲁棒但可能不够敏感
        
    返回:
        float: Hausdorff距离（像素单位），值越小表示边界越准确
            - 0: 完美匹配
            - 有限值: 最大边界偏差
            - inf: 其中一个掩码为空
    """
    # 将PyTorch张量转换为numpy数组，并转换为布尔类型
    y_pred_np = y_pred.detach().cpu().numpy().astype(bool)
    y_true_np = y_true.detach().cpu().numpy().astype(bool)
    
    # 检查输入的有效性
    if not y_pred_np.any() and not y_true_np.any():
        # 如果两个掩码都为空，认为距离为0（完美匹配）
        return 0.0
    elif not y_pred_np.any() or not y_true_np.any():
        # 如果其中一个掩码为空，返回无穷大
        return float('inf')
    
    # ==================== 计算从真实掩码到预测掩码的有向距离 ====================
    try:
        # 计算到预测掩码边界的欧氏距离变换
        # ~y_pred_np 表示预测掩码的补集（背景区域）
        # distance_transform_edt 计算到最近前景像素的欧氏距离
        dist_pred = distance_transform_edt(~y_pred_np)
        
        # 提取真实掩码位置处的距离值
        distances_from_true = dist_pred[y_true_np]
        
        # 计算指定百分位数的距离
        if len(distances_from_true) > 0:
            hausdorff_true_to_pred = np.percentile(distances_from_true, percentile)
        else:
            hausdorff_true_to_pred = 0.0
            
    except Exception as e:
        print(f"警告：计算真实到预测的距离时出错: {e}")
        hausdorff_true_to_pred = float('inf')
    
    # ==================== 计算从预测掩码到真实掩码的有向距离 ====================
    try:
        # 计算到真实掩码边界的欧氏距离变换
        dist_true = distance_transform_edt(~y_true_np)
        
        # 提取预测掩码位置处的距离值
        distances_from_pred = dist_true[y_pred_np]
        
        # 计算指定百分位数的距离
        if len(distances_from_pred) > 0:
            hausdorff_pred_to_true = np.percentile(distances_from_pred, percentile)
        else:
            hausdorff_pred_to_true = 0.0
            
    except Exception as e:
        print(f"警告：计算预测到真实的距离时出错: {e}")
        hausdorff_pred_to_true = float('inf')
    
    # ==================== 计算双向Hausdorff距离 ====================
    # Hausdorff距离是两个有向距离的最大值
    hausdorff_dist = max(hausdorff_true_to_pred, hausdorff_pred_to_true)
    
    return float(hausdorff_dist)


def multiclass_hausdorff_distance(y_pred, y_true, num_classes=4, percentile=95):
    """计算多类别分割的Hausdorff距离
    
    在多类别分割任务中，为每个前景类别单独计算Hausdorff距离，然后求平均值。
    这种方法能够：
    1. 评估每个解剖结构的边界精确度
    2. 识别模型在特定类别边界上的性能问题
    3. 提供细粒度的几何精度分析
    4. 支持不同类别形状复杂度的差异化评估
    
    计算流程：
    1. 将多类别预测转换为每个类别的二值预测
    2. 将多类别真实标签转换为每个类别的二值标签
    3. 为每个前景类别计算Hausdorff距离
    4. 计算所有前景类别的平均Hausdorff距离
    
    注意：
    - 跳过背景类别，只计算前景类别的距离
    - 处理空掩码的情况，避免计算错误
    - 使用百分位数提高鲁棒性
    
    参数:
        y_pred (Tensor): 模型预测logits，形状为 [B, C, D, H, W]
        y_true (Tensor): 真实分割标签，形状为 [B, D, H, W]
        num_classes (int): 总类别数量，默认为4（背景+3个肿瘤区域）
        percentile (int): 百分位数，用于计算鲁棒的Hausdorff距离，默认95
        
    返回:
        tuple: (每个前景类别的Hausdorff距离列表, 平均Hausdorff距离)
            - 距离单位为像素
            - 平均距离为所有前景类别的算术平均
    """
    hausdorff_distances = []
    
    # 将预测logits转换为概率分布
    y_pred_softmax = torch.softmax(y_pred, dim=1)  # [B, C, D, H, W]
    
    # 遍历所有前景类别（跳过背景类别，索引0）
    for class_idx in range(1, num_classes):
        try:
            # ==================== 提取当前类别的预测和真实标签 ====================
            # 获取当前类别的预测概率并二值化
            y_pred_class = y_pred_softmax[:, class_idx, ...]  # [B, D, H, W]
            y_pred_binary = (y_pred_class > 0.5).float()  # 二值化阈值0.5
            
            # 将真实标签转换为当前类别的二值掩码
            y_true_binary = (y_true == class_idx).float()  # [B, D, H, W]
            
            # ==================== 批量处理Hausdorff距离计算 ====================
            batch_hausdorff_distances = []
            
            for batch_idx in range(y_pred_binary.shape[0]):
                # 提取当前批次的单个样本
                pred_sample = y_pred_binary[batch_idx]  # [D, H, W]
                true_sample = y_true_binary[batch_idx]  # [D, H, W]
                
                # 计算单个样本的Hausdorff距离
                hausdorff_sample = hausdorff_distance(
                    pred_sample, true_sample, percentile
                )
                
                # 只有当距离是有限值时才加入统计
                if not np.isinf(hausdorff_sample):
                    batch_hausdorff_distances.append(hausdorff_sample)
            
            # ==================== 计算当前类别的平均Hausdorff距离 ====================
            if batch_hausdorff_distances:
                # 计算当前类别在所有批次中的平均距离
                avg_class_hausdorff = np.mean(batch_hausdorff_distances)
                hausdorff_distances.append(avg_class_hausdorff)
            else:
                # 如果所有样本都是无效的，使用无穷大
                print(f"警告：类别 {class_idx} 的所有样本都无法计算有效的Hausdorff距离")
                hausdorff_distances.append(float('inf'))
                
        except Exception as e:
            print(f"警告：计算类别 {class_idx} 的Hausdorff距离时出错: {e}")
            hausdorff_distances.append(float('inf'))
    
    # ==================== 计算所有类别的平均Hausdorff距离 ====================
    if hausdorff_distances:
        # 过滤掉无穷大值，只计算有效距离的平均值
        valid_distances = [d for d in hausdorff_distances if not np.isinf(d)]
        
        if valid_distances:
            avg_hausdorff = np.mean(valid_distances)
        else:
            # 如果所有距离都是无穷大，返回无穷大
            avg_hausdorff = float('inf')
            print("警告：所有类别的Hausdorff距离都是无穷大")
    else:
        # 如果没有计算任何距离，返回无穷大
        avg_hausdorff = float('inf')
        print("警告：没有计算任何有效的Hausdorff距离")
    
    return hausdorff_distances, avg_hausdorff