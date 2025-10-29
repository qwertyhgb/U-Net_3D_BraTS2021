import os
import numpy as np
import nibabel as nib  # 用于处理NIfTI格式医学图像
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage  # 用于图像处理和变换
import random
from skimage.transform import resize  # 用于图像重采样


def normalize_scan(scan):
    """MRI扫描图像强度归一化
    
    将MRI扫描的强度值归一化到[0,1]范围，这是深度学习预处理的关键步骤。
    MRI图像的强度值范围差异很大，归一化有助于：
    1. 加速网络收敛
    2. 提高数值稳定性
    3. 使不同模态的数据具有相似的数值范围
    4. 减少批归一化的负担
    
    采用Min-Max归一化方法：(x - min) / (max - min)
    
    参数:
        scan (ndarray): 输入的3D MRI扫描体积，任意强度范围
        
    返回:
        ndarray: 归一化后的扫描体积，强度范围为[0,1]，数据类型为float32
    """
    # 转换为float32类型，确保数值精度并节省内存
    scan = scan.astype(np.float32)
    
    # 计算全局最小值和最大值
    scan_min = scan.min()
    scan_max = scan.max()
    
    # 防止除零错误：如果图像是常数（min=max），直接返回零数组
    if scan_max > scan_min:
        # Min-Max归一化：将数值范围映射到[0,1]
        scan = (scan - scan_min) / (scan_max - scan_min)
    else:
        # 如果图像是常数，设置为全零（避免NaN）
        scan = np.zeros_like(scan, dtype=np.float32)
    
    return scan


def resample_volume(img, new_shape=(128, 128, 128), interpolation='linear'):
    """3D体积重采样到统一尺寸
    
    将不同尺寸的3D MRI体积重采样到统一的目标尺寸，这是批处理训练的必要步骤。
    BraTS数据集中不同病例的MRI尺寸可能不同，重采样确保：
    1. 所有样本具有相同的空间尺寸
    2. 能够进行批量处理
    3. 网络输入尺寸固定
    4. 内存使用可预测
    
    支持两种插值方法：
    - 线性插值：适用于连续的图像数据（MRI强度值）
    - 最近邻插值：适用于离散的标签数据（分割掩码）
    
    参数:
        img (ndarray): 输入的3D体积，形状为 (D, H, W)
        new_shape (tuple): 目标体积形状，默认为 (128, 128, 128)
        interpolation (str): 插值方法选择
            - 'linear': 双线性插值，适用于MRI图像数据
            - 'nearest': 最近邻插值，适用于分割标签
        
    返回:
        ndarray: 重采样后的3D体积，形状为 new_shape
        
    异常:
        ValueError: 当插值方法不被支持时抛出
    """
    # 验证输入参数
    if not isinstance(new_shape, (tuple, list)) or len(new_shape) != 3:
        raise ValueError("new_shape必须是包含3个元素的元组或列表")
    
    # 如果目标形状与当前形状相同，直接返回副本
    if img.shape == tuple(new_shape):
        return img.copy()
    
    # 计算各维度的缩放因子
    resize_factor = np.array(new_shape) / np.array(img.shape)
    
    if interpolation == 'linear':
        # 双线性插值（order=1）：适用于连续的MRI强度数据
        # preserve_range=True：保持原始数值范围不变
        # anti_aliasing=True：减少混叠效应，提高重采样质量
        return resize(
            img, new_shape, 
            order=1,                # 双线性插值
            preserve_range=True,    # 保持数值范围
            anti_aliasing=True      # 抗混叠
        ).astype(img.dtype)
        
    elif interpolation == 'nearest':
        # 最近邻插值（order=0）：适用于离散的标签数据
        # 确保标签值不会因插值而改变
        return resize(
            img, new_shape, 
            order=0,                # 最近邻插值
            preserve_range=True,    # 保持数值范围
            anti_aliasing=False     # 标签数据不需要抗混叠
        ).astype(img.dtype)
        
    else:
        # 不支持的插值方法
        raise ValueError(f"不支持的插值方法: {interpolation}。支持的方法: 'linear', 'nearest'")


def crop_brain_region(img, mask=None):
    """智能脑部区域裁剪
    
    自动检测并裁剪出脑部感兴趣区域，去除大量的背景空间。
    这个预处理步骤能够：
    1. 显著减少计算量和内存使用
    2. 让网络专注于脑部区域的特征学习
    3. 提高训练效率和推理速度
    4. 减少背景噪声的干扰
    
    算法流程：
    1. 使用自适应阈值检测脑组织区域
    2. 计算脑组织的3D边界框
    3. 添加安全边距防止裁剪过度
    4. 同步裁剪图像和对应的分割掩码
    
    参数:
        img (ndarray): 输入的3D MRI体积，形状为 (D, H, W)
        mask (ndarray, 可选): 对应的分割掩码，形状与img相同
        
    返回:
        如果提供了mask:
            tuple: (裁剪后的图像, 裁剪后的掩码)
        否则:
            ndarray: 裁剪后的图像
    """
    # 使用自适应阈值检测脑组织区域
    # 脑组织的强度通常高于背景（空气、颅骨外区域）
    img_mean = img.mean()
    img_std = img.std()
    
    # 使用均值+0.1*标准差作为阈值，这比单纯使用均值更稳健
    threshold = img_mean + 0.1 * img_std
    brain_mask = img > threshold
    
    # 使用形态学操作去除小的噪声区域
    from scipy import ndimage
    # 闭运算：先膨胀后腐蚀，填补小洞
    brain_mask = ndimage.binary_closing(brain_mask, structure=np.ones((3, 3, 3)))
    # 开运算：先腐蚀后膨胀，去除小的噪声点
    brain_mask = ndimage.binary_opening(brain_mask, structure=np.ones((3, 3, 3)))
    
    # 获取脑组织区域的坐标
    coords = np.array(np.where(brain_mask)).T
    
    # 如果没有检测到脑部区域，返回原始数据
    if len(coords) == 0:
        print("警告：未检测到脑部区域，返回原始图像")
        if mask is not None:
            return img, mask
        return img
    
    # 计算3D边界框的最小和最大坐标
    min_coords = coords.min(axis=0)  # 每个维度的最小坐标
    max_coords = coords.max(axis=0)  # 每个维度的最大坐标
    
    # 添加安全边距，防止裁剪掉脑部边缘的重要信息
    # 边距大小根据图像尺寸自适应调整
    margin = max(5, min(img.shape) // 20)  # 至少5个像素，最多为最小维度的1/20
    
    # 确保边界不超出图像范围
    min_coords = np.maximum(min_coords - margin, 0)
    max_coords = np.minimum(max_coords + margin, np.array(img.shape))
    
    # 执行裁剪操作
    cropped_img = img[
        min_coords[0]:max_coords[0],
        min_coords[1]:max_coords[1], 
        min_coords[2]:max_coords[2]
    ]
    
    # 如果提供了分割掩码，同步裁剪
    if mask is not None:
        cropped_mask = mask[
            min_coords[0]:max_coords[0],
            min_coords[1]:max_coords[1], 
            min_coords[2]:max_coords[2]
        ]
        return cropped_img, cropped_mask
    
    return cropped_img


class RandomRotation3D:
    """3D随机旋转数据增强
    
    在三个空间轴上对3D MRI体积进行随机旋转，模拟患者头部的不同姿态。
    这种增强方法能够：
    1. 增加训练数据的多样性
    2. 提高模型对头部姿态变化的鲁棒性
    3. 减少过拟合，提升泛化能力
    4. 模拟真实临床场景中的头部位置变化
    
    旋转策略：
    - 在X、Y、Z三个轴上分别进行小角度随机旋转
    - 图像数据使用双线性插值保持平滑性
    - 标签数据使用最近邻插值保持离散性
    - 不改变体积形状（reshape=False）避免信息丢失
    
    参数:
        max_angle (float): 最大旋转角度（度），默认为10度
            建议范围：5-15度，过大会导致解剖结构失真
    """
    def __init__(self, max_angle=10):
        self.max_angle = max_angle
        
    def __call__(self, sample):
        """对MRI样本应用随机3D旋转
        
        参数:
            sample (dict): 包含以下键值的字典
                - 'image': MRI图像数据，形状为 (4, D, H, W) 或 (D, H, W)
                - 'mask': 分割掩码数据，形状为 (D, H, W)
            
        返回:
            dict: 旋转后的样本，保持原始数据结构
        """
        image, mask = sample['image'], sample['mask']
        
        # 为三个旋转轴生成随机角度
        # 使用uniform分布确保各个角度等概率出现
        angle_x = random.uniform(-self.max_angle, self.max_angle)  # 绕X轴旋转（俯仰）
        angle_y = random.uniform(-self.max_angle, self.max_angle)  # 绕Y轴旋转（偏航）
        angle_z = random.uniform(-self.max_angle, self.max_angle)  # 绕Z轴旋转（翻滚）
        
        # 处理多通道图像数据（4个MRI模态）
        if len(image.shape) == 4:  # 多通道情况 (C, D, H, W)
            rotated_image = np.zeros_like(image)
            for c in range(image.shape[0]):  # 对每个通道分别旋转
                # 依次在三个轴上旋转，使用双线性插值（order=1）
                temp = ndimage.rotate(image[c], angle_x, axes=(1, 2), reshape=False, order=1)
                temp = ndimage.rotate(temp, angle_y, axes=(0, 2), reshape=False, order=1)
                rotated_image[c] = ndimage.rotate(temp, angle_z, axes=(0, 1), reshape=False, order=1)
            image = rotated_image
        else:  # 单通道情况 (D, H, W)
            # 依次在三个轴上旋转图像，使用双线性插值保持平滑性
            image = ndimage.rotate(image, angle_x, axes=(1, 2), reshape=False, order=1)
            image = ndimage.rotate(image, angle_y, axes=(0, 2), reshape=False, order=1)
            image = ndimage.rotate(image, angle_z, axes=(0, 1), reshape=False, order=1)
        
        # 旋转分割掩码，使用最近邻插值（order=0）保持标签的离散性
        # 这确保旋转后标签值不会因插值而改变
        mask = ndimage.rotate(mask, angle_x, axes=(1, 2), reshape=False, order=0)
        mask = ndimage.rotate(mask, angle_y, axes=(0, 2), reshape=False, order=0)
        mask = ndimage.rotate(mask, angle_z, axes=(0, 1), reshape=False, order=0)
        
        return {'image': image, 'mask': mask}


class RandomFlip3D:
    """3D随机翻转数据增强
    
    沿随机选择的空间轴对3D MRI体积进行镜像翻转。
    这种增强方法基于脑部解剖结构的对称性特点：
    1. 左右翻转：利用大脑左右半球的相似性
    2. 前后翻转：在某些情况下可以增加数据多样性
    3. 上下翻转：通常不推荐，因为违反解剖学常识
    
    翻转增强的优势：
    - 简单高效，计算开销小
    - 保持图像质量不变
    - 增加训练样本的有效数量
    - 提高模型对空间变换的鲁棒性
    
    注意事项：
    - 需要考虑解剖学合理性
    - 某些病理可能具有方向性，需谨慎使用
    
    参数:
        p (float): 执行翻转的概率，默认为0.5
            建议范围：0.3-0.7，平衡增强效果和训练稳定性
    """
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, sample):
        """对MRI样本应用随机翻转
        
        参数:
            sample (dict): 包含以下键值的字典
                - 'image': MRI图像数据，形状为 (4, D, H, W) 或 (D, H, W)
                - 'mask': 分割掩码数据，形状为 (D, H, W)
            
        返回:
            dict: 翻转后的样本，保持原始数据结构
        """
        image, mask = sample['image'], sample['mask']
        
        # 根据设定概率决定是否执行翻转
        if random.random() < self.p:
            # 随机选择翻转轴
            # 对于脑部MRI，通常选择以下轴：
            # - 轴0（深度轴）：上下翻转，通常不推荐
            # - 轴1（高度轴）：前后翻转，需谨慎
            # - 轴2（宽度轴）：左右翻转，最常用且安全
            
            # 这里我们限制为更安全的翻转轴
            if len(image.shape) == 4:  # 多通道情况
                # 对于多通道数据，翻转轴需要相应调整
                axis = random.choice([1, 2, 3])  # 对应空间维度的轴
            else:  # 单通道情况
                axis = random.choice([0, 1, 2])  # 直接对应空间轴
            
            # 执行翻转操作
            # np.flip()沿指定轴翻转数组
            # .copy()创建副本，避免内存视图问题
            if len(image.shape) == 4:  # 多通道图像
                image = np.flip(image, axis=axis).copy()
            else:  # 单通道图像
                image = np.flip(image, axis=axis).copy()
            
            # 对分割掩码执行相同的翻转操作
            if len(mask.shape) == 3:  # 标准3D掩码
                # 调整轴索引以匹配掩码的维度
                mask_axis = axis if len(image.shape) == 3 else axis - 1
                mask = np.flip(mask, axis=mask_axis).copy()
            else:
                mask = np.flip(mask, axis=axis).copy()
            
        return {'image': image, 'mask': mask}


class ElasticDeformation3D:
    """3D弹性变形数据增强
    
    模拟生物组织的非刚性变形，这是医学图像增强中的高级技术。
    弹性变形能够模拟：
    1. 脑组织的自然形变
    2. 不同患者间的解剖差异
    3. 成像过程中的微小运动
    4. 病理状态下的组织变形
    
    算法原理：
    1. 生成随机位移场（displacement field）
    2. 使用高斯滤波平滑位移场，确保变形的连续性
    3. 将位移场应用到原始坐标网格上
    4. 通过插值重采样得到变形后的图像
    
    变形特点：
    - 局部保持拓扑结构
    - 全局产生自然的形变效果
    - 可控的变形强度和平滑度
    
    参数:
        alpha (float): 变形强度系数，控制位移的幅度，默认为15
            - 较小值(5-10)：轻微变形，适合精细结构
            - 中等值(10-20)：中等变形，平衡真实性和多样性
            - 较大值(20+)：强烈变形，可能影响解剖合理性
        sigma (float): 高斯滤波标准差，控制变形的平滑度，默认为3
            - 较小值(1-2)：局部变形，细节丰富但可能不自然
            - 中等值(3-5)：平衡的变形平滑度
            - 较大值(5+)：全局平滑变形，更自然但细节较少
        p (float): 执行变形的概率，默认为0.5
    """
    def __init__(self, alpha=15, sigma=3, p=0.5):
        self.alpha = alpha      # 变形强度系数
        self.sigma = sigma      # 高斯滤波标准差
        self.p = p             # 执行概率
        
    def __call__(self, sample):
        """对MRI样本应用弹性变形
        
        参数:
            sample (dict): 包含以下键值的字典
                - 'image': MRI图像数据，形状为 (4, D, H, W) 或 (D, H, W)
                - 'mask': 分割掩码数据，形状为 (D, H, W)
            
        返回:
            dict: 变形后的样本，保持原始数据结构
        """
        # 根据概率决定是否执行变形
        if random.random() > self.p:
            return sample
            
        image, mask = sample['image'], sample['mask']
        
        # 确定变形的空间维度
        if len(image.shape) == 4:  # 多通道情况 (C, D, H, W)
            spatial_shape = image.shape[1:]  # (D, H, W)
        else:  # 单通道情况 (D, H, W)
            spatial_shape = image.shape
        
        # 生成三个方向的随机位移场
        # 使用标准正态分布生成随机噪声，然后用高斯滤波平滑
        displacement_x = ndimage.gaussian_filter(
            np.random.randn(*spatial_shape), 
            self.sigma, 
            mode='constant', 
            cval=0
        ) * self.alpha
        
        displacement_y = ndimage.gaussian_filter(
            np.random.randn(*spatial_shape), 
            self.sigma, 
            mode='constant', 
            cval=0
        ) * self.alpha
        
        displacement_z = ndimage.gaussian_filter(
            np.random.randn(*spatial_shape), 
            self.sigma, 
            mode='constant', 
            cval=0
        ) * self.alpha
        
        # 创建原始坐标网格
        # meshgrid生成坐标矩阵，indexing='ij'确保正确的索引顺序
        coords_x, coords_y, coords_z = np.meshgrid(
            np.arange(spatial_shape[0]),
            np.arange(spatial_shape[1]), 
            np.arange(spatial_shape[2]), 
            indexing='ij'
        )
        
        # 应用位移场到坐标网格
        # 新坐标 = 原坐标 + 位移
        deformed_coords = [
            coords_x + displacement_x,
            coords_y + displacement_y,
            coords_z + displacement_z
        ]
        
        # 对图像数据应用弹性变形
        if len(image.shape) == 4:  # 多通道情况
            deformed_image = np.zeros_like(image)
            for c in range(image.shape[0]):  # 对每个通道分别处理
                # 使用双线性插值（order=1）保持图像平滑性
                # mode='nearest'处理边界，prefilter=False加速计算
                deformed_image[c] = ndimage.map_coordinates(
                    image[c], 
                    deformed_coords, 
                    order=1,           # 双线性插值
                    mode='nearest',    # 边界处理
                    prefilter=False    # 不进行预滤波，加速计算
                )
            image = deformed_image
        else:  # 单通道情况
            image = ndimage.map_coordinates(
                image, 
                deformed_coords, 
                order=1, 
                mode='nearest', 
                prefilter=False
            )
        
        # 对分割掩码应用相同的变形
        # 使用最近邻插值（order=0）保持标签的离散性
        mask = ndimage.map_coordinates(
            mask, 
            deformed_coords, 
            order=0,           # 最近邻插值
            mode='nearest',    # 边界处理
            prefilter=False    # 不进行预滤波
        )
        
        return {'image': image, 'mask': mask}


class BraTSDataset(Dataset):
    """BraTS脑肿瘤分割数据集加载器
    
    专为BraTS（Brain Tumor Segmentation）挑战赛数据集设计的PyTorch数据集类。
    该类负责：
    1. 加载多模态MRI数据（T1、T1ce、T2、FLAIR）
    2. 加载对应的分割标注
    3. 执行标准化的预处理流程
    4. 应用数据增强（训练模式）
    5. 提供批量数据加载接口
    
    BraTS数据集特点：
    - 每个病例包含4种MRI模态
    - 分割标注包含4个类别（背景、坏死核心、水肿、增强肿瘤）
    - 数据格式为NIfTI (.nii.gz)
    - 不同病例的图像尺寸可能不同
    
    预处理流程：
    1. 读取NIfTI文件
    2. 强度归一化
    3. 脑部区域裁剪
    4. 重采样到统一尺寸
    5. 标签重映射
    6. 数据增强（可选）
    
    参数:
        data_dir (str): BraTS数据集根目录路径
        transform (callable, 可选): 数据增强变换函数
        mode (str): 数据集模式，支持 'train', 'val', 'test'
        target_shape (tuple): 目标体积形状，默认为 (128, 128, 128)
    """
    def __init__(self, data_dir, transform=None, mode='train', target_shape=(128, 128, 128)):
        super(BraTSDataset, self).__init__()
        
        self.data_dir = data_dir            # 数据集根目录
        self.transform = transform          # 数据增强变换
        self.mode = mode                   # 数据集模式
        self.target_shape = target_shape   # 目标重采样尺寸
        
        # 扫描数据目录，收集所有BraTS病例
        self.cases = []
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
            
        for case_folder in os.listdir(data_dir):
            case_path = os.path.join(data_dir, case_folder)
            # 只选择BraTS开头的目录，确保是有效的病例文件夹
            if os.path.isdir(case_path) and case_folder.startswith('BraTS'):
                # 验证病例文件夹是否包含必要的文件
                if self._validate_case(case_path, case_folder):
                    self.cases.append(case_folder)
                else:
                    print(f"警告：病例 {case_folder} 缺少必要文件，已跳过")
        
        if len(self.cases) == 0:
            raise ValueError(f"在目录 {data_dir} 中未找到有效的BraTS病例")
        
        # 对病例列表进行排序，确保数据集划分的一致性
        self.cases.sort()
        
        # 按模式划分数据集
        # 使用固定的划分比例：训练集70%，验证集15%，测试集15%
        total_cases = len(self.cases)
        train_end = int(total_cases * 0.7)
        val_end = int(total_cases * 0.85)
        
        if mode == 'train':
            self.cases = self.cases[:train_end]
        elif mode == 'val':
            self.cases = self.cases[train_end:val_end]
        elif mode == 'test':
            self.cases = self.cases[val_end:]
        else:
            raise ValueError(f"不支持的数据集模式: {mode}。支持的模式: 'train', 'val', 'test'")
        
        print(f"{mode.upper()}数据集: {len(self.cases)} 个病例")
    
    def _validate_case(self, case_path, case_folder):
        """验证病例文件夹是否包含所有必要的文件
        
        参数:
            case_path (str): 病例文件夹路径
            case_folder (str): 病例文件夹名称
            
        返回:
            bool: 如果包含所有必要文件返回True，否则返回False
        """
        required_files = [
            f"{case_folder}_t1.nii.gz",      # T1加权图像
            f"{case_folder}_t1ce.nii.gz",    # T1对比增强图像
            f"{case_folder}_t2.nii.gz",      # T2加权图像
            f"{case_folder}_flair.nii.gz",   # FLAIR图像
            f"{case_folder}_seg.nii.gz"      # 分割标注
        ]
        
        for file_name in required_files:
            file_path = os.path.join(case_path, file_name)
            if not os.path.exists(file_path):
                return False
        return True
        
    def __len__(self):
        """返回数据集中的样本数量
        
        返回:
            int: 当前数据集模式下的病例数量
        """
        return len(self.cases)
    
    def __getitem__(self, idx):
        """获取指定索引的样本数据
        
        执行完整的数据加载和预处理流程：
        1. 读取多模态MRI数据和分割标注
        2. 强度归一化
        3. 脑部区域裁剪
        4. 重采样到统一尺寸
        5. 标签重映射
        6. 数据增强（训练模式）
        7. 转换为PyTorch张量
        
        参数:
            idx (int): 样本索引，范围为 [0, len(self.cases)-1]
            
        返回:
            dict: 包含以下键值的字典
                - 'image': 预处理后的多模态MRI数据，形状为 [4, D, H, W]
                - 'mask': 预处理后的分割掩码，形状为 [D, H, W]
                - 'case_id': 病例标识符（文件夹名称）
        """
        try:
            # 获取病例信息
            case_folder = self.cases[idx]
            case_path = os.path.join(self.data_dir, case_folder)
            
            # 构建各模态文件的完整路径
            file_paths = {
                't1': os.path.join(case_path, f"{case_folder}_t1.nii.gz"),
                't1ce': os.path.join(case_path, f"{case_folder}_t1ce.nii.gz"),
                't2': os.path.join(case_path, f"{case_folder}_t2.nii.gz"),
                'flair': os.path.join(case_path, f"{case_folder}_flair.nii.gz"),
                'seg': os.path.join(case_path, f"{case_folder}_seg.nii.gz")
            }
            
            # ==================== 数据读取 ====================
            # 读取四种MRI模态的NIfTI文件
            modalities = {}
            for modality, path in file_paths.items():
                if modality != 'seg':  # 分割标注单独处理
                    try:
                        nii_data = nib.load(path)
                        modalities[modality] = nii_data.get_fdata().astype(np.float32)
                    except Exception as e:
                        raise RuntimeError(f"读取文件失败 {path}: {str(e)}")
            
            # 读取分割标注
            try:
                seg_nii = nib.load(file_paths['seg'])
                seg = seg_nii.get_fdata().astype(np.uint8)
            except Exception as e:
                raise RuntimeError(f"读取分割文件失败 {file_paths['seg']}: {str(e)}")
            
            # ==================== 强度归一化 ====================
            # 对每个MRI模态进行独立的强度归一化
            for modality in modalities:
                modalities[modality] = normalize_scan(modalities[modality])
            
            # ==================== 脑部区域裁剪 ====================
            # 使用FLAIR图像作为参考进行脑部区域检测和裁剪
            # FLAIR图像通常具有良好的脑组织对比度
            reference_image = modalities['flair']
            
            # 对所有模态应用相同的裁剪区域，确保空间对齐
            cropped_modalities = {}
            for modality, image in modalities.items():
                if modality == 'flair':
                    # 对参考图像和分割标注同时裁剪
                    cropped_modalities[modality], seg = crop_brain_region(image, seg)
                else:
                    # 对其他模态单独裁剪
                    cropped_modalities[modality], _ = crop_brain_region(image)
            
            # ==================== 重采样到统一尺寸 ====================
            # 将所有数据重采样到目标尺寸，便于批处理
            resampled_modalities = {}
            for modality, image in cropped_modalities.items():
                resampled_modalities[modality] = resample_volume(
                    image, self.target_shape, interpolation='linear'
                )
            
            # 对分割掩码使用最近邻插值，保持标签的离散性
            seg = resample_volume(seg, self.target_shape, interpolation='nearest')
            
            # ==================== 多模态数据组合 ====================
            # 将四种MRI模态按固定顺序堆叠为多通道图像
            # 通道顺序：[T1, T1ce, T2, FLAIR]
            image = np.stack([
                resampled_modalities['t1'],
                resampled_modalities['t1ce'],
                resampled_modalities['t2'],
                resampled_modalities['flair']
            ], axis=0)  # 形状: (4, D, H, W)
            
            # ==================== 分割标签重映射 ====================
            # BraTS原始标签编码：0=背景, 1=坏死核心, 2=水肿, 4=增强肿瘤
            # 重映射为连续标签：0=背景, 1=坏死核心, 2=水肿, 3=增强肿瘤
            mask = np.zeros_like(seg, dtype=np.uint8)
            mask[seg == 1] = 1  # 坏死核心 (Necrotic Core, NCR)
            mask[seg == 2] = 2  # 水肿区域 (Edema, ED)
            mask[seg == 4] = 3  # 增强肿瘤 (Enhancing Tumor, ET)
            # 背景区域保持为0
            
            # 创建样本字典
            sample = {
                'image': image,
                'mask': mask,
                'case_id': case_folder
            }
            
            # ==================== 数据增强 ====================
            # 仅在训练模式下应用数据增强
            if self.transform and self.mode == 'train':
                # 应用数据增强变换
                augmented = self.transform(sample)
                sample['image'] = augmented['image']
                sample['mask'] = augmented['mask']
            
            # ==================== 转换为PyTorch张量 ====================
            # 转换为PyTorch张量，设置正确的数据类型
            image_tensor = torch.from_numpy(sample['image']).float()  # 图像数据：float32
            mask_tensor = torch.from_numpy(sample['mask']).long()     # 标签数据：int64
            
            return {
                'image': image_tensor,
                'mask': mask_tensor,
                'case_id': case_folder
            }
            
        except Exception as e:
            # 详细的错误信息，便于调试
            error_msg = f"处理病例 {case_folder} (索引 {idx}) 时发生错误: {str(e)}"
            print(f"错误: {error_msg}")
            raise RuntimeError(error_msg)


def get_transforms(mode='train'):
    """获取数据增强转换"""
    if mode == 'train':
        transforms = [
            RandomRotation3D(max_angle=15),
            RandomFlip3D(p=0.5),
            ElasticDeformation3D(alpha=15, sigma=3, p=0.3)
        ]
        
        def transform(sample):
            for t in transforms:
                sample = t(sample)
            return sample
            
        return transform
    else:
        return None


def get_data_loaders(data_dir, batch_size=2, num_workers=4, target_shape=(128, 128, 128)):
    """获取数据加载器"""
    train_transform = get_transforms(mode='train')
    
    train_dataset = BraTSDataset(
        data_dir=data_dir,
        transform=train_transform,
        mode='train',
        target_shape=target_shape
    )
    
    val_dataset = BraTSDataset(
        data_dir=data_dir,
        transform=None,
        mode='val',
        target_shape=target_shape
    )
    
    test_dataset = BraTSDataset(
        data_dir=data_dir,
        transform=None,
        mode='test',
        target_shape=target_shape
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader