import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """双重3D卷积模块
    
    这是U-Net架构的基础构建块，执行两次连续的3D卷积操作。
    每次卷积后都进行批归一化和ReLU激活，有助于网络的稳定训练和非线性表达。
    支持可选的残差连接，可以缓解深层网络的梯度消失问题。
    
    网络结构:
        输入 -> Conv3D -> BatchNorm3D -> ReLU -> Conv3D -> BatchNorm3D -> ReLU -> 输出
        (可选) 输入 -> 1x1 Conv3D -> BatchNorm3D -> 与主路径相加
    
    参数:
        in_channels (int): 输入特征图的通道数
        out_channels (int): 输出特征图的通道数
        mid_channels (int, 可选): 中间层的通道数，如果未指定则等于输出通道数
        residual (bool, 可选): 是否启用残差连接，默认为False
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super(DoubleConv3D, self).__init__()
        
        # 如果未指定中间通道数，则设为与输出通道数相同
        if not mid_channels:
            mid_channels = out_channels
            
        # 构建双重卷积序列：Conv3D -> BN -> ReLU -> Conv3D -> BN -> ReLU
        self.double_conv = nn.Sequential(
            # 第一个3D卷积层：3x3x3卷积核，padding=1保持空间尺寸不变
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # 批归一化：标准化特征图，加速收敛并提高稳定性
            nn.BatchNorm3d(mid_channels),
            # ReLU激活函数：引入非线性，inplace=True节省内存
            nn.ReLU(inplace=True),
            
            # 第二个3D卷积层：继续提取特征
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # 第二次批归一化
            nn.BatchNorm3d(out_channels),
            # 第二次ReLU激活
            nn.ReLU(inplace=True)
        )
        
        # 残差连接配置
        self.residual = residual
        if self.residual:
            # 1x1x1卷积用于调整输入通道数，使其与输出通道数匹配
            # 这样才能进行残差相加操作
            self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            # 残差路径的批归一化
            self.res_bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        """前向传播
        
        参数:
            x (Tensor): 输入特征图，形状为 [B, C, D, H, W]
            
        返回:
            Tensor: 输出特征图，形状为 [B, out_channels, D, H, W]
        """
        if self.residual:
            # 残差连接模式：输出 = 主路径输出 + 残差路径输出
            # 残差路径：通过1x1卷积调整通道数，然后批归一化
            residual = self.res_bn(self.res_conv(x))
            # 主路径 + 残差路径
            return self.double_conv(x) + residual
        else:
            # 标准模式：直接返回双重卷积的结果
            return self.double_conv(x)


class Down3D(nn.Module):
    """3D下采样模块
    
    U-Net编码器路径的下采样模块，负责逐步降低空间分辨率并增加特征通道数。
    通过最大池化操作将特征图的空间尺寸减半，然后通过双重卷积提取更高级的特征。
    这种设计能够捕获不同尺度的特征信息，为后续的上采样和特征融合提供丰富的语义信息。
    
    网络结构:
        输入 -> MaxPool3D(2x2x2) -> DoubleConv3D -> 输出
        空间尺寸: (D,H,W) -> (D/2,H/2,W/2)
        通道数: in_channels -> out_channels
    
    参数:
        in_channels (int): 输入特征图的通道数
        out_channels (int): 输出特征图的通道数，通常是输入通道数的2倍
        residual (bool, 可选): 是否在双重卷积中使用残差连接，默认为False
    """
    def __init__(self, in_channels, out_channels, residual=False):
        super(Down3D, self).__init__()
        
        # 构建下采样序列：最大池化 + 双重卷积
        self.maxpool_conv = nn.Sequential(
            # 3D最大池化：kernel_size=2, stride=2，将空间尺寸减半
            # 选择最大池化而非平均池化，因为它能更好地保留重要特征
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # 双重卷积：在降低分辨率后提取更高级的特征表示
            DoubleConv3D(in_channels, out_channels, residual=residual)
        )

    def forward(self, x):
        """前向传播
        
        参数:
            x (Tensor): 输入特征图，形状为 [B, in_channels, D, H, W]
            
        返回:
            Tensor: 下采样后的特征图，形状为 [B, out_channels, D/2, H/2, W/2]
        """
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """3D上采样模块
    
    U-Net解码器路径的上采样模块，负责恢复空间分辨率并融合多尺度特征。
    该模块实现了U-Net的核心思想：将低分辨率的深层特征与高分辨率的浅层特征结合，
    既保留了语义信息又恢复了空间细节。跳跃连接的设计有助于梯度传播和细节恢复。
    
    网络结构:
        低分辨率特征 -> 上采样 -> 与跳跃连接特征拼接 -> DoubleConv3D -> 输出
        空间尺寸: (D/2,H/2,W/2) -> (D,H,W)
        通道数: in_channels -> out_channels
    
    参数:
        in_channels (int): 输入特征图的通道数（来自下层的特征）
        out_channels (int): 输出特征图的通道数
        bilinear (bool, 可选): 上采样方式选择，True使用三线性插值，False使用转置卷积
        residual (bool, 可选): 是否在双重卷积中使用残差连接，默认为False
    """
    def __init__(self, in_channels, out_channels, bilinear=False, residual=False):
        super(Up3D, self).__init__()

        if bilinear:
            # 三线性插值上采样方式
            # 优点：参数少，计算快，内存占用小
            # 缺点：上采样质量相对较低，无法学习上采样参数
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            # 由于使用插值上采样，需要通过卷积调整通道数
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2, residual=residual)
        else:
            # 转置卷积上采样方式（推荐）
            # 优点：可学习的上采样参数，能够产生更好的上采样效果
            # 缺点：参数多，计算量大，可能产生棋盘效应
            self.up = nn.ConvTranspose3d(
                in_channels, in_channels // 2, 
                kernel_size=2, stride=2,  # 2x2x2核，步长2，实现2倍上采样
                bias=False  # 通常与BatchNorm配合使用时不需要bias
            )
            # 拼接后的通道数为 in_channels（in_channels//2 + in_channels//2）
            self.conv = DoubleConv3D(in_channels, out_channels, residual=residual)

    def forward(self, x1, x2):
        """前向传播
        
        参数:
            x1 (Tensor): 来自下层的低分辨率特征图，形状为 [B, in_channels, D/2, H/2, W/2]
            x2 (Tensor): 来自编码器的跳跃连接特征图，形状为 [B, in_channels//2, D, H, W]
            
        返回:
            Tensor: 融合后的特征图，形状为 [B, out_channels, D, H, W]
        """
        # 第一步：对低分辨率特征进行上采样
        x1 = self.up(x1)  # [B, in_channels//2, D, H, W]
        
        # 第二步：处理尺寸不匹配问题
        # 由于池化和上采样的舍入误差，x1和x2的尺寸可能略有差异
        # 计算三个空间维度的尺寸差异
        diffD = x2.size()[2] - x1.size()[2]  # 深度维度差异
        diffH = x2.size()[3] - x1.size()[3]  # 高度维度差异  
        diffW = x2.size()[4] - x1.size()[4]  # 宽度维度差异

        # 对x1进行对称填充，使其尺寸与x2完全匹配
        # F.pad的参数顺序是从最后一个维度开始：[W_left, W_right, H_left, H_right, D_left, D_right]
        x1 = F.pad(x1, [
            diffW // 2, diffW - diffW // 2,  # 宽度填充：左侧和右侧
            diffH // 2, diffH - diffH // 2,  # 高度填充：上侧和下侧
            diffD // 2, diffD - diffD // 2   # 深度填充：前侧和后侧
        ])
        
        # 第三步：在通道维度上拼接特征图
        # x2: 跳跃连接特征（高分辨率，浅层语义）
        # x1: 上采样特征（恢复分辨率，深层语义）
        x = torch.cat([x2, x1], dim=1)  # [B, in_channels, D, H, W]
        
        # 第四步：通过双重卷积融合特征并输出最终结果
        return self.conv(x)


class OutConv3D(nn.Module):
    """3D输出卷积层
    
    网络的最终输出层，负责将特征图映射到目标分割类别数。
    使用1x1x1卷积核实现逐像素的分类，不改变空间尺寸，只调整通道数。
    这一层的输出通常需要经过softmax或sigmoid激活函数得到最终的分割概率图。
    
    网络结构:
        输入特征图 -> Conv3D(1x1x1) -> 分割logits
        空间尺寸: 保持不变 (D,H,W)
        通道数: in_channels -> out_channels (分割类别数)
    
    参数:
        in_channels (int): 输入特征图的通道数，通常是U-Net最后一层的特征通道数
        out_channels (int): 输出通道数，等于分割任务的类别数（包括背景）
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        
        # 1x1x1卷积：逐像素分类器
        # 不使用padding，因为1x1x1卷积不会改变空间尺寸
        # 使用bias=True，因为这是最后一层，需要偏置项进行最终调整
        self.conv = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=1,  # 1x1x1卷积核
            stride=1,       # 步长为1
            padding=0,      # 无需填充
            bias=True       # 启用偏置项
        )

    def forward(self, x):
        """前向传播
        
        参数:
            x (Tensor): 输入特征图，形状为 [B, in_channels, D, H, W]
            
        返回:
            Tensor: 分割logits，形状为 [B, out_channels, D, H, W]
                   每个通道对应一个分割类别的未归一化概率
        """
        return self.conv(x)


class DeepSupervision3D(nn.Module):
    """3D深度监督模块
    
    深度监督是一种训练技巧，通过在网络的中间层添加辅助分类器来改善梯度流动。
    这种方法特别适用于深层网络，能够：
    1. 缓解梯度消失问题，使深层网络更容易训练
    2. 提供多尺度的监督信号，改善分割边界的精确度
    3. 加速网络收敛，提高训练稳定性
    
    该模块将中间层的特征图转换为与最终输出相同尺寸的分割预测，
    在训练时与真实标签计算损失，但权重会逐层递减。
    
    网络结构:
        中间特征图 -> Conv3D(1x1x1) -> 上采样 -> 辅助分割预测
    
    参数:
        in_channels (int): 输入特征图的通道数
        out_channels (int): 输出通道数，应与最终分割类别数相同
        scale_factor (int): 上采样倍数，用于将特征图恢复到原始输入尺寸
    """
    def __init__(self, in_channels, out_channels, scale_factor):
        super(DeepSupervision3D, self).__init__()
        
        # 1x1x1卷积：将特征图转换为分割logits
        self.conv = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=1,  # 逐像素分类
            bias=True       # 启用偏置项
        )
        
        # 保存上采样倍数
        self.scale_factor = scale_factor

    def forward(self, x):
        """前向传播
        
        参数:
            x (Tensor): 中间层特征图，形状为 [B, in_channels, D/scale, H/scale, W/scale]
            
        返回:
            Tensor: 辅助分割预测，形状为 [B, out_channels, D, H, W]
        """
        # 第一步：通过1x1x1卷积生成分割logits
        x = self.conv(x)
        
        # 第二步：使用三线性插值上采样到原始尺寸
        # mode='trilinear': 3D三线性插值，提供平滑的上采样结果
        # align_corners=True: 对齐角点，保持几何一致性
        x = F.interpolate(
            x, 
            scale_factor=self.scale_factor, 
            mode='trilinear', 
            align_corners=True
        )
        
        return x


class UNet3D(nn.Module):
    """3D U-Net模型
    
    专为3D医学图像分割设计的深度卷积神经网络。该模型基于经典的U-Net架构，
    采用编码器-解码器结构，通过跳跃连接融合多尺度特征信息。
    
    模型特点:
    1. 编码器路径：逐步下采样，提取高级语义特征
    2. 解码器路径：逐步上采样，恢复空间分辨率
    3. 跳跃连接：融合不同尺度的特征，保留细节信息
    4. 残差连接：缓解梯度消失，提升深层网络训练效果
    5. 深度监督：多层级监督，改善梯度流动和分割精度
    
    网络架构:
        输入(4通道) -> 编码器(5层) -> 解码器(4层) -> 输出(分割类别数)
        特征通道数: [16, 32, 64, 128, 256] -> [128, 64, 32, 16] -> 分割类别数
        空间尺寸: 原始 -> 1/16 -> 原始
    
    参数:
        in_channels (int): 输入通道数，默认为4（T1、T1ce、T2、FLAIR四种MRI模态）
        out_channels (int): 输出通道数，默认为4（背景+3个肿瘤区域）
        features (list): 每层编码器的特征通道数，默认为[16, 32, 64, 128, 256]
        deep_supervision (bool): 是否启用深度监督，默认为False
        residual (bool): 是否在卷积块中使用残差连接，默认为True
    """
    def __init__(self, in_channels=4, out_channels=4, features=[16, 32, 64, 128, 256], 
                 deep_supervision=False, residual=True):
        super(UNet3D, self).__init__()
        
        # 保存深度监督配置
        self.deep_supervision = deep_supervision
        
        # ==================== 编码器路径（下采样路径）====================
        # 编码器负责提取多尺度的特征表示，从低级纹理特征到高级语义特征
        
        # 输入卷积层：处理原始多模态MRI数据
        self.inc = DoubleConv3D(in_channels, features[0], residual=residual)
        
        # 第一层下采样：空间尺寸减半，特征通道数增加
        self.down1 = Down3D(features[0], features[1], residual=residual)  # 1/2分辨率
        
        # 第二层下采样：继续提取更抽象的特征
        self.down2 = Down3D(features[1], features[2], residual=residual)  # 1/4分辨率
        
        # 第三层下采样：高级语义特征提取
        self.down3 = Down3D(features[2], features[3], residual=residual)  # 1/8分辨率
        
        # 第四层下采样（瓶颈层）：最深层特征，感受野最大
        self.down4 = Down3D(features[3], features[4], residual=residual)  # 1/16分辨率
        
        # ==================== 解码器路径（上采样路径）====================
        # 解码器负责恢复空间分辨率，并融合编码器的多尺度特征
        
        # 第一层上采样：从瓶颈层开始恢复分辨率
        # 输入通道数 = 当前层特征 + 跳跃连接特征
        self.up1 = Up3D(features[4], features[3], residual=residual)  # 1/8分辨率
        
        # 第二层上采样：继续恢复分辨率并融合特征
        self.up2 = Up3D(features[3], features[2], residual=residual)  # 1/4分辨率
        
        # 第三层上采样：接近原始分辨率
        self.up3 = Up3D(features[2], features[1], residual=residual)  # 1/2分辨率
        
        # 第四层上采样：恢复到原始分辨率
        self.up4 = Up3D(features[1], features[0], residual=residual)  # 原始分辨率
        
        # ==================== 输出层 ====================
        # 最终分割输出层：将特征图转换为分割概率图
        self.outc = OutConv3D(features[0], out_channels)
        
        # ==================== 深度监督模块 ====================
        # 如果启用深度监督，在多个解码器层添加辅助分类器
        if deep_supervision:
            # 从第一次上采样后的特征生成辅助输出（1/8分辨率 -> 原始分辨率）
            self.ds1 = DeepSupervision3D(features[3], out_channels, scale_factor=8)
            
            # 从第二次上采样后的特征生成辅助输出（1/4分辨率 -> 原始分辨率）
            self.ds2 = DeepSupervision3D(features[2], out_channels, scale_factor=4)
            
            # 从第三次上采样后的特征生成辅助输出（1/2分辨率 -> 原始分辨率）
            self.ds3 = DeepSupervision3D(features[1], out_channels, scale_factor=2)

    def forward(self, x):
        """前向传播
        
        执行完整的U-Net前向传播过程，包括编码器下采样、解码器上采样和特征融合。
        如果启用深度监督，还会在多个解码器层生成辅助输出用于训练监督。
        
        数据流向:
        1. 编码器路径：逐层下采样，提取多尺度特征
        2. 解码器路径：逐层上采样，融合跳跃连接特征
        3. 深度监督：在中间层生成辅助分割预测（可选）
        4. 最终输出：生成完整分辨率的分割结果
        
        参数:
            x (Tensor): 输入的多模态MRI数据，形状为 [B, C, D, H, W]
                B: 批量大小
                C: 输入通道数（4个MRI模态）
                D, H, W: 体积的深度、高度、宽度
                
        返回:
            如果启用深度监督:
                tuple: (主输出, 辅助输出1, 辅助输出2, 辅助输出3)
                - 主输出: 最终分割结果，形状为 [B, out_channels, D, H, W]
                - 辅助输出: 中间层分割结果，用于深度监督训练
            否则:
                Tensor: 主输出，形状为 [B, out_channels, D, H, W]
        """
        # ==================== 编码器路径（特征提取）====================
        # 逐步降低空间分辨率，提取多尺度特征表示
        
        # 输入层：处理原始多模态数据 [B, 4, D, H, W] -> [B, 16, D, H, W]
        x1 = self.inc(x)
        
        # 编码器第1层：[B, 16, D, H, W] -> [B, 32, D/2, H/2, W/2]
        x2 = self.down1(x1)
        
        # 编码器第2层：[B, 32, D/2, H/2, W/2] -> [B, 64, D/4, H/4, W/4]
        x3 = self.down2(x2)
        
        # 编码器第3层：[B, 64, D/4, H/4, W/4] -> [B, 128, D/8, H/8, W/8]
        x4 = self.down3(x3)
        
        # 编码器第4层（瓶颈层）：[B, 128, D/8, H/8, W/8] -> [B, 256, D/16, H/16, W/16]
        x5 = self.down4(x4)
        
        # ==================== 解码器路径（特征融合与上采样）====================
        # 逐步恢复空间分辨率，融合编码器的跳跃连接特征
        
        # 解码器第1层：融合最深层特征
        # [B, 256, D/16, H/16, W/16] + [B, 128, D/8, H/8, W/8] -> [B, 128, D/8, H/8, W/8]
        x = self.up1(x5, x4)
        
        # 深度监督输出1（如果启用）：从1/8分辨率生成辅助分割预测
        if self.deep_supervision:
            ds1 = self.ds1(x)  # [B, out_channels, D, H, W]
        
        # 解码器第2层：继续融合特征
        # [B, 128, D/8, H/8, W/8] + [B, 64, D/4, H/4, W/4] -> [B, 64, D/4, H/4, W/4]
        x = self.up2(x, x3)
        
        # 深度监督输出2（如果启用）：从1/4分辨率生成辅助分割预测
        if self.deep_supervision:
            ds2 = self.ds2(x)  # [B, out_channels, D, H, W]
        
        # 解码器第3层：接近原始分辨率
        # [B, 64, D/4, H/4, W/4] + [B, 32, D/2, H/2, W/2] -> [B, 32, D/2, H/2, W/2]
        x = self.up3(x, x2)
        
        # 深度监督输出3（如果启用）：从1/2分辨率生成辅助分割预测
        if self.deep_supervision:
            ds3 = self.ds3(x)  # [B, out_channels, D, H, W]
        
        # 解码器第4层：恢复到原始分辨率
        # [B, 32, D/2, H/2, W/2] + [B, 16, D, H, W] -> [B, 16, D, H, W]
        x = self.up4(x, x1)
        
        # ==================== 最终输出层 ====================
        # 生成最终的分割预测：[B, 16, D, H, W] -> [B, out_channels, D, H, W]
        output = self.outc(x)
        
        # ==================== 返回结果 ====================
        if self.deep_supervision:
            # 深度监督模式：返回主输出和所有辅助输出
            # 训练时会对所有输出计算损失，权重递减
            return output, ds1, ds2, ds3
        else:
            # 标准模式：只返回主输出
            return output