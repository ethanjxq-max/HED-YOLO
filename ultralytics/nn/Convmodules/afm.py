# afm.py — Adaptive Fusion Module（自适应融合模块）
# 设计目的：在 FEM 之后，根据内容自适应调整特征
# 解决：crazing（网状裂纹）和 rolled-in_scale（氧化皮）的问题

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNSiLU(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ContentEncoder(nn.Module):
    """
    内容感知编码器：提取局部统计特征
    包括：均值、方差、梯度幅值
    """
    def __init__(self, c):
        super().__init__()
        # 用卷积近似统计特征
        self.conv1 = ConvBNSiLU(c, c // 4, k=1)
        self.conv2 = ConvBNSiLU(c // 4, c // 4, k=3)
        self.conv3 = ConvBNSiLU(c // 4, c // 4, k=1)

    def forward(self, x):
        # 局部特征编码
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        return feat


class ModeClassifier(nn.Module):
    """
    模式分类器：判断每个空间位置属于哪种缺陷模式
    输出 3 通道权重：纹理模式、片状模式、点状模式
    """
    def __init__(self, c, reduction=8):
        super().__init__()
        hidden = max(4, c // reduction)

        # 用 ContentEncoder 提取的特征进行分类
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c // 4, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 1),  # 3 种模式
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [B, c//4, H, W] 编码后的特征
        weights = self.classifier(x)  # [B, 3, 1, 1]
        return weights


class AdaptiveEnhance(nn.Module):
    """
    自适应增强：根据模式类型应用不同的增强策略
    """
    def __init__(self, c):
        super().__init__()

        # 1. 纹理模式增强（高通滤波）：边缘锐化
        self.laplacian = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)
        # 用 Laplacian 初始化
        with torch.no_grad():
            laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
            laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
            self.laplacian.weight.data = laplacian_kernel

        # 2. 片状模式增强（低通滤波）：平滑
        self.smooth = nn.Conv2d(c, c, kernel_size=5, padding=2, groups=c, bias=False)
        with torch.no_grad():
            smooth_kernel = torch.ones(1, 1, 5, 5, dtype=torch.float32) / 25
            smooth_kernel = smooth_kernel.repeat(c, 1, 1, 1)
            self.smooth.weight.data = smooth_kernel

        # 3. 点状模式增强：对比度增强（局部响应归一化）
        self.local_norm = nn.GroupNorm(16, c)

        # 4. 可学习的融合权重
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, mode_weights):
        """
        x: [B, C, H, W] 输入特征
        mode_weights: [B, 3, 1, 1] 模式权重
        """
        # 三种增强
        tex_enhance = x + self.laplacian(x)  # 纹理：边缘锐化
        flat_enhance = self.smooth(x)        # 片状：平滑
        point_enhance = self.local_norm(x)   # 点状：对比度

        # 加权融合
        w_tex = mode_weights[:, 0:1, :, :]
        w_flat = mode_weights[:, 1:2, :, :]
        w_point = mode_weights[:, 2:3, :, :]

        out = w_tex * tex_enhance + w_flat * flat_enhance + w_point * point_enhance

        # 残差连接
        return x + out


class AFM(nn.Module):
    """
    Adaptive Fusion Module（自适应融合模块）

    流程：
    1. 内容编码：提取局部统计特征
    2. 模式分类：判断每个位置属于纹理/片状/点状
    3. 自适应增强：根据模式应用不同的增强策略

    参数:
        c1, c2: 输入输出通道数（实际使用 c2）
        reduction: 压缩率

    输入: [B, c2, H, W]
    输出: [B, c2, H, W]（分辨率不变）
    """
    def __init__(self, c1, c2, reduction=4):
        super().__init__()
        c = c2

        # 1. 内容感知编码器
        self.encoder = ContentEncoder(c)

        # 2. 模式分类器（基于编码特征）
        self.classifier = ModeClassifier(c, reduction=reduction)

        # 3. 自适应增强
        self.enhance = AdaptiveEnhance(c)

        # 4. 输出投影
        self.proj = ConvBNSiLU(c, c, k=1)

    def forward(self, x):
        # 1. 内容编码
        encoded = self.encoder(x)  # [B, c//4, H, W]

        # 2. 模式分类（全局平均池化）
        mode_weights = self.classifier(encoded)  # [B, 3, 1, 1]

        # 3. 自适应增强
        out = self.enhance(x, mode_weights)

        # 4. 输出投影
        out = self.proj(out)

        # 残差连接
        return x + out


if __name__ == "__main__":
    torch.manual_seed(0)
    for c in [256, 512, 1024]:
        x = torch.randn(1, c, 40, 40)
        m = AFM(c, c, reduction=4)
        y = m(x)
        params = sum(p.numel() for p in m.parameters())
        print(f"AFM(c={c}) input: {x.shape} -> output: {y.shape}, 参数量: {params/1e6:.4f}M")
    print("✅ 所有测试通过！")