# aif_sppf.py — AIF-SPPF（Adaptive Interaction Fusion SPPF）
# 设计理念：去掉 SAE，用差分激励保留尺度间差异，用残差连接防止信息丢失

import torch
import torch.nn as nn


class ConvBNSiLU(nn.Module):
    """基础卷积块：Conv + BN + SiLU"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DifferenceStimulus(nn.Module):
    """
    差分激励模块：计算相邻尺度之间的差异，保留多尺度信息
    """
    def __init__(self, c):
        super().__init__()
        # 用 1x1 卷积增强差分信号
        self.conv = ConvBNSiLU(c, c, k=1, s=1)

    def forward(self, high_res, low_res):
        """
        high_res: 高分辨率特征（上一级）
        low_res: 低分辨率特征（池化后）
        返回: 差分激励 + 残差连接
        """
        # 低分辨率上采样到高分辨率
        low_res_up = nn.functional.interpolate(
            low_res, size=high_res.shape[2:], mode='bilinear', align_corners=False
        )
        # 差分 = 高分辨率 - 低分辨率（保留细节差异）
        diff = high_res - low_res_up
        # 激励增强
        diff = self.conv(diff)
        # 残差连接：原始高分辨率 + 差分激励
        return high_res + diff


class AIF_SPPF(nn.Module):
    """
    Adaptive Interaction Fusion SPPF
    参数:
        c1: 输入通道
        c2: 输出通道
        k: 池化核大小
    输入: [B, c1, H, W]
    输出: [B, c2, H, W]
    """
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = max(1, c1 // 2)  # 降维通道数

        # 1) 降维
        self.cv1 = ConvBNSiLU(c1, c_, k=1, s=1)

        # 2) 最大池化
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        # 3) 差分激励模块（每级一个）
        self.diff1 = DifferenceStimulus(c_)
        self.diff2 = DifferenceStimulus(c_)
        self.diff3 = DifferenceStimulus(c_)

        # 4) 输出融合（4个尺度 concat → 输出）
        self.cv2 = ConvBNSiLU(c_ * 4, c2, k=1, s=1)

    def forward(self, x):
        x = self.cv1(x)  # [B, c_, H, W]

        # 第一级：原始特征
        y1 = x

        # 第二级：池化 + 差分激励
        p1 = self.m(y1)
        y2 = self.diff1(y1, p1)

        # 第三级：池化 + 差分激励
        p2 = self.m(y2)
        y3 = self.diff2(y2, p2)

        # 第四级：池化 + 差分激励
        p3 = self.m(y3)
        y4 = self.diff3(y3, p3)

        # 四尺度融合
        out = self.cv2(torch.cat([y1, y2, y3, y4], dim=1))
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1, 256, 20, 20)
    m = AIF_SPPF(c1=256, c2=256, k=5)
    y = m(x)
    print(f"AIF-SPPF input: {x.shape} -> output: {y.shape}")
    params = sum(p.numel() for p in m.parameters())
    print(f"参数量: {params/1e6:.3f}M")