# spp_sae.py — SAEAIF-SPPF（IGARSS 2025）
# SAEAIF-SPPF：挤压-聚合-激励（SAE）通道注意力 + 尺度内特征交互 + SPPF 多尺度池化
# 结构：SAE 通道激励 → 1x1 降维 → 串行最大池化（每级先与上级特征逐元素相加交互）
#       → 四尺度 concat 融合 → 输出；可直接替换 SPPF
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


class SAE(nn.Module):
    """
    Squeeze-Aggregation-Excitation 通道激励块
    输入: [B, C, H, W]
    输出: [B, C, H, W]（通道加权后的特征）
    """
    def __init__(self, c, branches=4, reduction=8):
        super().__init__()
        hidden = max(1, c // reduction)
        # 多分支挤压：多个池化描述符分支
        self.squeeze = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c, hidden, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
            ) for _ in range(branches)
        ])
        # 聚合 + 激励
        self.aggregate = nn.Conv2d(hidden * branches, hidden, kernel_size=1, bias=True)
        self.excite = nn.Conv2d(hidden, c, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        outs = [s(x) for s in self.squeeze]                  # 每个 [B, hidden, 1, 1]
        agg = self.aggregate(torch.cat(outs, dim=1))         # [B, hidden, 1, 1]
        w = self.sigmoid(self.excite(agg))                   # [B, C, 1, 1]
        return x * w


class SAEAIF_SPPF(nn.Module):
    """
    SAEAIF-SPPF（工程适配版，可直接替换 SPPF）

    参数:
        c1        : 输入通道数
        c2        : 输出通道数
        k         : 最大池化核大小，默认 5
        reduction : SAE 压缩率，默认 8

    输入: [B, c1, H, W]
    输出: [B, c2, H, W]
    """

    # ← reduction 从 8 改成 16
    def __init__(self, c1, c2, k=5, reduction=16):
        super().__init__()
        c_ = max(1, c1 // 2)
        # 1) 通道激励：branches 从 4 改成 2
        # 1) 通道激励
        self.sae = SAE(c1, branches=2, reduction=reduction)
        # 2) 降维
        self.cv1 = ConvBNSiLU(c1, c_, k=1, s=1)
        # 3) 串行最大池化
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        # 4) 尺度内交互：每级用 1x1 卷积融合交互结果
        self.cv_interact1 = ConvBNSiLU(c_, c_, k=1, s=1)
        self.cv_interact2 = ConvBNSiLU(c_, c_, k=1, s=1)
        self.cv_interact3 = ConvBNSiLU(c_, c_, k=1, s=1)
        # 5) 四尺度融合
        self.cv2 = ConvBNSiLU(c_ * 4, c2, k=1, s=1)

    def forward(self, x):
        # 通道激励
        x = self.sae(x)                                        # [B, c1, H, W]
        x = self.cv1(x)                                        # [B, c_, H, W]

        # 尺度内交互：每级 = 上级特征 + 池化特征 → 1x1 交互
        y1 = x
        p1 = self.m(y1)
        y2 = self.cv_interact1(y1 + p1)
        p2 = self.m(y2)
        y3 = self.cv_interact2(y2 + p2)
        p3 = self.m(y3)
        y4 = self.cv_interact3(y3 + p3)

        out = self.cv2(torch.cat([y1, y2, y3, y4], dim=1))     # [B, c2, H, W]
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1, 256, 20, 20)
    m = SAEAIF_SPPF(c1=256, c2=256, k=5, reduction=8)
    y = m(x)
    print("SAEAIF-SPPF input shape :", x.shape)
    print("SAEAIF-SPPF output shape:", y.shape)
