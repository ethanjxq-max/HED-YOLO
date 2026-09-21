# caa.py — Context Anchor Attention (ESWA 2024)
# 参考论文：LSKA 同源模块，用于 Neck 特征增强
# 设计理念：可学习锚点聚合全局上下文，轻量级注意力

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CAA(nn.Module):
    """
    Context Anchor Attention（上下文锚点注意力）

    核心思想：
    1. 用可学习的锚点（anchor）聚合全局上下文
    2. 通过锚点和特征图之间的注意力，动态增强特征

    参数:
        c1          : 输入通道数
        c2          : 输出通道数
        num_anchors : 锚点数量，默认 4
        reduction   : 压缩率，默认 4

    输入: [B, c1, H, W]
    输出: [B, c2, H, W]（分辨率不变）
    """
    def __init__(self, c1, c2, num_anchors=4, reduction=4):
        super().__init__()

        # 输入输出通道适配
        self.cv_in = ConvBNSiLU(c1, c2, k=1, s=1) if c1 != c2 else nn.Identity()

        # 可学习的锚点：每个锚点是一个 c2 维向量
        self.anchors = nn.Parameter(torch.randn(1, num_anchors, c2) * 0.02)

        # 锚点注意力：计算每个空间位置对每个锚点的响应
        # 先压缩通道
        hidden = max(4, c2 // reduction)
        self.q = nn.Conv2d(c2, hidden, kernel_size=1, bias=False)
        self.k = nn.Linear(c2, hidden, bias=False)  # 锚点作为 key
        self.v = nn.Linear(c2, c2, bias=False)      # 锚点作为 value

        # 输出投影
        self.proj = ConvBNSiLU(c2, c2, k=1, s=1)

        # 残差门控
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x = self.cv_in(x)  # [B, C, H, W]
        b, c, h, w = x.shape
        n_anchors = self.anchors.shape[1]

        # 1) 从特征图生成 query
        q = self.q(x)  # [B, H, H, W]，其中 H = c // reduction
        q = q.view(b, -1, h * w).permute(0, 2, 1)  # [B, N, H]，N = H*W

        # 2) 锚点作为 key 和 value
        anchors = self.anchors.expand(b, -1, -1)  # [B, num_anchors, C]
        k = self.k(anchors)  # [B, num_anchors, H]
        v = self.v(anchors)  # [B, num_anchors, C]

        # 3) 计算注意力：每个空间位置对每个锚点的响应
        # q: [B, N, H], k: [B, num_anchors, H]
        attn = torch.bmm(q, k.transpose(1, 2)) / (c ** 0.5)  # [B, N, num_anchors]
        attn = F.softmax(attn, dim=-1)  # [B, N, num_anchors]

        # 4) 聚合锚点 value：每个位置 = 所有锚点的加权和
        # attn: [B, N, num_anchors], v: [B, num_anchors, C]
        out = torch.bmm(attn, v)  # [B, N, C]
        out = out.permute(0, 2, 1).view(b, c, h, w)  # [B, C, H, W]

        # 5) 输出投影 + 残差门控
        out = self.proj(out)
        gate = self.gate(out)
        out = x * (1 - gate) + out * gate  # 自适应残差

        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1, 512, 40, 40)
    m = CAA(c1=512, c2=512, num_anchors=4, reduction=4)
    y = m(x)
    print(f"CAA input: {x.shape} -> output: {y.shape}")
    params = sum(p.numel() for p in m.parameters())
    print(f"参数量: {params/1e6:.3f}M")