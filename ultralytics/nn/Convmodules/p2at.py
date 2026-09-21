# p2at.py — Pyramid Pooling Axial Transformer（ESWA 2024）
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


class AxialAttention(nn.Module):
    """
    轴向注意力：行方向 + 列方向两次 1D 注意力
    输入: [B, C, H, W]  输出: [B, C, H, W]
    """
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.hd = dim // heads
        self.scale = self.hd ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def _attn(self, q, k, v):
        # q,k,v: [B, H, D, L]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return attn @ v

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)                                        # [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)                            # 各 [B, C, H, W]

        # 行方向注意力：沿 W 轴
        q_h = q.reshape(b, self.heads, self.hd, h, w).permute(0, 1, 3, 2, 4)
        k_h = k.reshape(b, self.heads, self.hd, h, w).permute(0, 1, 3, 2, 4)
        v_h = v.reshape(b, self.heads, self.hd, h, w).permute(0, 1, 3, 2, 4)
        out_h = self._attn(q_h, k_h, v_h)                        # [B, H, h, D, w]
        out_h = out_h.permute(0, 1, 3, 2, 4).reshape(b, c, h, w)

        # 列方向注意力：沿 H 轴
        q_w = q.reshape(b, self.heads, self.hd, h, w).permute(0, 1, 4, 2, 3)
        k_w = k.reshape(b, self.heads, self.hd, h, w).permute(0, 1, 4, 2, 3)
        v_w = v.reshape(b, self.heads, self.hd, h, w).permute(0, 1, 4, 2, 3)
        out_w = self._attn(q_w, k_w, v_w)                        # [B, H, w, D, h]
        out_w = out_w.permute(0, 1, 3, 4, 2).reshape(b, c, h, w)

        return self.proj(out_h + out_w)


class P2AT(nn.Module):
    """
    Pyramid Pooling Axial Transformer（工程适配版）
    金字塔池化（多尺度上下文） + 轴向注意力（长距离依赖）
    参数:
        c1          : 输入通道数
        c2          : 输出通道数
        pool_scales : 金字塔池化尺度，默认 (1, 2, 3, 6)
        heads       : 轴向注意力头数
    输入: [B, c1, H, W]  输出: [B, c2, H, W]
    """
    def __init__(self, c1, c2, pool_scales=(1, 2, 3, 6), heads=4):
        super().__init__()
        self.cv_in = ConvBNSiLU(c1, c2, k=1, s=1) if c1 != c2 else nn.Identity()

        # 金字塔池化：每个尺度一个 1x1 卷积
        self.pyramid_convs = nn.ModuleList([
            ConvBNSiLU(c2, c2, k=1, s=1) for _ in pool_scales
        ])
        self.pool_scales = pool_scales

        # 轴向注意力
        self.axial = AxialAttention(c2, heads=heads)

        # 融合金字塔特征与注意力特征
        self.cv_fuse = ConvBNSiLU(c2 * 2, c2, k=1, s=1)

    def forward(self, x):
        x = self.cv_in(x)                                        # [B, c2, H, W]
        b, c, h, w = x.shape

        # 金字塔池化分支
        pys = []
        for scale, conv in zip(self.pool_scales, self.pyramid_convs):
            ps = F.adaptive_avg_pool2d(x, (h // scale, w // scale))
            ps = conv(ps)
            ps = F.interpolate(ps, size=(h, w), mode="bilinear", align_corners=False)
            pys.append(ps)
        py = sum(pys)                                            # [B, c2, H, W]

        # 轴向注意力分支
        ax = self.axial(x)                                       # [B, c2, H, W]

        out = self.cv_fuse(torch.cat([py, ax], dim=1))           # [B, c2, H, W]
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1, 256, 40, 40)
    m = P2AT(c1=256, c2=256, pool_scales=(1, 2, 3, 6), heads=4)
    y = m(x)
    print("P2AT input shape :", x.shape)
    print("P2AT output shape:", y.shape)