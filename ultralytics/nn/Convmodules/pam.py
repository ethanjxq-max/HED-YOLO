# pam.py — Progressive Aggregation Module（渐进式聚合模块）
# 设计目的：在 FEM 之后，渐进式聚合三个分支的特征
# 核心原则：不强制选择，保留所有信息

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


class IntraBranchRefinement(nn.Module):
    """
    阶段 1：分支内精炼
    每个分支用轻量卷积自己强化自己
    """
    def __init__(self, c):
        super().__init__()
        self.conv1 = ConvBNSiLU(c, c, k=1)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # 1x1 通道混合
        out = self.conv1(x)
        # 3x3 深度卷积（空间精炼）
        out = self.act(self.bn(self.conv2(out)))
        # 残差
        return x + out


class InterBranchInteraction(nn.Module):
    """
    阶段 2：分支间交互
    轻量级交叉注意力，让三个分支交换信息
    """
    def __init__(self, c, reduction=4):
        super().__init__()
        hidden = max(4, c // reduction)

        # 三个分支各自生成 query/key/value
        self.q = nn.Conv2d(c, hidden, kernel_size=1, bias=False)
        self.k = nn.Conv2d(c, hidden, kernel_size=1, bias=False)
        self.v = nn.Conv2d(c, c, kernel_size=1, bias=False)

        # 输出投影
        self.proj = ConvBNSiLU(c, c, k=1)

        self.scale = hidden ** -0.5

    def forward(self, x, context):
        """
        x: 当前分支特征 [B, C, H, W]
        context: 其他分支的特征（或融合特征）[B, C, H, W]
        """
        b, c, h, w = x.shape

        # Query: 当前分支
        q = self.q(x).view(b, -1, h * w).permute(0, 2, 1)  # [B, N, H]
        # Key/Value: 上下文分支
        k = self.k(context).view(b, -1, h * w)              # [B, H, N]
        v = self.v(context).view(b, c, h * w)               # [B, C, N]

        # 注意力
        attn = torch.bmm(q, k) * self.scale  # [B, N, N]
        attn = F.softmax(attn, dim=-1)

        # 聚合
        out = torch.bmm(v, attn.permute(0, 2, 1))  # [B, C, N]
        out = out.view(b, c, h, w)

        # 残差
        out = self.proj(out)
        return x + out


class ProgressiveFusion(nn.Module):
    """
    阶段 3：渐进式融合
    先融合纹理+结构，再融合全局
    """
    def __init__(self, c):
        super().__init__()
        # 第一步：纹理 + 结构
        self.fuse_ts = ConvBNSiLU(c * 2, c, k=1)
        # 第二步：TS + 全局
        self.fuse_tsg = ConvBNSiLU(c * 2, c, k=1)
        # 第三步：输出精炼
        self.refine = ConvBNSiLU(c, c, k=3)

    def forward(self, t, s, g):
        # 第一步：纹理 + 结构
        ts = self.fuse_ts(torch.cat([t, s], dim=1))  # [B, C, H, W]
        # 第二步：TS + 全局
        tsg = self.fuse_tsg(torch.cat([ts, g], dim=1))  # [B, C, H, W]
        # 第三步：精炼
        out = self.refine(tsg)
        return out


class PAM(nn.Module):
    """
    Progressive Aggregation Module（渐进式聚合模块）

    流程：
    1. 分支内精炼：每个分支自己强化
    2. 分支间交互：T↔S, T↔G, S↔G 双向交换信息
    3. 渐进式融合：先融合 T+S，再融合 G

    参数:
        c1, c2: 输入输出通道数（实际使用 c2）
        reduction: 压缩率

    输入: [B, c2, H, W] (FEM 的输出)
    输出: [B, c2, H, W]（分辨率不变）
    """
    def __init__(self, c1, c2, reduction=4):
        super().__init__()
        c = c2

        # 用于分支内精炼的 1x1 卷积（将输入映射到三个分支）
        # 注意：FEM 的输出已经是三个分支的融合，我们需要"解耦"出三个分支的特征
        # 这里用三个独立的 1x1 卷积来近似三个分支的响应
        self.branch_t = ConvBNSiLU(c, c, k=1)
        self.branch_s = ConvBNSiLU(c, c, k=1)
        self.branch_g = ConvBNSiLU(c, c, k=1)

        # 阶段 1：分支内精炼
        self.refine_t = IntraBranchRefinement(c)
        self.refine_s = IntraBranchRefinement(c)
        self.refine_g = IntraBranchRefinement(c)

        # 阶段 2：分支间交互（双向）
        self.interact_ts = InterBranchInteraction(c, reduction)
        self.interact_st = InterBranchInteraction(c, reduction)  # 对称，但参数独立
        self.interact_tg = InterBranchInteraction(c, reduction)
        self.interact_gt = InterBranchInteraction(c, reduction)
        self.interact_sg = InterBranchInteraction(c, reduction)
        self.interact_gs = InterBranchInteraction(c, reduction)

        # 阶段 3：渐进式融合
        self.fusion = ProgressiveFusion(c)

        # 输出投影
        self.proj = ConvBNSiLU(c, c, k=1)

    def forward(self, x):
        # ---- 分解为三个分支 ----
        # 用三个 1x1 卷积从融合特征中提取三种模式的响应
        t = self.branch_t(x)  # 纹理分支
        s = self.branch_s(x)  # 结构分支
        g = self.branch_g(x)  # 全局分支

        # ---- 阶段 1：分支内精炼 ----
        t = self.refine_t(t)
        s = self.refine_s(s)
        g = self.refine_g(g)

        # ---- 阶段 2：分支间交互 ----
        # T ↔ S
        t2 = self.interact_ts(t, s)  # T 从 S 获取信息
        s2 = self.interact_st(s, t)  # S 从 T 获取信息
        # T ↔ G
        t3 = self.interact_tg(t2, g)  # T 从 G 获取信息
        g2 = self.interact_gt(g, t2)  # G 从 T 获取信息
        # S ↔ G
        s3 = self.interact_sg(s2, g2)  # S 从 G 获取信息
        g3 = self.interact_gs(g2, s2)  # G 从 S 获取信息

        # ---- 阶段 3：渐进式融合 ----
        out = self.fusion(t3, s3, g3)

        # 输出投影 + 残差
        out = self.proj(out)
        return x + out


if __name__ == "__main__":
    torch.manual_seed(0)
    for c in [256, 512, 1024]:
        x = torch.randn(1, c, 40, 40)
        m = PAM(c, c, reduction=4)
        y = m(x)
        params = sum(p.numel() for p in m.parameters())
        print(f"PAM(c={c}) input: {x.shape} -> output: {y.shape}, 参数量: {params/1e6:.4f}M")
    print("✅ 所有测试通过！")