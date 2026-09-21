# fem.py — Feature Enhancement Module（特征增强模块）
# 设计目的：在 DB 输出后，针对纹理/结构缺陷进行定向增强
# 三分支设计：纹理增强 + 结构保持 + 全局上下文

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


class TextureEnhance(nn.Module):
    """
    纹理增强分支：针对线状/网状缺陷（crazing、scratches）
    使用可学习的边缘检测卷积 + 残差连接
    """

    def __init__(self, c, reduction=4):
        super().__init__()
        # 可学习的边缘检测核（4个方向：水平、垂直、对角线1、对角线2）
        # 使用分组卷积，每组独立学习
        self.edge_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)

        # 用 Sobel 初始化
        with torch.no_grad():
            # 4个方向的 Sobel 核
            sobel_h = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            sobel_v = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            sobel_d1 = torch.tensor([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=torch.float32)
            sobel_d2 = torch.tensor([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=torch.float32)

            # 堆叠成 [4, 1, 3, 3]
            kernels = torch.stack([sobel_h, sobel_v, sobel_d1, sobel_d2], dim=0)  # [4, 3, 3]
            kernels = kernels.unsqueeze(1)  # [4, 1, 3, 3]

            # 重复到 c 个通道
            # 对于分组卷积，权重形状是 [out_channels, in_channels//groups, k, k]
            # 这里 groups=c，所以 in_channels//groups = 1
            # 需要 [c, 1, 3, 3]
            if c >= 4:
                # 先重复 4 个核到 c 个
                repeat_times = c // 4
                remaining = c % 4
                kernel_list = []
                for _ in range(repeat_times):
                    kernel_list.append(kernels)
                if remaining > 0:
                    kernel_list.append(kernels[:remaining])
                final_kernel = torch.cat(kernel_list, dim=0)  # [c, 1, 3, 3]
            else:
                final_kernel = kernels[:c]  # [c, 1, 3, 3]

            # 确保形状匹配
            if final_kernel.shape[0] != c:
                # 补全到 c
                pad = c - final_kernel.shape[0]
                final_kernel = torch.cat([final_kernel, final_kernel[:pad]], dim=0)

            self.edge_conv.weight.data = final_kernel

        # 通道压缩 + 恢复（减少计算量）
        hidden = max(4, c // reduction)
        self.compress = ConvBNSiLU(c, hidden, k=1)
        self.expand = ConvBNSiLU(hidden, c, k=1)

        # 残差门控
        self.gate = nn.Sigmoid()

    def forward(self, x):
        # 边缘检测
        edge = self.edge_conv(x)
        edge = torch.abs(edge)  # 取绝对值，响应方向无关

        # 通道压缩 + 恢复
        edge = self.compress(edge)
        edge = self.expand(edge)

        # 门控残差：只增强"有纹理"的区域
        gate = self.gate(edge)
        return x + edge * gate


class StructurePreserve(nn.Module):
    """
    结构保持分支：针对片状/块状缺陷（rolled-in_scale、patches）
    使用多尺度空洞卷积，保持面状结构的完整性
    """

    def __init__(self, c, dilations=[1, 2, 3], reduction=4):
        super().__init__()
        hidden = max(4, c // reduction)

        # 多尺度空洞卷积（不改变分辨率）
        self.dilated_1 = nn.Conv2d(hidden, hidden, kernel_size=3, dilation=dilations[0], padding=dilations[0],
                                   groups=hidden)
        self.dilated_2 = nn.Conv2d(hidden, hidden, kernel_size=3, dilation=dilations[1], padding=dilations[1],
                                   groups=hidden)
        self.dilated_3 = nn.Conv2d(hidden, hidden, kernel_size=3, dilation=dilations[2], padding=dilations[2],
                                   groups=hidden)

        self.compress = ConvBNSiLU(c, hidden, k=1)
        self.fuse = ConvBNSiLU(hidden * 3, hidden, k=1)
        self.expand = ConvBNSiLU(hidden, c, k=1)

        self.gate = nn.Sigmoid()

    def forward(self, x):
        x_comp = self.compress(x)  # [B, hidden, H, W]

        # 多尺度空洞卷积
        d1 = self.dilated_1(x_comp)
        d2 = self.dilated_2(x_comp)
        d3 = self.dilated_3(x_comp)

        # 融合
        fused = self.fuse(torch.cat([d1, d2, d3], dim=1))
        out = self.expand(fused)

        gate = self.gate(out)
        return x + out * gate


class GlobalContext(nn.Module):
    """
    全局上下文分支：轻量级全局信息补充
    用多尺度池化 + 1x1 卷积，不叠加大计算量
    """

    def __init__(self, c, pool_scales=[1, 2, 4]):
        super().__init__()
        self.pool_convs = nn.ModuleList([
            ConvBNSiLU(c, c // 4, k=1) for _ in pool_scales
        ])
        self.pool_scales = pool_scales
        self.fuse = ConvBNSiLU(c // 4 * len(pool_scales), c, k=1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        outs = []
        for scale, conv in zip(self.pool_scales, self.pool_convs):
            # 池化到不同尺度
            ph, pw = max(1, h // scale), max(1, w // scale)
            pooled = F.adaptive_avg_pool2d(x, (ph, pw))
            pooled = conv(pooled)
            pooled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
            outs.append(pooled)

        out = self.fuse(torch.cat(outs, dim=1))
        gate = self.gate(out)
        return x + out * gate


class FEM(nn.Module):
    """
    Feature Enhancement Module（特征增强模块）

    三分支设计：
    1. 纹理增强（TextureEnhance）：针对线状/网状缺陷
    2. 结构保持（StructurePreserve）：针对片状/块状缺陷
    3. 全局上下文（GlobalContext）：补充感受野

    参数:
        c1: 输入通道数（由 tasks.py 自动传入，实际用 c2）
        c2: 输出通道数
        reduction: 压缩率（控制计算量）

    输入: [B, c2, H, W]
    输出: [B, c2, H, W]（分辨率不变）
    """
    def __init__(self, c1, c2, reduction=4):  # ← 改成 3 个参数，c1 保留但不用
        super().__init__()
        c = c2  # 使用 c2 作为通道数
        self.tex = TextureEnhance(c, reduction=reduction)
        self.struct = StructurePreserve(c, reduction=reduction)
        self.glob = GlobalContext(c)

        # 自适应融合权重（通道注意力）
        self.fusion_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c * 3, max(4, c // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, c // reduction), 3, 1),
            nn.Softmax(dim=1)
        )

        # 最终投影
        self.proj = ConvBNSiLU(c, c, k=1)

    def forward(self, x):
        # 三个分支
        t_out = self.tex(x)
        s_out = self.struct(x)
        g_out = self.glob(x)

        # 生成融合权重
        concat_feat = torch.cat([t_out, s_out, g_out], dim=1)
        weights = self.fusion_weight(concat_feat)

        # 加权融合
        out = weights[:, 0:1, :, :] * t_out + \
              weights[:, 1:2, :, :] * s_out + \
              weights[:, 2:3, :, :] * g_out

        out = self.proj(out)
        return x + out


if __name__ == "__main__":
    torch.manual_seed(0)
    for c in [256, 512, 1024]:
        x = torch.randn(1, c, 40, 40)
        m = FEM(c, c, reduction=4)  # ← 改成 3 个参数
        y = m(x)
        params = sum(p.numel() for p in m.parameters())
        print(f"FEM(c={c}) input: {x.shape} -> output: {y.shape}, 参数量: {params/1e6:.4f}M")
    print("✅ 所有测试通过！")