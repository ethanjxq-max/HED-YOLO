# eaup.py — EA-Up：Edge-Aware Upsampling（边缘感知细节补偿上采样）
# ==================================================================
# 研发背景（2026-09-11，基于 FreqFusion(TPAMI 2024, 清单F2) 精读 + 自有 24 组实验）：
#   FreqFusion 的观察：标准 FPN 融合（nearest 上采样 + 相加）存在边界模糊
#   （下采样丢高频）与类内不一致（上采样引入高频扰动）；其 ALPF 生成器用
#   空间可变低通滤波"平滑"高层特征来解决类内不一致（+0.9 mIoU，检测 +1.9 AP）。
#
#   ⚠️ 场景差异（本模块的自研出发点）：
#     FreqFusion 的 ALPF 设计目标是"抹掉大目标内部的高频扰动"（COCO 的
#     公交/卡车内部纹理）；而 NEU-DET 的 crazing(细网纹 1~3px)/scratches
#     (细划线) 本身就是高频信号——"处处平滑"会直接抹掉缺陷细节。
#     因此本模块**反向设计**：不做平滑，改做"边缘/细节感知的锐度补偿"——
#     只在细节显著处（边缘、细纹）补偿上采样丢失的高频，平坦区不补偿
#     （避免放大背景噪声，吸取 WDM"无差别高频注入"的失败教训）。
#
# 机制（替换 head 融合路径中的 nn.Upsample，输入输出形状不变）：
#   x [B,C,H,W]（高层语义特征）
#   ├─ 基础上采样：nearest ×2 → U [B,C,2H,2W]（与原网络行为一致）
#   ├─ 细节提取：hf = x − AvgPool5(x)（无参，高层自身带通细节）
#   ├─ 边缘门控：E = |hf| 通道均值 → 相对能量归一化 → g = σ(s·(E−b))
#   │            （s、b 为可学习标量；细节强处 g→1，平坦处 g→0）
#   └─ out = U + γ ⊙ ( g ⊙ bilinear(hf) )
#        γ 为通道门控，**零初始化 → 训练起步严格等价于原 nearest 上采样**
#        （涨多少完全由数据决定，杜绝"开局伤特征"）
#
# 设计依据：
#   1) FreqFusion（TPAMI 2024）：上采样融合是检测性能的关键点（其检测实验
#      只替换 FPN 上采样即 +1.9 AP、APS +1.7）——位置选择有文献背书；
#   2) 其消融（Table 12）：上采样端改造贡献 +0.9~1.8 mIoU，值得做；
#   3) 自有教训：WDM 无差别高频注入失败（背景噪声被放大）、LDTE 方向注入
#      失败（类间跷跷板）→ 本模块用"零初始化 + 细节能量门控"把补偿限制在
#      细节显著区，且起步与原网络严格一致；
#   4) v9s 的 AConv 保信息下采样（自有分析）：同属"采样质量"层级改进。
#
# 用法（yaml，替换 head 里的 nn.Upsample；数字按缩放前写，模块内部用 c1）：
#   - [-1, 1, EAUp, [512]]   # 原 [-1, 1, nn.Upsample, [None, 2, "nearest"]]
#   - [-1, 1, EAUp, [256]]
# 参数量（c=256 实测）：γ 256 + s/b 2 ≈ 258 参数/处；计算量 ≈ O(C·H·W)（可忽略）
# 消融开关：use_gate=False（无边缘门控，全局补偿）；use_detail=False（纯 nearest）
# 只依赖 torch，不需要 import ultralytics。
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["EAUp"]


class EAUp(nn.Module):
    """
    输入 [B,C,H,W] → 输出 [B,C,2H,2W]（与 nn.Upsample(scale_factor=2) 等价替换）

    参数:
        c1: 输入通道数（ultralytics 自动传入，输出通道 = c1）
        c2: 兼容占位（yaml 第二参数），不参与计算
        blur_k: 细节提取的平滑窗（默认 5）
        use_gate: 边缘门控开关（默认 True；False = 全局补偿，消融用）
        use_detail: 细节补偿总开关（默认 True；False = 纯 nearest 上采样，消融用）
    """

    def __init__(self, c1, c2=None, blur_k=5, use_gate=True, use_detail=True):
        super().__init__()
        c = c1
        self.use_gate = use_gate
        self.use_detail = use_detail
        self.blur = nn.AvgPool2d(kernel_size=blur_k, stride=1, padding=blur_k // 2)
        # 通道门控：零初始化 → 起步 = 原 nearest 上采样
        self.gamma = nn.Parameter(torch.zeros(c, 1, 1))
        # 边缘门控参数：初始 s=2, b=1 → g = σ(2(E/Ē−1))，细节强处才补偿
        self.edge_scale = nn.Parameter(torch.tensor(2.0))
        self.edge_bias = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        up = F.interpolate(x, scale_factor=2, mode="nearest")  # 与原网络一致的起步
        if not self.use_detail:
            return up

        hf = x - self.blur(x)  # [B,C,H,W] 高层自身带通细节（无参）

        if self.use_gate:
            e = hf.abs().mean(dim=1, keepdim=True)  # [B,1,H,W] 细节能量
            e = e / (e.mean(dim=(2, 3), keepdim=True) + 1e-5)  # 相对能量（均值≈1）
            g = torch.sigmoid(self.edge_scale * (e - self.edge_bias))  # 细节处→1，平坦处→0
            g = F.interpolate(g, scale_factor=2, mode="nearest")
        else:
            g = 1.0

        hf_up = F.interpolate(hf, scale_factor=2, mode="bilinear", align_corners=False)
        return up + self.gamma * (g * hf_up)


if __name__ == "__main__":
    torch.manual_seed(0)
    c = 256
    x = torch.randn(2, c, 40, 40)

    # 1) 起步等价性：γ=0 → 输出 == nearest 上采样
    m = EAUp(c, c)
    y = m(x)
    ref = F.interpolate(x, scale_factor=2, mode="nearest")
    print(f"形状: {x.shape} -> {y.shape}")
    print(f"γ=0 与 nearest 的误差: {(y - ref).abs().max().item():.2e}")
    assert y.shape == (2, c, 80, 80), "输出分辨率错误！"
    assert torch.allclose(y, ref, atol=1e-6), "γ 零初始化应等价于 nearest！"

    # 2) 学习通路：γ、s、b 均有梯度
    m.gamma.data.fill_(0.3)
    y2 = m(x)
    y2.pow(2).mean().backward()
    grads = {
        "gamma": m.gamma.grad.abs().sum().item(),
        "edge_scale": m.edge_scale.grad.abs().sum().item(),
        "edge_bias": m.edge_bias.grad.abs().sum().item(),
    }
    print(f"梯度量级: { {k: round(v, 5) for k, v in grads.items()} }")
    assert all(v > 0 for v in grads.values()), "存在无梯度的部件！"

    # 3) 边缘门控行为：门控图应有空间动态（强纹理区 g 高、平坦区 g 低）
    with torch.no_grad():
        # 构造"左半平坦 + 右半强条纹"的合成输入
        inp = torch.randn(1, c, 40, 40) * 0.1
        yy = torch.arange(40, dtype=torch.float32).view(1, 1, 40, 1)
        stripe = torch.sin(6 * yy) * 2.0
        inp[:, :, :, 20:] = inp[:, :, :, 20:] + stripe
        hf = inp - m.blur(inp)
        e = hf.abs().mean(1, keepdim=True)
        e = e / (e.mean((2, 3), keepdim=True) + 1e-5)
        g = torch.sigmoid(m.edge_scale * (e - m.edge_bias))
        g_left, g_right = g[:, :, :, :20].mean().item(), g[:, :, :, 20:].mean().item()
        print(f"  门控空间动态: 平坦区 g={g_left:.3f} vs 强纹理区 g={g_right:.3f} (std={g.std().item():.3f})")
        assert g_right > g_left, "门控未对强纹理区响应！"
        assert g.std().item() > 1e-3, "门控退化为常数图！"

    # 4) 消融变体前向
    for kw in (dict(use_gate=False), dict(use_detail=False)):
        m2 = EAUp(c, c, **kw)
        assert m2(x).shape == (2, c, 80, 80), f"变体 {kw} 形状错误"

    # 5) 参数量
    n = sum(p.numel() for p in m.parameters())
    print(f"EAUp(c={c}) 参数量: {n} ({n/1e3:.3f} K)")
    print("✅ 所有测试通过！")
