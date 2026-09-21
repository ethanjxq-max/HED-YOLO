# wsdm.py — WSDM：Wavelet Shrinkage & Spatial-selective Detail Module
#         （小波收缩·空间选择性细节模块）
# ------------------------------------------------------------------
# 针对问题：实验A（WDM@P3, 0.734/0.381）的类级分析——纯高频注入在
#   crazing(+1.3)/rolled-in_scale(+3.1)/patches(+2.1) 等纹理/线状类上
#   涨 mAP50-95，却在 pitted(-2.5)/inclusion(-0.6)/scratches(-1.1) 上跌，
#   整体 mAP50 掉 1.2。根因假说：80×80 分辨率下钢材表面的磨痕/轧制纹理
#   背景噪声被"无差别高频注入"一并放大 → 块状类误检、光滑区域被污染。
#
# 设计依据（2024-2026 文献 + 自有实验数据）：
#   1) FreqFusion（TPAMI 2024）：融合特征含两类高频——"破坏性高频"
#      （目标内部/光滑区/纹理背景噪声）与"边界有效高频"；应对是
#      **逐像素自适应频域选择**（ALPF 压噪声 + AHPF 保边界），而非
#      全局恒定的高频增强 → 本模块加入"空间选择性"机制；
#   2) DWWA-Net（TNNLS 2024）：小波域滤波专职去背景噪声、注意力专职
#      指向弱缺陷，两个角色解耦 → 本模块把"收缩(去噪)"与"空间门控
#      (定位)"分开实现、各自可开关；
#   3) Donoho 软阈值收缩（经典小波去噪）：参数化后嵌入网络，只放大
#      超过阈值的强结构、抑制弱噪声。
#
# 机制（相对 WDM 只加两个轻量部件，均可独立开关做消融）：
#   x ─DWT(固定Haar)→ {LL, LH, HL, HH}（40×40）
#   细节子带 concat → 1×1(3c→h) → DW3×3 → 1×1(h→3c) → 拆回 3 子带
#     │
#     ├─[可开关] 空间门控 g(x,y) = σ(1×1 conv(3c→1)(精炼后子带))   ← 定位"该增强的位置"
#     ├─[可开关] 每通道软阈值收缩 τ_c（子带域）                     ← 抑制"弱噪声"
#     │          （可选 use_hh=False 丢弃对角子带，方向选择的廉价变体）
#     ▼
#   IDWT(0, g·s(LH), g·s(HL), g·s(HH)) → 高频细节残差
#   out = x + γ ⊙ detail                （γ 通道门控，零初始化=恒等起步）
#
# 用法（yaml，ultralytics 自动传入 c1；数字按缩放前写，与 FEM/WDM 行一致）：
#   [-1, 1, WSDM, [256]]      # P3 头输出后（yolo26s 实际宽度 128）
# 注册：Convmodules/__init__.py 加 from .wsdm import *；tasks.py 的 conv_modules 加 WSDM
#       （不加 conv_repeat_modules——它没有 n 参数）
# 变体（论文消融用，改构造参数即可）：
#   WSDM(c1, shrink=False)    # 只测空间门控
#   WSDM(c1, spatial=False)   # 只测软阈值收缩（≈原 WSDM v1 设计）
#   WSDM(c1, use_hh=False)    # 丢弃对角子带（背景噪声能量最高的廉价假设）
# 只依赖 torch，不需要 import ultralytics。
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["WSDM"]


# ---------------- 固定 Haar 正交小波（无学习参数，与 WDM 相同） ----------------
def _haar_kernels():
    """4 个正交 Haar 核 [LL, LH, HL, HH]，各 [1,1,2,2]，已归一化(×0.5)。"""
    k = torch.tensor(
        [
            [[[1.0, 1.0], [1.0, 1.0]]],    # LL  低频近似
            [[[-1.0, 1.0], [-1.0, 1.0]]],  # LH  水平细节
            [[[-1.0, -1.0], [1.0, 1.0]]],  # HL  垂直细节
            [[[1.0, -1.0], [-1.0, 1.0]]],  # HH  对角细节
        ],
        dtype=torch.float32,
    ) * 0.5
    return k  # [4, 1, 2, 2]


class _DWT(nn.Module):
    """分组 Haar 分解：x[B,c,H,W] → (ll, lh, hl, hh)，各 [B,c,H/2,W/2]。"""

    def __init__(self, c):
        super().__init__()
        self.register_buffer("w", _haar_kernels().repeat(c, 1, 1, 1))  # [4c,1,2,2]

    def forward(self, x):
        out = F.conv2d(x, self.w, stride=2, groups=x.shape[1])  # [B,4c,H/2,W/2]
        return out.chunk(4, dim=1)


class _IDWT(nn.Module):
    """分组 Haar 合成：4 个子带 [B,c,H,W] → [B,c,2H,2W]。Haar 正交 → 逆核 = 正核。"""

    def __init__(self, c):
        super().__init__()
        self.register_buffer("w", _haar_kernels().repeat(c, 1, 1, 1))  # [4c,1,2,2]

    def forward(self, x):
        return F.conv_transpose2d(x, self.w, stride=2, groups=x.shape[1] // 4)


# ---------------- 机制①：可学习软阈值收缩（抑制弱噪声） ----------------
class SoftShrink(nn.Module):
    """
    每通道软阈值收缩（Donoho soft-threshold 的可学习版本）：
        y = sign(x) · ReLU(|x| − τ_c)
    τ_c 初始 ≈ 0.05（≈ 恒等），训练自动决定每个通道保留多少高频：
    τ_c 越大 = 该通道越"挑食"，只放大强结构（裂纹/划痕边缘），
    背景磨痕、块状内部等弱高频被收缩掉。
    数值安全：τ = softplus(raw) 恒 ≥ 0。
    """

    def __init__(self, c):
        super().__init__()
        self.raw = nn.Parameter(torch.full((c, 1, 1), -3.0))  # τ ≈ 0.049 起步，梯度可观

    @property
    def tau(self):
        return F.softplus(self.raw)

    def forward(self, x):
        t = self.tau
        return torch.sign(x) * torch.relu(x.abs() - t)


# ---------------- 机制②：空间门控（逐位置选择"该增强哪里"） ----------------
class SpatialGate(nn.Module):
    """
    小波域逐位置门控（FreqFusion TPAMI2024 的"自适应选择"思想的轻量版）：
        g(x,y) = σ( 1×1 conv(3c→1) (精炼后的三个细节子带) )
    输入取"精炼后、收缩前"的子带 → 门控看到的是完整高频信息，学会按
    跨通道模式区分"缺陷边界高频"与"背景纹理高频"；输出一张 40×40 的
    [0,1] 门控图，逐位置决定细节注入强度。
    初始化 bias=0 → g ≡ 0.5（中性起步；配合 γ=0 恒等，无开局风险）。
    """

    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c * 3, 1, kernel_size=1, bias=True)
        # 中性但非退化初始化：bias=0（均值≈0.5），权重小随机（门控图从一开始就有空间差异）
        nn.init.normal_(self.conv.weight, std=0.05)
        nn.init.zeros_(self.conv.bias)

    def forward(self, rlh, rhl, rhh):
        # 输入为精炼后的子带（收缩前）：[B,c,H/2,W/2] × 3
        g = torch.sigmoid(self.conv(torch.cat([rlh, rhl, rhh], dim=1)))  # [B,1,H/2,W/2]
        return g


class WSDM(nn.Module):
    """
    Wavelet Shrinkage & Spatial-selective Detail Module
    （小波收缩·空间选择性细节模块，WDM 的修复版 v2）

    相对 WDM 新增两个可开关的轻量机制：
      ① SoftShrink   —— 每通道软阈值收缩，抑制背景弱噪声（去噪角色）；
      ② SpatialGate  —— 小波域逐位置门控，选择"该增强的位置"（定位角色）。
    两者解耦（DWWA-Net 思路），可分别关闭做消融。

    参数:
        c1: 输入通道数（ultralytics 自动传入；内部宽度一律以 c1 为准）
        c2: 兼容占位（yaml 第二参数），可不用
        reduction: 精炼分支隐藏通道压缩率，默认 4
        shrink: 是否启用软阈值收缩（默认 True）
        spatial: 是否启用空间门控（默认 True）
        use_hh: 是否保留对角子带 HH（默认 True；False = 丢弃对角高频，
                若 NEU 背景噪声集中在 HH 上会立即见效，属廉价假设测试）

    输入/输出: [B, c, H, W] → [B, c, H, W]（分辨率不变）
    参数量（c=128 实测）：≈ 25.8K（WDM 25.1K + 128 阈值 + 385 门控）
    """

    def __init__(self, c1, c2=None, reduction=4, shrink=True, spatial=True, use_hh=True):
        super().__init__()
        c = c1
        h = max(4, c // reduction)
        self.use_hh = use_hh
        self.shrink_on = shrink
        self.spatial_on = spatial

        self.dwt = _DWT(c)
        self.idwt = _IDWT(c)
        # 高频细节精炼（与 WDM 相同）：3c → h → DW3x3 → 3c
        self.refine = nn.Sequential(
            nn.Conv2d(c * 3, h, 1, bias=False),
            nn.BatchNorm2d(h),
            nn.SiLU(inplace=True),
            nn.Conv2d(h, h, 3, padding=1, groups=h, bias=False),
            nn.BatchNorm2d(h),
            nn.SiLU(inplace=True),
            nn.Conv2d(h, c * 3, 1, bias=False),
        )
        # 机制①：每通道软阈值收缩（三个子带共享同一组 τ_c）
        self.shrink = SoftShrink(c)
        # 机制②：空间门控（输入精炼后子带 → 逐位置 [0,1]）
        self.spatial = SpatialGate(c)
        # 通道级输出门控：零初始化 → 初始输出 = 恒等映射
        self.gate = nn.Parameter(torch.zeros(c, 1, 1))

    def forward(self, x):
        h0, w0 = x.shape[-2:]
        if h0 % 2 or w0 % 2:  # 防御：奇数分辨率自动补齐再裁回
            x = F.pad(x, (0, w0 % 2, 0, h0 % 2))
        ll, lh, hl, hh = self.dwt(x)
        rlh, rhl, rhh = self.refine(torch.cat([lh, hl, hh], dim=1)).chunk(3, dim=1)

        # 机制②：空间门控（用收缩前的精炼子带判断"哪里该增强"）
        if self.spatial_on:
            g = self.spatial(rlh, rhl, rhh)  # [B,1,H/2,W/2]，对三个子带共享
            rlh = g * rlh
            rhl = g * rhl
            rhh = g * rhh

        # 机制①：每通道软阈值收缩（抑制弱噪声）
        if self.shrink_on:
            rlh = self.shrink(rlh)
            rhl = self.shrink(rhl)
            rhh = self.shrink(rhh)

        if not self.use_hh:  # 丢弃对角子带（廉价方向选择变体）
            rhh = torch.zeros_like(rhh)

        # 纯高频细节残差：LL 置零，只让处理后的细节子带参与逆变换
        detail = self.idwt(torch.cat([torch.zeros_like(ll), rlh, rhl, rhh], dim=1))
        out = x + self.gate * detail
        return out[..., :h0, :w0]


if __name__ == "__main__":
    torch.manual_seed(0)
    c = 128
    x = torch.randn(2, c, 80, 80)

    # 1) 零初始化门控 → 输出应等于输入（恒等起步）
    m = WSDM(c, c)
    y = m(x)
    print(f"零初始化恒等误差: {(y - x).abs().max().item():.2e}")
    assert torch.allclose(y, x, atol=1e-5), "门控零初始化应输出恒等！"

    # 2) 学习通路：gate、tau、空间门控、refine 全部有梯度
    m.gate.data.fill_(0.5)
    y2 = m(x)
    y2.pow(2).mean().backward()
    gs = {
        "gate": m.gate.grad.abs().sum().item(),
        "tau": m.shrink.raw.grad.abs().sum().item(),
        "spatial_conv": m.spatial.conv.weight.grad.abs().sum().item(),
        "refine": m.refine[0].weight.grad.abs().sum().item(),
    }
    print(f"梯度量级: { {k: round(v, 5) for k, v in gs.items()} }")
    assert all(v > 0 for v in gs.values()), "存在无梯度的部件！"

    # 3) 空间门控确实逐位置变化（不是常数图）
    with torch.no_grad():
        ll, lh, hl, hh = m.dwt(x)
        rlh, rhl, rhh = m.refine(torch.cat([lh, hl, hh], dim=1)).chunk(3, dim=1)
        g = m.spatial(rlh, rhl, rhh)
    print(f"空间门控: 均值 {g.mean().item():.3f}, 标准差 {g.std().item():.4f}, 范围 [{g.min().item():.3f}, {g.max().item():.3f}]")
    assert g.std().item() > 1e-3, "空间门控退化为常数图！"

    # 4) 软阈值行为：τ 调大后弱高频被抑制、强高频保留
    m.shrink.raw.data.fill_(2.0)  # τ = softplus(2) ≈ 2.13
    d_weak = torch.randn(2, c, 40, 40) * 0.1
    d_strong = torch.randn(2, c, 40, 40) * 2.0
    print(f"τ≈2.13：弱噪声保留率 {(m.shrink(d_weak).abs().mean() / (d_weak.abs().mean() + 1e-8)).item():.3f}, "
          f"强结构保留率 {(m.shrink(d_strong).abs().mean() / (d_strong.abs().mean() + 1e-8)).item():.3f}")
    assert m.shrink(d_weak).abs().mean() < m.shrink(d_strong).abs().mean() * 0.5

    # 5) 各开关组合可正常前向（消融变体不报错）
    for kw in (dict(shrink=False), dict(spatial=False), dict(use_hh=False)):
        m2 = WSDM(c, c, **kw)
        assert m2(x).shape == x.shape, f"变体 {kw} 输出形状错误"

    # 6) 参数量
    n = sum(p.numel() for p in m.parameters())
    print(f"WSDM(c={c}) 参数量: {n} ({n/1e3:.2f} K)")
    print("✅ 所有测试通过！")
