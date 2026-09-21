# ltdte.py — LDTE：Learnable Directional Texture Enhancement（可学习方向纹理增强模块）
# ==================================================================
# v2 修正（2026-09-10，基于实验20 + 消融①② 的归因数据）：
#   实验20（全开）0.733/0.382：crazing 破 baseline（0.210）但 scratches −2.9；
#   消融①（无位置门控 h）0.729/0.378：crazing 崩（0.182）→ h 是 crazing 增益来源；
#   消融②（无通道门控 g）0.741/0.380：scratches 恢复（0.422）、rolled 追平 E（0.228）、
#        inclusion 改善（0.414）→ **g 是伤 scratches/rolled/inclusion 的主犯**；
#   analyze_ltdte.py 诊断：训练后 λ 仅 2.27~2.98（std 0.12）→ 8 方向核全是 ~2.6px
#        同一窄尺度，"多尺度"自由度未生效 → 对 patches/pitted（无 g 版仅 0.566/0.440）
#        等中尺度类无匹配滤波器、只吃扰动 → **"偏科"的结构性根源**。
#   v2 改动（两处，均有数据支撑）：
#     ① 默认配置改为"无 g"（yaml 传 use_global=False）→ 保住 scratches/rolled/inclusion；
#     ② Gabor 波长 λ 改为**多尺度初始化**（按方向组内通道交替 2.2px / 5.0px，
#        覆盖细网纹 crazing 与中尺度 pitted/patches），不再依赖训练去"学会分化"。
#   两尺度参数可经构造参数 ms_lambdas 调整（默认 (2.2, 5.0)）。
#   ⚠️ 注意：v1 初始化（单一 λ≈2.61）的源码备份为素材文件夹 ltdte_v1文件代码_backup.txt；
#      用旧 yaml 重跑将得到 v2 行为（本项目历史结果均来自已保存的权重，不受影响）。
# ------------------------------------------------------------------
# 定位（针对 yolo26s DB+FEM 基座 E = 0.743/0.388 之后、第三改进点的研发）：
#   进度表实验 15 证明：E 基座上 WDM@P3（无差别全高频注入）= 0.736/0.377 掉点。
#   根因假说（教程 7.9 + 实验数据）：80×80 下钢材背景的磨痕/轧制纹理同为高频，
#   "只放大高频"把背景噪声一并放大（crazing 检出跌、scratches 跌），而
#   WSDM 的每通道软阈值又削掉了大尺度纹理类的强高频（rolled mAP50 崩 9.3）。
#   → 缺的不是"去噪"或"全频注入"，而是**方向选择性**：NEU-DET 六类里
#     crazing(网状)/scratches(线状)/rolled-in_scale(轧制方向) 占验证实例 49%，
#     缺陷纹理有方向结构，背景磨痕是各向同性弱纹理 —— 二者在方向谱上可分。
#
# 机制（相对 WDM 只换"注入什么"，其余工程经验全部继承）：
#   x ─可学习Gabor方向深度卷积(核前向重建)→ D（每通道一个方向角，带通纹理响应）
#     ├─ 全局门控 g_c = σ(MLP(GAP(D²)))   ← 学"哪些方向通道的纹理可信"（通道×方向选择）
#     ├─ 局部门控 h   = 相对局部纹理能量      ← 学"哪里纹理显著"（位置选择，弱背景≈0）
#     ▼
#   detail = (g_c ⊗ h) ⊙ D
#   out = x + γ ⊙ detail          （γ 通道门控，零初始化 = 恒等起步，同 WDM）
#
# 设计依据（2024-2026 文献 + 自有实验）：
#   1) LGTR-Net / MGM（Computers in Industry 2026，钢带表面缺陷同域）：
#      可学习 Gabor 卷积的参数化配方（σ/λ/长宽比 softplus+下限、核去直流、
#      L1 归一化、每前向重建）——本文件 Gabor 核构造沿用其数值方案；
#      **差异化**：MGM 用 Gabor 响应替换特征（块替换、expansion=2），本模块
#      以"带通细节残差 + 能量双重门控"注入，γ=0 恒等起步、<10K 参数即插即用；
#  2) FreqFusion（TPAMI 2024，论文清单 F2 🔴）：高频分"破坏性(背景/光滑区)"
#     与"有效(边界)"，应逐像素自适应选择而非全局增强 → 局部门控 h 的设计来源；
#  3) DWWA-Net（TNNLS 2024，F7 🟡）："去噪"与"增强"角色解耦 → 通道门控(选方向)
#     与位置门控(选地点)分开实现、可独立开关做消融；
#  4) 相对 FEM 的 TextureEnhance（Sobel 初始化组卷积，训练后方向结构漂移不可解释）：
#     LDTE 全程保持显式方向参数（θ/波长/尺度物理可解释，可画核可视化）；
#     位置互补：FEM 在 P5(20×20) 救细纹是亚像素，LDTE 放 P3(80×80) 尺度盲区。
#  5) 消融开关（论文用）：use_spatial=False（关位置门）、use_global=False（关通道门）。
#
# 用法（yaml，ultralytics 自动传入 c1；数字按缩放前写，与 FEM/WDM 行一致）：
#   [-1, 1, LDTE, [256]]      # P3 头输出后（yolo26s 实际宽度 128）
# 参数量（c=128 实测）：≈ 6K 量级（核参数仅 3c+8 标量 + 8K 门控 + 128 输出门）
# 只依赖 torch，不需要 import ultralytics。
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LDTE"]


class _LearnableGaborDWConv(nn.Module):
    """
    可学习 Gabor 方向深度卷积（无固定核存储，每前向按参数重建核）。

    方向分配：c 个通道轮流分配到 num_dirs 个方向基
        θ_i = π·(i mod num_dirs)/num_dirs + Δθ_{i mod num_dirs}
    其中 Δθ 为可学习小偏移（初始 0）→ 覆盖 0~π 全方向谱后微调对齐真实缺陷方向；
    σ（尺度）/λ（波长）/γ（长宽比）每通道独立可学习，softplus+下限保证正值
    （数值配方沿用 LGTR-Net/MGM，Computers in Industry 2026）。

    v2 多尺度初始化：λ 按"方向组内通道"交替取 ms_lambdas 中的值（如 2.2/5.0px），
    使滤波器组起点即具备细/中双尺度覆盖（v1 单一 λ≈2.6 经诊断确认未分化）。
    """

    def __init__(self, c, kernel_size=7, num_dirs=8, ms_lambdas=(2.2, 5.0)):
        super().__init__()
        self.c = c
        self.k = kernel_size
        self.num_dirs = num_dirs

        # 方向基（buffer）：θ 基础值按通道轮流铺满 [0, π)
        base = torch.tensor(
            [(math.pi * (i % num_dirs)) / num_dirs for i in range(c)],
            dtype=torch.float32,
        )
        self.register_buffer("theta_base", base)
        # Δθ：每个方向基共享一个可学习偏移，初始 0（不破坏均匀覆盖）
        self.dtheta = nn.Parameter(torch.zeros(num_dirs))

        # Gabor 物理参数（每通道独立），对数参数化 + softplus 保证正值
        self.log_sigma = nn.Parameter(torch.zeros(c))  # σ 初始 softplus(0)+0.5 ≈ 1.19
        # v2 多尺度 λ 初始化：λ_target = softplus(log_lambda)+1 → log_lambda = log(expm1(λ-1))
        lam_list = list(ms_lambdas) if ms_lambdas else [2.61]
        scale_ids = (torch.arange(c) // num_dirs) % len(lam_list)
        lam_target = torch.tensor(lam_list, dtype=torch.float32)[scale_ids]
        self.log_lambda = nn.Parameter(torch.log(torch.expm1(lam_target - 1.0)))
        self.log_ratio = nn.Parameter(torch.full((c,), math.log(0.5)))  # 长宽比 ≈ 0.61

    def _build_kernels(self, device, dtype):
        radius = self.k // 2
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        y, x = torch.meshgrid(coords, coords, indexing="ij")  # [k,k]

        theta = (
            self.theta_base.to(device=device, dtype=dtype)
            + self.dtheta.to(device=device, dtype=dtype)[
                torch.arange(self.c, device=device) % self.num_dirs
            ]
        )[:, None, None]  # [c,1,1]

        # 坐标旋转到核主方向
        x_t = x * theta.cos() + y * theta.sin()
        y_t = -x * theta.sin() + y * theta.cos()

        sigma = (F.softplus(self.log_sigma) + 0.5).to(dtype=dtype)[:, None, None]
        wavelength = (F.softplus(self.log_lambda) + 1.0).to(dtype=dtype)[:, None, None]
        ratio = (F.softplus(self.log_ratio) + 0.1).to(dtype=dtype)[:, None, None]

        # 高斯包络（σ=尺度、ratio=长宽比）× 余弦载波（λ=波长/频率）
        gaussian = torch.exp(-(x_t.square() + ratio.square() * y_t.square()) / (2 * sigma.square()))
        wave = torch.cos(2 * math.pi * x_t / wavelength)
        kernel = gaussian * wave

        # 去直流（避免退化为平滑核）→ L1 归一化（数值稳定）
        kernel = kernel - kernel.mean(dim=(-2, -1), keepdim=True)
        kernel = kernel / kernel.abs().sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return kernel[:, None]  # [c,1,k,k]

    def forward(self, x):
        kernels = self._build_kernels(x.device, x.dtype)
        return F.conv2d(x, kernels, stride=1, padding=self.k // 2, groups=x.shape[1])


class LDTE(nn.Module):
    """
    Learnable Directional Texture Enhancement（可学习方向纹理增强模块）

    输入 [B,c,H,W] → 输出 [B,c,H,W]（分辨率不变，恒等起步）

    参数:
        c1: 输入通道数（ultralytics 自动传入；内部宽度一律以 c1 为准）
        c2: 兼容占位（yaml 第二参数），可不用
        kernel_size: Gabor 核大小，默认 7（细纹 1~3px 在 80×80 上需要足够核内上下文）
        num_dirs: 方向基数，默认 8（0~π 均匀 8 等分）
        reduction: 通道门控 MLP 压缩率，默认 8
        use_global: 全局通道×方向门控开关（默认 True，消融用）
        use_spatial: 局部位置门控开关（默认 True，消融用）
    """

    def __init__(self, c1, c2=None, kernel_size=7, num_dirs=8, reduction=8,
                 use_global=True, use_spatial=True, ms_lambdas=(2.2, 5.0)):
        super().__init__()
        c = c1
        self.use_global = use_global
        self.use_spatial = use_spatial

        # ① 方向纹理分解（Gabor 深度卷积，v2 多尺度 λ 初始化）
        self.gabor = _LearnableGaborDWConv(c, kernel_size=kernel_size, num_dirs=num_dirs,
                                           ms_lambdas=ms_lambdas)

        # ② 全局通道×方向门控：纹理能量大的方向通道被放大（角色=方向选择）
        hidden = max(8, c // reduction)
        self.dir_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=True),
            nn.Sigmoid(),
        )

        # ③ 局部位置门控：5×5 均值池化平滑 |D| 得局部纹理能量（角色=位置选择）
        self.local_pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

        # 输出通道门控：零初始化 → 初始输出 = 恒等映射（WDM 成功经验）
        self.gate = nn.Parameter(torch.zeros(c, 1, 1))

    def forward(self, x):
        d = self.gabor(x)  # [B,c,H,W] 方向带通纹理响应（近似零均值）

        # 全局通道×方向选择：GAP(D²) → MLP → σ
        if self.use_global:
            g = self.dir_gate(d.pow(2))  # [B,c,1,1]
        else:
            g = None

        # 局部位置选择：|D| 跨通道均值 → 5×5 平滑 → 相对能量（全局均值归一，弱区<1 抑制）
        if self.use_spatial:
            e = d.abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
            e = self.local_pool(e)
            h = e / (e.mean(dim=(2, 3), keepdim=True) + 1e-5)
            h = h.clamp_max(4.0)  # 防止个别强响应主导
        else:
            h = None

        detail = d if g is None else g * d
        detail = detail if h is None else h * detail
        return x + self.gate * detail


if __name__ == "__main__":
    torch.manual_seed(0)
    c = 128
    x = torch.randn(2, c, 80, 80)

    # 1) 恒等起步：γ=0 → 输出 == 输入
    m = LDTE(c, c)
    y = m(x)
    print(f"零初始化恒等误差: {(y - x).abs().max().item():.2e}")
    assert torch.allclose(y, x, atol=1e-5), "门控零初始化应输出恒等！"

    # 2) 学习通路：gate、Gabor 参数、门控 MLP 全部有梯度
    m.gate.data.fill_(0.5)
    y2 = m(x)
    y2.pow(2).mean().backward()
    grads = {
        "gate": m.gate.grad.abs().sum().item(),
        "dtheta": m.gabor.dtheta.grad.abs().sum().item(),
        "log_sigma": m.gabor.log_sigma.grad.abs().sum().item(),
        "log_lambda": m.gabor.log_lambda.grad.abs().sum().item(),
        "log_ratio": m.gabor.log_ratio.grad.abs().sum().item(),
        "dir_gate": m.dir_gate[1].weight.grad.abs().sum().item(),
    }
    print(f"梯度量级: { {k: round(v, 5) for k, v in grads.items()} }")
    assert all(v > 0 for v in grads.values()), "存在无梯度的部件！"

    # 3) 方向覆盖检查：θ_base 应均匀铺满 0~π（8 等分：0..7π/8）
    theta = m.gabor.theta_base.numpy()
    uniq = sorted(set(round(float(t), 4) for t in theta))
    print(f"方向基个数: {len(uniq)}, 值域: [{uniq[0]:.3f}, {uniq[-1]:.3f}]")
    assert len(uniq) == 8, "方向基应为 8 个均匀值"
    # 相邻方向间隔应≈π/8（允许微小误差）
    gaps = [b - a for a, b in zip(uniq, uniq[1:])]
    assert all(abs(g - math.pi / 8) < 1e-3 for g in gaps), f"方向间隔不均匀: {gaps}"

    # 3b) v2 多尺度 λ 初始化检查：应出现两个尺度簇（≈2.2 与 ≈5.0）
    lam = (F.softplus(m.gabor.log_lambda) + 1.0).detach().numpy()
    lam_uniq = sorted(set(round(float(v), 2) for v in lam))
    print(f"λ 初始值簇: {lam_uniq}  (期望 [2.2, 5.0])")
    assert len(lam_uniq) == 2, f"λ 应为双尺度初始化，实际 {lam_uniq}"
    assert abs(lam_uniq[0] - 2.2) < 0.05 and abs(lam_uniq[1] - 5.0) < 0.05, "λ 尺度值不符"

    # 4) 方向选择性检查：沿 y 振荡的"水平条纹"（边界沿 x、梯度沿 y）
    #    应由 θ=π/2 的核（沿 y 振荡）响应，θ=0 核（沿 x 振荡）应弱响应
    with torch.no_grad():
        yy = torch.arange(80, dtype=torch.float32).view(1, 1, 80, 1) / 80.0
        stripe = (torch.sin(2 * math.pi * 6 * yy) * 8.0).repeat(2, c, 1, 80)  # [B,c,80,80]
        resp = m.gabor(stripe)  # [B,c,80,80]
        # θ=0 的通道（i%8==0）vs θ=π/2 的通道（i%8==4）
        dir0 = resp[:, 0::8].pow(2).mean().item()
        dir90 = resp[:, 4::8].pow(2).mean().item()
        print(f"水平条纹响应: θ=0 通道能量={dir0:.4f}, θ=π/2 通道能量={dir90:.4f}")
        assert dir90 > dir0 * 1.5, "方向选择性失效：θ=π/2 核未显著响应水平条纹！"

    # 5) 消融变体可正常前向
    for kw in (dict(use_global=False), dict(use_spatial=False), dict(use_global=False, use_spatial=False)):
        m2 = LDTE(c, c, **kw)
        assert m2(x).shape == x.shape, f"变体 {kw} 输出形状错误"

    # 6) 参数量
    n = sum(p.numel() for p in m.parameters())
    print(f"LDTE(c={c}) 参数量: {n} ({n/1e3:.2f} K)")
    print("✅ 所有测试通过！")
