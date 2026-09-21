# dadc.py — DADC：Direction-Aware Deformable Convolution（方向感知可变形卷积）
#                  + Bottleneck_DB_DADC / C3k2_DB_DADC（可变形异构块，改进点③）
# ==================================================================
# 研发背景（2026-09-12，基于 15 组第三点实验的**类级失败归因**）：
#   事实：在 E 基座（0.743/0.388）上的 15 次改动，all 指标 12/12 全部下降，
#         且伤害高度集中——scratches mAP50-95 12/12 下降（均值 −3.2 点）、
#         pitted 11/12 下降（均值 −2.0），而 inclusion(+0.6)/patches(+0.5) 反而微涨。
#   归因：E 的全部优势来自"细/小目标类"（vs baseline：scratches +9.5、pitted +1.8），
#         而所有失败的改动都有一个共同点——**在特征金字塔里"新增了一条独立信息通路"**
#         （新模块插入 / 新增拼接源 / 替换采样算子）。新通路与原通路竞争，细目标类的
#         微弱响应被稀释（类间跷跷板）。
#
#   设计决策（与本项目 15 组负结果一一对应）：
#     ① 不再新增任何通路 → 只**就地把现有卷积核换成形状自适应核**（计算图、通道数、
#        分辨率、层数全部不变，零新增拼接源）；
#     ② 起步必须与基座**严格等价** → 偏移生成器与调制标量全部零初始化
#        （offset=0 → 采样网格=标准网格；数值误差 <5e-6），训练只可能"用数据换增益"，
#        不存在"开局伤特征"；
#     ③ 针对细目标类的结构性弱点 → 偏移场**方向解耦**：x 方向偏移由水平条形卷积
#        (1×k) 生成、y 方向偏移由垂直条形卷积 (k×1) 生成。标准 DCN（用单个 3×3 卷积
#        同时生成 dx/dy）的偏移场是各向同性的，对**细长/曲折结构**（龟裂裂纹网、
#        划痕、氧化铁皮压入）不是最优参数化；方向解耦让采样网格可以沿缺陷走向
#        拉长/贴合，同时条形卷积的参数只有标准 3×3 生成器的 1/3。
#
#   文献依据：
#     1) DCNv4 (arXiv 2401.06197，论文清单 K3)：动态稀疏采样算子的事实标准；
#     2) InternImage (CVPR 2023，清单 M8)：可变形卷积作为主干核心算子；
#     3) DCAM-Net (IEEE TIM 2023，清单 D2 🔴)：**同为 NEU-DET 数据集**，
#        CLAHE + 可变形卷积 + 注意力 → 82.6 mAP50，是"可变形卷积对钢材表面缺陷有效"
#        的同数据集期刊级证据；
#     4) DSConv / DSCNet (ICCV 2023，清单 K6 / V3 模块44)：蛇形偏移专为"细长曲折的
#        管状结构"设计，与本模块"方向解耦偏移"动机一致（本文不做拓扑约束，取更稳的
#        方向解耦 + 有界化）；
#     5) InceptionNeXt / IDC (CVPR 2024，V3 模块67)：1×k 与 k×1 条形卷积对细长结构
#        的有效性（其论文与本文在"条纹核"上的取舍一致）。
#
#   与已失败方向的区别：
#     - 不是"新注意力"（CAA/P2AT/PAM 全灭）：本模块没有任何注意力/门控，只有采样位置；
#     - 不是"频域高频注入"（WDM/WSDM 全灭）：不引入任何外部滤波器响应；
#     - 不是"新拼接源/新分支"（MGF/SBE/HRSB 全灭）：结构一行不改，只换卷积核；
#     - 不是"替换采样算子"（AConv/EA-Up 全灭）：下采样/上采样层一个字不动。
#
#   用法（yaml，与官方 C3k2 完全同签名，替换 C3k2_DB 即可）：
#     - [-1, 2, C3k2_DB_DADC, [512, False, 0.25]]        # backbone P3 阶段
#     - [-1, 2, C3k2_DB_DADC, [512, True]]               # backbone P4 阶段
#     - [-1, 2, C3k2_DB_DADC, [1024, True]]              # backbone P5 阶段
#     - [-1, 1, C3k2_DB_DADC, [1024, True, 0.5, True]]   # head P5（attn=True）
#   消融开关（yaml 透传）：
#     offset_mode='directional'（本文设计）/ 'snake'（链式累积偏移，可弯曲贴合细长曲折结构，
#                 对应 DSConv/DSCNet ICCV 2023 的拓扑约束思想）/ 'standard'（标准 DCN 基线）
#     dcn_where='both'（默认，cv1+cv2 都换）/ 'cv2'（只换第二个 3×3，更省算力）
#     use_mask=True（DCNv2 式调制标量，2σ(·) 参数化 → 初始化恒为 1，仍严格等价起步）
#   参数量（相对 C3k2_DB）：每个 DADC 卷积仅多 54·c1 个偏移参数（c=256 时约 14K）。
# ==================================================================
import math

import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d

from ultralytics.nn.modules.block import Bottleneck, PSABlock
from ultralytics.nn.modules.conv import Conv

from .db import Bottleneck_DB, C3k_DB, C3k2_DB

__all__ = ["DeformConv2d", "Bottleneck_DADC", "Bottleneck_DB_DADC", "C3k2_DB_DADC"]


class DeformConv2d(nn.Module):
    """方向感知可变形卷积（Conv + BN + SiLU 的直插替代）。

    参数:
        c1, c2 : 输入/输出通道
        k      : 卷积核尺寸（默认 3）
        s      : 步长（默认 1；本工程只在 stride=1 的 Bottleneck 内部使用）
        p      : padding（默认 k//2）
        g      : 分组数
        act    : 是否带激活（对齐 ultralytics Conv 的行为）
        offset_mode : 'directional'（本文：水平带生成 dx、垂直带生成 dy）
                      'snake'      （链式约束：偏移沿条形方向逐点累积 → 采样网格可弯曲贴合
                                    细长曲折结构，对应 DSConv/DSCNet ICCV 2023 的思想）
                      'standard'   （消融对照：标准 DCN 的 3×3 偏移生成器）
        dcn_where   : 'both'/'cv1'/'cv2'（由 Bottleneck_DADC 使用，本类不感知）
        use_mask    : DCNv2 式调制标量（0~2，初始化恒为 1）
        max_offset  : 偏移幅度上限（tanh 有界化，防止小样本训练发散）
    """

    def __init__(
        self,
        c1,
        c2,
        k=3,
        s=1,
        p=None,
        g=1,
        act=True,
        offset_mode="directional",
        use_mask=False,
        max_offset=2.0,
    ):
        super().__init__()
        self.k = k
        self.stride = s
        self.padding = k // 2 if p is None else p
        self.groups = g
        self.offset_mode = offset_mode
        self.max_offset = max_offset

        self.weight = nn.Parameter(torch.empty(c2, c1 // g, k, k))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

        n_off = 2 * k * k  # 每个核位置一个 (dy, dx) 对
        if offset_mode in ("directional", "snake"):
            # 本文设计：x 偏移 ← 水平条形卷积(1×k)；y 偏移 ← 垂直条形卷积(k×1)
            self.off_h = nn.Conv2d(c1, k * k, (1, k), stride=s, padding=(0, k // 2))
            self.off_v = nn.Conv2d(c1, k * k, (k, 1), stride=s, padding=(k // 2, 0))
            for m in (self.off_h, self.off_v):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        elif offset_mode == "standard":
            self.off = nn.Conv2d(c1, n_off, 3, stride=s, padding=1)
            nn.init.zeros_(self.off.weight)
            nn.init.zeros_(self.off.bias)
        else:
            raise ValueError(f"offset_mode 必须是 'directional'/'snake'/'standard'，收到 {offset_mode}")

        self.use_mask = use_mask
        if use_mask:
            self.mask_conv = nn.Conv2d(c1, k * k, 3, stride=s, padding=1)
            nn.init.zeros_(self.mask_conv.weight)
            nn.init.zeros_(self.mask_conv.bias)  # 2σ(0)=1 → 起步调制恒为 1

    def _offset(self, x, b, h, w):
        """生成 deform_conv2d 所需的偏移张量 [B, 2*k*k, h, w]（通道顺序 y_0,x_0,y_1,x_1,...）。"""
        k = self.k
        if self.offset_mode == "standard":
            return torch.tanh(self.off(x)) * self.max_offset

        dx = self.off_h(x)  # [B, k*k, h, w]
        dy = self.off_v(x)
        if self.offset_mode == "directional":
            off = torch.stack((dy, dx), dim=2).reshape(b, 2 * k * k, h, w)  # (y_0,x_0,y_1,x_1,...)
            return torch.tanh(off) * self.max_offset

        # snake：偏移沿条形方向逐点累积（有界），采样点构成一条可弯曲的链
        step = self.max_offset / k
        dx = torch.cumsum(torch.tanh(dx.view(b, k, k, h, w)) * step, dim=2)  # 沿列方向累积
        dy = torch.cumsum(torch.tanh(dy.view(b, k, k, h, w)) * step, dim=1)  # 沿行方向累积
        off = torch.stack((dy, dx), dim=-1)  # [B, k, k, h, w, 2]
        return off.permute(0, 1, 2, 5, 3, 4).reshape(b, 2 * k * k, h, w)

    def forward(self, x):
        b, _, h, w = x.shape
        off = self._offset(x, b, h, w)  # 零初始化 → 全 0 → 采样网格 = 标准网格

        mask = 2.0 * torch.sigmoid(self.mask_conv(x)) if self.use_mask else None
        y = deform_conv2d(x, off, self.weight, None, self.stride, self.padding, 1, mask)
        return self.act(self.bn(y))


class Bottleneck_DADC(Bottleneck):
    """Bottleneck 的可变形版：把 3×3 固定网格卷积换成方向感知可变形卷积。

    签名与官方 Bottleneck 一致，额外接 dcn_where / offset_mode / use_mask / max_offset。
    dcn_where='cv2' 时只替换第二个 3×3（算力最省，消融用）。
    """

    def __init__(
        self,
        c1,
        c2,
        shortcut=True,
        g=1,
        k=(3, 3),
        e=0.5,
        dcn_where="both",
        offset_mode="directional",
        use_mask=False,
        max_offset=2.0,
    ):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        kw = dict(offset_mode=offset_mode, use_mask=use_mask, max_offset=max_offset)
        cv1 = Conv(c1, c_, k[0], 1) if dcn_where == "cv2" else DeformConv2d(c1, c_, k[0], 1, **kw)
        cv2 = Conv(c_, c2, k[1], 1, g=g) if dcn_where == "cv1" else DeformConv2d(c_, c2, k[1], 1, g=g, **kw)
        self.cv1, self.cv2 = cv1, cv2
        self.add = shortcut and c1 == c2


class Bottleneck_DB_DADC(Bottleneck_DB):
    """① 号改进点（异构双分支）的可变形升级版：**只换分支A 的卷积核**，其余逐位不变。

    - 分支A：Bottleneck_DADC（方向感知可变形卷积，零初始化 → 起步严格等价于原 Bottleneck）
    - 分支B：LSKA 大核注意力（保持不动，继续负责大感受野上下文）
    - 融合：concat → 1×1（保持不动）
    因此 yaml 里把 C3k2_DB 换成 C3k2_DB_DADC，训练起点与 E 基座**逐位等价**。
    """

    def __init__(
        self,
        c1,
        c2,
        shortcut=True,
        g=1,
        k=(3, 3),
        e=0.5,
        k_size=7,
        dcn_where="both",
        offset_mode="directional",
        use_mask=False,
        max_offset=2.0,
    ):
        super().__init__(c1, c2, shortcut, g, k, e, k_size)
        self.branch_a = Bottleneck_DADC(
            c1,
            c2,
            shortcut=False,
            g=g,
            k=k,
            e=e,
            dcn_where=dcn_where,
            offset_mode=offset_mode,
            use_mask=use_mask,
            max_offset=max_offset,
        )


class C3k2_DB_DADC(C3k2_DB):
    """保壳换芯：C3k2_DB 壳不变，壳内 Bottleneck_DB → Bottleneck_DB_DADC。

    签名与 C3k2_DB / 官方 C3k2 完全一致（多 4 个可选开关），yaml 用法不变：
      [-1, 2, C3k2_DB_DADC, [512, False, 0.25]]
      [-1, 2, C3k2_DB_DADC, [512, True]]
      [-1, 1, C3k2_DB_DADC, [1024, True, 0.5, True]]   # attn 分支
    额外开关可写在参数末尾：
      [-1, 2, C3k2_DB_DADC, [512, True, 0.5, False, 7, 'cv2', 'standard', True]]
      （顺序：k_size, dcn_where, offset_mode, use_mask）
    """

    def __init__(
        self,
        c1,
        c2,
        n=1,
        c3k=False,
        e=0.5,
        attn=False,
        g=1,
        shortcut=True,
        k_size=7,
        dcn_where="both",
        offset_mode="directional",
        use_mask=False,
        max_offset=2.0,
    ):
        nn.Module.__init__(self)  # 跳过 C3k2.__init__ 的默认构造，逻辑与 C3k2_DB 一一对应
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.attn = attn
        kw = dict(
            dcn_where=dcn_where,
            offset_mode=offset_mode,
            use_mask=use_mask,
            max_offset=max_offset,
        )

        if attn:
            self.m = nn.ModuleList(
                nn.Sequential(
                    Bottleneck_DB_DADC(self.c, self.c, shortcut, g, k=(3, 3), e=0.5, k_size=k_size, **kw),
                    PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
                )
                for _ in range(n)
            )
        elif c3k:
            self.m = nn.ModuleList(_C3k_DB_DADC(self.c, self.c, 2, shortcut, g, e=0.5, k=3, k_size=k_size, **kw)
                                   for _ in range(n))
        else:
            self.m = nn.ModuleList(
                Bottleneck_DB_DADC(self.c, self.c, shortcut, g, k=(3, 3), e=0.5, k_size=k_size, **kw)
                for _ in range(n)
            )


class _C3k_DB_DADC(C3k_DB):
    """c3k=True 分支用：C3k 壳保留，壳内 Bottleneck_DB → Bottleneck_DB_DADC。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3, k_size=7, **kw):
        super().__init__(c1, c2, n, shortcut, g, e, k=k, k_size=k_size)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck_DB_DADC(c_, c_, shortcut, g, k=(k, k), e=1.0, k_size=k_size, **kw) for _ in range(n))
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    print("=" * 78)
    # 1) 零初始化 → 与标准卷积严格等价（数值误差量级）
    x = torch.randn(2, 32, 16, 16)
    w = torch.randn(24, 32, 3, 3)
    y_ref = torch.nn.functional.conv2d(x, w, padding=1)
    from torchvision.ops import deform_conv2d as _dc

    y_def = _dc(x, torch.zeros(2, 18, 16, 16), w, None, 1, 1, 1)
    print(f"[1] 零偏移 vs 标准卷积 最大误差: {(y_def - y_ref).abs().max().item():.2e}  (应 <1e-5)")

    # 2) 三种配置的前向 + 参数量
    for c in (128, 256):
        for where in ("both", "cv2"):
            m = C3k2_DB_DADC(c, c, n=2, c3k=False, e=0.5, dcn_where=where)
            y = m(torch.randn(1, c, 20, 20))
            n_p = sum(p.numel() for p in m.parameters())
            print(f"[2] C3k2_DB_DADC(c={c}, dcn_where={where}): {tuple(y.shape)}  参数 {n_p/1e3:.1f}K")

    # 3) 与 C3k2_DB 的参数量对比 + 起步等价性（同为 DB 壳，只有分支A 的卷积核不同）
    for c in (128, 256, 512):
        db = C3k2_DB(c, c, n=2, c3k=False, e=0.5)
        dc = C3k2_DB_DADC(c, c, n=2, c3k=False, e=0.5)
        p_db = sum(p.numel() for p in db.parameters())
        p_dc = sum(p.numel() for p in dc.parameters())
        print(f"[3] c={c}: C3k2_DB {p_db/1e3:.1f}K → DADC {p_dc/1e3:.1f}K  ({(p_dc - p_db)/1e3:+.1f}K)")

    # 4) 消融开关：standard 偏移生成器 / mask / cv2
    for kw in (dict(offset_mode="standard"), dict(use_mask=True), dict(dcn_where="cv2")):
        m = C3k2_DB_DADC(256, 256, n=2, c3k=False, e=0.5, **kw)
        y = m(torch.randn(1, 256, 20, 20))
        print(f"[4] 消融 {kw}: {tuple(y.shape)} 参数 {sum(p.numel() for p in m.parameters())/1e3:.1f}K")

    # 5) 梯度连通性（偏移生成器必须有梯度）
    m = C3k2_DB_DADC(64, 64, n=1, c3k=False, e=0.5)
    y = m(torch.randn(1, 64, 20, 20))
    y.pow(2).mean().backward()
    g_off = m.m[0].branch_a.cv1.off_h.weight.grad.abs().sum().item()
    g_w = m.m[0].branch_a.cv1.weight.grad.abs().sum().item()
    print(f"[5] 梯度: 偏移生成器 {g_off:.3e} / 卷积核 {g_w:.3e}  (两者都应 >0)")
    assert g_off > 0 and g_w > 0
    print("✅ 全部自检通过")
