# repblock.py — MRB：多分支重参数化瓶颈块（Multi-branch Reparameterized Bottleneck）
# ==================================================================================
# 设计背景（2026-09-13，针对本项目 20+ 次"模块不涨点"的诊断）：
#   本项目已实测：单 seed 的种子噪声 ≈ ±0.8 mAP50-95（E 基座 seed42=0.388 / seed0=0.380），
#   而过去 20 次模块尝试全部落在基座 ±1.0 以内。诊断结论：
#     (a) 从零训练 + 1440 张训练图 → 网络处于"数据受限"而非"容量受限"；
#     (b) 历史成功的改动（DB 双分支、FEM 多分支、宽 P4）都是"结构性的多分支/多尺度"，
#         历史失败的改动大多是"零初始化小残差支路（安全但学不动）+ 单机制信号注入"；
#     (c) 零初始化支路在 250 epoch 从零训练里没有梯度优势，收敛到近恒等 → 不涨点。
#
#   本模块换一类干预：**训练期多分支、推理期精确折叠**（RepVGG CVPR2021 / DBB NeurIPS2021）。
#   - 训练期：一个 3×3 卷积被替换为多分支并行（每个分支自带 BN），分支数×感受野多样性都增加，
#     等价于"隐式集成 + 更优的优化地形"，这正是数据受限从零训练最缺的东西；
#   - 推理期：所有分支（含 BN）**数学上精确**合并为一个 3×3 卷积 → 参数量/FLOPs 与基线持平
#     （实测还略低：基线 bottleneck 是 3×3(c→c/2)+3×3(c/2→c) 两个卷积，折叠后只剩一个 3×3）。
#
#   分支构成与"本项目已测短板"的对应关系（不是拍脑袋堆分支）：
#     ① k×k 主分支        —— 基础感受野（滚动方向尺度带 rolled-in_scale / patches）
#     ② 1×1 分支          —— 点式/小目标响应（pitted_surface 小凹坑）
#     ③ 1×k 水平条带分支   —— 沿轧制方向的线状缺陷（scratches；本项目 94.3% 目标长宽比>3）
#     ④ k×1 垂直条带分支   —— 垂直向细纹（crazing 网状裂纹的交错走向）
#     ⑤ k×k 平均池化分支   —— 低通/背景纹理抑制（本项目噪声主线的反面：不做高频注入，做低频保留）
#     ⑥ 恒等分支（无外残差时）—— 梯度直通（从零训练小数据）
#     ⚠ 与已失败的 TSAC（条带核）区别：TSAC 是把条带核作为**新增的零初始化残差支路**（学不动）；
#        本模块是把条带核放进**重参数化并行分支**里，训练期与主分支同步被优化、推理期被折叠，
#        不引入任何额外推理开销，也不依赖"支路自己学会开门"。
#
#   文献：RepVGG (arXiv:2101.03697, CVPR 2021)、DBB Diverse Branch Block (arXiv:2103.13425, NeurIPS 2021)；
#         素材库 Larry同学整理的模块V3.md 第 9131-9140 行（PlainUSR 的 RepMBConv：训练复杂的 MBConv
#         结构等价折叠为推理阶段的单一标准卷积）亦为同一技术路线在本项目素材库内的出处。
#
# 用法（yaml，签名与官方 C3k2 完全一致，参数一个字不用改）：
#   [-1, 2, C3k2_MRB, [512, False, 0.25]]      # 主干
#   [-1, 2, C3k2_MRB, [512, True]]             # c3k=True 分支
#   [-1, 1, C3k2_MRB, [1024, True, 0.5, True]] # attn=True 分支（内部 PSABlock 保留）
# 消融开关（写在 yaml 第 5 个参数之后，可选）：
#   [512, True, 0.5, False, 1, True, False, True] → use_cross=False
# 部署（论文里的"零推理开销"实验）：见文件末尾 rep_fuse_model()
import copy

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck, C3k, C3k2, PSABlock

__all__ = ["RepConvDiverse", "Bottleneck_MRB", "C3k_MRB", "C3k2_MRB", "rep_fuse_model"]


class RepConvDiverse(nn.Module):
    """多样多分支卷积（训练）→ 精确折叠为单个 k×k 卷积（推理）。

    分支：k×k、1×1、(1×k)、(k×1)、k×k 平均池化、恒等；每个带 BN 的分支都能被**精确**
    折进一个 k×k 卷积核（1×1 折到核中心、1×k 折到核中间行、k×1 折到核中间列、池化分支
    的核为 1/k² 的常数核、恒等分支折到核中心为 1）。因此 fuse() 之后前向结果逐位等价
    （浮点误差 <1e-5），但计算量只有原来的 1/分支数。

    参数:
        c1, c2: 输入/输出通道（c1 == c2 时恒等分支与池化分支才可用）
        k: 卷积核尺寸（奇数，默认 3）
        g: 分组数（g=1 时 1×k / k×1 分支为常规条带卷积；g=c 时退化为深度可分离条带）
        use_cross/use_pool/use_id: 消融开关
        deploy: 直接构建折叠后的单卷积（用于导出/推理）
    """

    def __init__(self, c1, c2, k=3, g=1, use_cross=True, use_pool=True, use_id=True, deploy=False):
        super().__init__()
        assert k % 2 == 1, "核尺寸必须是奇数（保证中心对齐折叠）"
        self.c1, self.c2, self.k, self.g = c1, c2, k, g
        self.pad = k // 2
        self.identity_ok = c1 == c2
        self.use_cross = bool(use_cross)
        self.use_pool = bool(use_pool and self.identity_ok)
        self.use_id = bool(use_id and self.identity_ok)
        self.deploy = bool(deploy)

        if self.deploy:
            self.fused = nn.Conv2d(c1, c2, k, 1, self.pad, groups=g, bias=True)
            return

        self.br_main = self._conv_bn(c1, c2, k, self.pad, g)  # ① k×k
        self.br_p11 = self._conv_bn(c1, c2, 1, 0, g)  # ② 1×1
        if self.use_cross:
            self.br_h = self._conv_bn(c1, c2, (1, k), (0, self.pad), g)  # ③ 1×k
            self.br_v = self._conv_bn(c1, c2, (k, 1), (self.pad, 0), g)  # ④ k×1
        if self.use_pool:
            self.br_pool = nn.Sequential(nn.AvgPool2d(k, 1, self.pad), nn.BatchNorm2d(c2))  # ⑤
        if self.use_id:
            self.br_id = nn.BatchNorm2d(c2)  # ⑥

    @staticmethod
    def _conv_bn(c1, c2, k, p, g):
        return nn.Sequential(nn.Conv2d(c1, c2, k, 1, p, groups=g, bias=False), nn.BatchNorm2d(c2))

    def _branch_names(self):
        return ["br_main", "br_p11", "br_h", "br_v", "br_pool", "br_id"]

    def forward(self, x):
        if self.deploy:
            return self.fused(x)
        y = self.br_main(x) + self.br_p11(x)
        if self.use_cross:
            y = y + self.br_h(x) + self.br_v(x)
        if self.use_pool:
            y = y + self.br_pool(x)
        if self.use_id:
            y = y + self.br_id(x)
        return y

    @torch.no_grad()
    def _fold(self):
        """把所有分支合并成一个 [c2, c1//g, k, k] 的核与偏置（BN 折进权重）。"""
        k, g = self.k, self.g
        ci = self.c1 // g
        W = torch.zeros(self.c2, ci, k, k, dtype=self.br_main[0].weight.dtype, device=self.br_main[0].weight.device)
        b = torch.zeros(self.c2, dtype=W.dtype, device=W.device)

        def embed(conv_w, bn):
            co, cig, kh, kw = conv_w.shape
            Wt = conv_w.new_zeros(co, cig, k, k)
            i0, j0 = (k - kh) // 2, (k - kw) // 2
            Wt[:, :, i0 : i0 + kh, j0 : j0 + kw] = conv_w
            s = bn.weight / torch.sqrt(bn.running_var + bn.eps)  # BN 缩放
            Wt = Wt * s.view(-1, 1, 1, 1)
            bt = (bn.bias if bn.bias is not None else torch.zeros_like(s)) - bn.running_mean * s
            return Wt, bt

        for name in self._branch_names():
            br = getattr(self, name, None)
            if br is None:
                continue
            if name == "br_pool":  # 常数核 1/k² 再折 BN
                Wp = torch.zeros_like(W)
                idx = torch.arange(self.c2, device=W.device)
                Wp[idx, idx % ci, :, :] = 1.0 / (k * k)
                s = br[1].weight / torch.sqrt(br[1].running_var + br[1].eps)
                W = W + Wp * s.view(-1, 1, 1, 1)
                b = b + (br[1].bias - br[1].running_mean * s)
            elif name == "br_id":  # 恒等 → 核中心 1，再折 BN
                Wi = torch.zeros_like(W)
                idx = torch.arange(self.c2, device=W.device)
                Wi[idx, idx % ci, k // 2, k // 2] = 1.0
                s = br.weight / torch.sqrt(br.running_var + br.eps)
                W = W + Wi * s.view(-1, 1, 1, 1)
                b = b + (br.bias - br.running_mean * s)
            else:
                Wt, bt = embed(br[0].weight, br[1])
                W = W + Wt
                b = b + bt
        return W, b

    @torch.no_grad()
    def fuse(self):
        """原地折叠：删除全部分支，只留一个 3×3 卷积（推理用）。返回 self。"""
        if self.deploy:
            return self
        W, b = self._fold()
        conv = nn.Conv2d(self.c1, self.c2, self.k, 1, self.pad, groups=self.g, bias=True,
                         dtype=W.dtype, device=W.device)
        conv.weight.copy_(W)
        conv.bias.copy_(b)
        for name in self._branch_names():
            if hasattr(self, name):
                delattr(self, name)
        self.fused = conv
        self.deploy = True
        return self

    def extra_repr(self):
        return f"c1={self.c1}, c2={self.c2}, k={self.k}, g={self.g}, cross={self.use_cross}, pool={self.use_pool}, id={self.use_id}, deploy={self.deploy}"


class Bottleneck_MRB(Bottleneck):
    """MRB 版 Bottleneck：整块的 cv1+cv2（3×3 → 3×3，中间无残差）被替换为
    一个 RepConvDiverse（多分支）+ SiLU + 外残差。

    - 训练期：多分支并行（感受野/核形状多样），每个分支独立 BN；
    - 推理期：折叠为单个 3×3 卷积 —— 参数量与 FLOPs ≤ 原 Bottleneck（原为两个卷积）。
    签名与官方 Bottleneck 一致：e 仅为兼容保留（本模块不再做通道压缩）。
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5,
                 use_cross=True, use_pool=True, use_id=True):
        super().__init__(c1, c2, shortcut, g, k, e)
        add = self.add  # c1 == c2 and shortcut
        self.rep = RepConvDiverse(
            c1, c2, k=k[1], g=g,
            use_cross=use_cross, use_pool=use_pool,
            use_id=(use_id and not add),  # 有外残差时不再叠加内部恒等分支（避免双恒等）
        )
        self.act = nn.SiLU()
        del self.cv1, self.cv2  # 由 self.rep 取代

    def forward(self, x):
        y = self.act(self.rep(x))
        return x + y if self.add else y


class C3k_MRB(C3k):
    """c3k=True 层用：C3k 壳保留，壳内 Bottleneck → Bottleneck_MRB。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3,
                 use_cross=True, use_pool=True, use_id=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_MRB(c_, c_, shortcut, g, k=(k, k), e=1.0,
                                                use_cross=use_cross, use_pool=use_pool, use_id=use_id)
                                 for _ in range(n)))


class C3k2_MRB(C3k2):
    """方案 B（保壳换芯）的 MRB 版：C3k2 壳保留，壳内 Bottleneck → Bottleneck_MRB。
    签名与官方 C3k2 完全一致：(c1, c2, n, c3k, e, attn, g, shortcut) + 三个消融开关。
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True,
                 use_cross=True, use_pool=True, use_id=True):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        if attn:
            self.m = nn.ModuleList(
                nn.Sequential(
                    Bottleneck_MRB(self.c, self.c, shortcut, g, k=(3, 3), e=0.5,
                                   use_cross=use_cross, use_pool=use_pool, use_id=use_id),
                    PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
                )
                for _ in range(n)
            )
        elif c3k:
            self.m = nn.ModuleList(
                C3k_MRB(self.c, self.c, 2, shortcut, g, e=0.5, k=3,
                        use_cross=use_cross, use_pool=use_pool, use_id=use_id)
                for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(
                Bottleneck_MRB(self.c, self.c, shortcut, g, k=(3, 3), e=0.5,
                               use_cross=use_cross, use_pool=use_pool, use_id=use_id)
                for _ in range(n)
            )


@torch.no_grad()
def rep_fuse_model(model):
    """把整网所有 RepConvDiverse 折叠为单卷积（推理/导出前调用）。

    返回 (折叠数量, 折叠前参数量, 折叠后参数量)。
    """
    before = sum(p.numel() for p in model.parameters())
    n = 0
    for m in model.modules():
        if isinstance(m, RepConvDiverse) and not m.deploy:
            m.fuse()
            n += 1
    after = sum(p.numel() for p in model.parameters())
    return n, before, after


if __name__ == "__main__":
    torch.manual_seed(0)

    # 1) 形状与折叠等价性
    for c in (64, 128, 256):
        x = torch.randn(2, c, 40, 40)
        m = RepConvDiverse(c, c, k=3, g=1)
        m.eval()
        y = m(x)
        assert y.shape == x.shape, "形状错误"
        p_train = sum(p.numel() for p in m.parameters())
        # 折叠前把 BN 设成非平凡状态（否则 running_mean=0/var=1 的恒等 BN 掩盖折叠错误）
        for name in m._branch_names():
            br = getattr(m, name, None)
            if br is None:
                continue
            bn = br if name == "br_id" else br[1]
            with torch.no_grad():
                bn.running_mean.normal_(0, 0.5)
                bn.running_var.uniform_(0.5, 2.0)
                bn.weight.uniform_(0.5, 1.5)
                bn.bias.normal_(0, 0.3)
        # ① float64 下验证折叠**数学精确**；② float32 下验证工程可用（相对误差）
        m64 = copy.deepcopy(m).double().eval()
        y1 = m64(x.double())
        m64.fuse()
        y2 = m64(x.double())
        err64 = (y1 - y2).abs().max().item()
        m32 = copy.deepcopy(m).eval()
        ya = m32(x)
        m32.fuse()
        yb = m32(x)
        err32 = (ya - yb).abs().max().item() / max(ya.abs().max().item(), 1e-6)
        p_deploy = sum(p.numel() for p in m32.parameters())
        print(f"RepConvDiverse(c={c}): 训练参数 {p_train/1e3:.1f}K → 折叠后 {p_deploy/1e3:.1f}K, "
              f"折叠误差 float64={err64:.2e} / float32 相对={err32:.2e}")
        assert err64 < 1e-9, "折叠在 float64 下不等价（数学错误）！"
        assert err32 < 1e-4, "折叠在 float32 下误差过大！"

    # 2) 所有分支都有梯度（可学性）
    m = RepConvDiverse(64, 64, k=3)
    m(torch.randn(2, 64, 20, 20)).pow(2).mean().backward()
    gsum = {n: sum(p.grad.abs().sum().item() for p in mod.parameters() if p.grad is not None)
            for n, mod in m.named_children()}
    print("各分支梯度量级:", {k: round(v, 4) for k, v in gsum.items()})
    assert all(v > 0 for v in gsum.values()), "存在无梯度分支！"

    # 3) C3k2_MRB 形状/参数量（训练期 vs 折叠后）
    x = torch.randn(1, 256, 40, 40)
    for c3k, attn in ((False, False), (True, False), (False, True)):
        ref = C3k2(256, 256, 2, c3k, 0.25, attn)
        new = C3k2_MRB(256, 256, 2, c3k, 0.25, attn)
        p_ref = sum(p.numel() for p in ref.parameters())
        p_new = sum(p.numel() for p in new.parameters())
        y = new(x)
        assert y.shape == ref(x).shape, "C3k2_MRB 输出形状与 C3k2 不一致"
        n_fused, p_b, p_a = rep_fuse_model(new)
        assert new(x).shape == (1, 256, 40, 40)
        print(f"C3k2_MRB(c3k={c3k}, attn={attn}): 形状✓ 训练参数 {p_new/1e6:.3f}M "
              f"(官方 C3k2 {p_ref/1e6:.3f}M) → 折叠 {n_fused} 处 → {p_a/1e6:.3f}M")
        assert p_a < p_new, "折叠后参数量应下降"

    # 4) 折叠等价性（在真实 C3k2_MRB 上，eval 模式、随机 BN 状态、float64 精确验证）
    new = C3k2_MRB(256, 256, 2, False, 0.25, False).eval()
    for mod in new.modules():
        if isinstance(mod, nn.BatchNorm2d):
            with torch.no_grad():
                mod.running_mean.normal_(0, 0.4)
                mod.running_var.uniform_(0.6, 1.8)
                mod.weight.uniform_(0.6, 1.4)
                mod.bias.normal_(0, 0.2)
    xi = torch.randn(1, 256, 40, 40)
    nd = copy.deepcopy(new).double()
    y1 = nd(xi.double())
    rep_fuse_model(nd)
    y2 = nd(xi.double())
    err = (y1 - y2).abs().max().item()
    print(f"整块折叠误差（float64）: {err:.2e}")
    assert err < 1e-9, "整块折叠不等价！"

    print("✅ repblock.py 全部测试通过！")
