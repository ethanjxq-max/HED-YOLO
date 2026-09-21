# asff.py — ASFF：检测头输入的自适应空间特征融合（Adaptively Spatial Feature Fusion）
# ==================================================================================
# 出处：ASFF, "Learning Spatial Fusion for Single-Shot Object Detection" (arXiv:1911.09516)；
#       本项目素材库 Larry同学整理的模块V3.md 第 1463-1490 行（含原始代码）；
#       同域先例：SSA-YOLO（IEEE TIM 2024，热轧带钢缺陷检测）用 ASFF 替换检测头输入融合。
#
# 为什么用它（针对本项目实测短板，不是随便加模块）：
#   1) 本项目最强的一次结构改动是"融合点"上的（P5→C2PSA +1.1、宽 P4 +2.6、E 基座 FEM@融合后 +1.7），
#      说明在 yolo26 这个架构里，**融合点**是杠杆最高的位置；
#   2) 现有颈部是"单向"的：P3 检测头只看 P3 的 concat 结果，P5 的全局语义到不了 P3。
#      实测短板里 crazing 漏检（R 最低）是细纹理缺上下文，rolled-in_scale 误检（P 最低）是
#      低对比大区域与背景纹理混淆 —— 两者都是"单尺度视角"造成的典型错误；
#   3) ASFF 给**每个检测尺度**都做一次跨尺度自适应加权（空间位置上逐像素 softmax 权重），
#      低对比/尺度不确定的区域可以自动向"更有判别力的那一层"借信息，而无需新增信息通路
#      （输入仍是颈部已有的 P3/P4/P5 三个特征，没有新的自顶向下/自底向上路径）；
#   4) 参数/计算可控：三层合计约 +0.46M 参数（s 尺度），融合只在 1×1 卷积后的同分辨率特征上做。
#
# ⚠ 与历史失败族的区别：不是新的注意力机制（不产生新的注意力图去乘特征），
#    而是"多尺度证据的凸组合"（softmax 权重和为 1，输出是有界加权平均，不存在某层被放大）。
#
# 用法（yaml，插在 Detect 之前；f 是三个尺度特征的层号，args = [通道数, 目标尺度 0/1/2]）：
#   - [[16, 19, 22], 1, ASFF, [256, 0]]   # P3 级
#   - [[16, 19, 22], 1, ASFF, [512, 1]]   # P4 级
#   - [[16, 19, 22], 1, ASFF, [1024, 2]]  # P5 级
#   - [[23, 24, 25], 1, Detect, [nc]]
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ASFF"]


class ASFF(nn.Module):
    """把三个尺度的特征重采样到目标尺度，按逐像素 softmax 权重自适应融合。

    参数:
        c1_list: [c_p3, c_p4, c_p5] 三个输入通道数（由 tasks.py 自动传入）
        c2: 输出通道数（= 该尺度检测头原输入通道）
        level: 目标尺度，0=P3/8（最高分辨率）、1=P4/16、2=P5/32
        init_bias: 权重卷积的偏置初值（目标尺度给大值 → 起步以本尺度为主，训练中再学跨尺度）
    """

    def __init__(self, c1_list, c2, level=0, init_bias=4.0):
        super().__init__()
        assert level in (0, 1, 2), "level 只能是 0/1/2"
        self.level = int(level)
        self.c2 = c2
        self.compress = nn.ModuleList(nn.Conv2d(c, c2, 1, 1, 0, bias=False) for c in c1_list)
        self.bn = nn.BatchNorm2d(c2 * 3)
        self.wconv = nn.Conv2d(c2 * 3, 3, 1, 1, 0)
        with torch.no_grad():  # 起步：目标尺度占主导（softmax 权重 ≈ 0.96），其余两层可学
            self.wconv.weight.mul_(0.01)
            b = torch.zeros(3)
            b[self.level] = init_bias
            self.wconv.bias.copy_(b)

    def forward(self, x):
        """x = [P3 特征, P4 特征, P5 特征]（ultralytics 多输入模块的固定传参形式）"""
        x3, x4, x5 = x
        feats = [x3, x4, x5]
        tgt = feats[self.level]
        h, w = tgt.shape[-2:]
        ys = []
        for i, f in enumerate(feats):
            if f.shape[-2] != h or f.shape[-1] != w:
                f = F.adaptive_avg_pool2d(f, (h, w)) if f.shape[-2] > h else \
                    F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)
            ys.append(self.compress[i](f))
        cat = self.bn(torch.cat(ys, dim=1))
        wgt = self.wconv(cat).softmax(dim=1)  # [B,3,H,W]，和为 1
        return ys[0] * wgt[:, 0:1] + ys[1] * wgt[:, 1:2] + ys[2] * wgt[:, 2:3]

    def extra_repr(self):
        return f"level={self.level}, c2={self.c2}"


if __name__ == "__main__":
    torch.manual_seed(0)
    chs = [128, 256, 512]
    levels = [(80, 80), (40, 40), (20, 20)]
    outs = []
    for lv in range(3):
        xs = [torch.randn(2, chs[i], levels[i][0], levels[i][1]) for i in range(3)]
        m = ASFF(chs, chs[lv], level=lv)
        y = m(xs)
        p = sum(p.numel() for p in m.parameters())
        print(f"ASFF(level={lv}): 输入 {[tuple(x.shape) for x in xs]} → 输出 {tuple(y.shape)}，参数 {p/1e3:.1f}K")
        assert y.shape == (2, chs[lv], levels[lv][0], levels[lv][1]), "ASFF 输出形状错误"
        outs.append(y)
    # 梯度：三条压缩支路 + 权重卷积都要有梯度
    lv = 1
    xs = [torch.randn(1, chs[i], levels[i][0], levels[i][1]) for i in range(3)]
    m = ASFF(chs, chs[lv], level=lv)
    m(xs).pow(2).mean().backward()
    g = {"compress0": m.compress[0].weight.grad.abs().sum().item(),
         "compress1": m.compress[1].weight.grad.abs().sum().item(),
         "compress2": m.compress[2].weight.grad.abs().sum().item(),
         "wconv": m.wconv.weight.grad.abs().sum().item()}
    print("梯度量级:", {k: round(v, 5) for k, v in g.items()})
    assert all(v > 0 for v in g.values()), "存在无梯度支路！"
    # 起步行为：目标尺度权重应占主导（≈0.95+）
    with torch.no_grad():
        xs = [torch.randn(1, chs[i], levels[i][0], levels[i][1]) for i in range(3)]
        m = ASFF(chs, chs[lv], level=lv)
        ys = []
        tgt = xs[lv]
        for i, f in enumerate(xs):
            if f.shape[-2] != tgt.shape[-2]:
                f = F.adaptive_avg_pool2d(f, tgt.shape[-2:]) if f.shape[-2] > tgt.shape[-2] else \
                    F.interpolate(f, size=tgt.shape[-2:], mode="bilinear", align_corners=False)
            ys.append(m.compress[i](f))
        wgt = m.wconv(m.bn(torch.cat(ys, 1))).softmax(1)
        print("起步时各尺度平均权重:", [round(wgt[:, i].mean().item(), 3) for i in range(3)])
        assert wgt[:, lv].mean().item() > 0.9, "起步权重未偏向目标尺度"
    print("✅ asff.py 全部测试通过！")
