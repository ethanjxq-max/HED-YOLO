# db.py — C3k2-DB：保壳换芯（方案B）—— ultralytics 8.4.137（YOLO26）适配版
# 结构：C3k2 壳保留 → 按官方开关换芯：
#       attn=True ：内部 Bottleneck+PSABlock 序列 → Bottleneck_DB+PSABlock
#       c3k=True  ：内部 n 个 C3k → C3k_DB（C3k 壳内 Bottleneck → Bottleneck_DB）
#       c3k=False ：内部 n 个 Bottleneck → Bottleneck_DB
# 组件出处：Bottleneck/C3k/C3k2/PSABlock（官方 block.py）+ LSKA（ESWA 2024，自带）
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck, C3k, C3k2, PSABlock

__all__ = ["Bottleneck_DB", "C3k_DB", "C3k2_DB"]  # 只导出这三个，避免把 import 的官方类名带进全局命名空间


# ---------------- 组件：LSKA（ESWA 2024，工程内没有，此处自带） ----------------
class LSKA(nn.Module):
    """
    Large Separable Kernel Attention（ESWA 2024）
    大核注意力分解为：水平DWConv(1,k) → 垂直DWConv(k,1) → 1x1通道混合
    """
    def __init__(self, dim, k_size=7):
        super().__init__()
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(1, k_size), stride=1,
                              padding=(0, k_size // 2), groups=dim, bias=False)
        self.dw_v = nn.Conv2d(dim, dim, kernel_size=(k_size, 1), stride=1,
                              padding=(k_size // 2, 0), groups=dim, bias=False)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        attn = self.dw_h(x)
        attn = self.dw_v(attn)
        attn = self.pw(attn)
        return identity * attn


# ---------------- 核心：异构双分支并行块 ----------------
class Bottleneck_DB(nn.Module):
    """
    异构双分支并行块
    分支A：官方 Bottleneck（局部细节，内部无残差，残差统一在外层）
    分支B：1x1 对齐 + LSKA 大核注意力（上下文）
    融合：concat → 1x1
    签名与官方 Bottleneck 完全兼容：(c1, c2, shortcut, g, k, e) + k_size
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, k_size=7):
        super().__init__()
        self.branch_a = Bottleneck(c1, c2, shortcut=False, g=g, k=k, e=e)
        self.cv3 = Conv(c1, c2, 1, 1)
        self.lska = LSKA(c2, k_size=k_size)
        self.fuse = Conv(c2 * 2, c2, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y1 = self.branch_a(x)                   # 分支A：[B, c2, H, W]
        y2 = self.lska(self.cv3(x))             # 分支B：[B, c2, H, W]
        out = self.fuse(torch.cat([y1, y2], dim=1))  # 并行融合
        return x + out if self.add else out


# ---------------- 方案B：保壳换芯 ----------------
class C3k_DB(C3k):
    """c3k=True 层用：C3k 壳保留，壳内 Bottleneck → Bottleneck_DB"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3, k_size=7):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_DB(c_, c_, shortcut, g, k=(k, k), e=1.0, k_size=k_size)
                                 for _ in range(n)))


class C3k2_DB(C3k2):
    """
    方案B：保壳换芯（ultralytics 8.4.137 / YOLO26 适配版）
    签名与官方 C3k2 完全一致：(c1, c2, n, c3k, e, attn, g, shortcut) + k_size
    yaml 用法与官方完全一致，参数一个字不用改：
      [-1, 2, C3k2_DB, [512, False, 0.25]]   → c3k=False 分支
      [-1, 2, C3k2_DB, [512, True]]         → c3k=True 分支
      [-1, 2, C3k2_DB, [1024, True, 0.5, True]] → attn=True 分支（YOLO26 head 深层）
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True, k_size=7):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        if attn:
            # attn 分支：Bottleneck → Bottleneck_DB，PSABlock 保留（壳内结构一一对应）
            self.m = nn.ModuleList(
                nn.Sequential(
                    Bottleneck_DB(self.c, self.c, shortcut, g, k=(3, 3), e=0.5, k_size=k_size),
                    PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
                )
                for _ in range(n)
            )
        elif c3k:
            self.m = nn.ModuleList(
                C3k_DB(self.c, self.c, 2, shortcut, g, e=0.5, k=3, k_size=k_size)
                for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(
                Bottleneck_DB(self.c, self.c, shortcut, g, k=(3, 3), e=0.5, k_size=k_size)
                for _ in range(n)
            )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1, 256, 40, 40)
    for c3k, attn in ((False, False), (True, False), (False, True)):
        m = C3k2_DB(c1=256, c2=256, n=2, c3k=c3k, e=0.25, attn=attn, k_size=7)
        y = m(x)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"C3k2_DB(c3k={c3k}, attn={attn}) : {x.shape} -> {y.shape}  参数量 {n_params/1e6:.3f} M")