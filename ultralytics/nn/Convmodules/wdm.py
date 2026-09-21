# wdm.py — WDM：Wavelet Detail Module（小波细节增强模块）
# ------------------------------------------------------------------
# 设计来源（基于 YOLO26缝合模块实验进度表 11 组实验的类级分析）：
#   1) 六类缺陷中 crazing(发纹) 与 rolled-in_scale 是长期短板，
#      两者 mAP50-95 常年 0.15~0.25，其余四类 0.34~0.59；
#      crazing 细（1~3 px）但延展长：baseline 0.534/0.206 之后
#      几乎所有模块都把它做差（DB+FEM+AFM 只剩 0.469/0.180）。
#   2) 尺度盲区：此前所有模块（FEM/AFM/P2AT/CAA/SAEAIFI）都加在
#      P5/SPPF/深层头（stride 32，20×20），细裂纹在该分辨率下接近
#      亚像素，所以 FEM 的 Sobel 纹理分支能救 scratches(+5.2)却救
#      不了 crazing 的 mAP50-95；P3 头（stride 8，80×80）从未被
#      任何增强模块触碰过。
#   3) 头侧加注意力的先例（CAA@20、P2AT@22）全部掉点 → 本模块刻意
#      不用注意力，改走频域：固定 Haar 小波分解 → 轻量精炼高频
#      子带 → 逆变换重建"纯高频细节残差" → 通道门控注入。
#   4) 门控 γ 零初始化 → 训练起步 = 恒等映射，理论上不会出现
#      CAA/SAEAIFI 那种开局就伤 mAP50 的情况；涨多少由数据决定。
#
# 机制（1 级 DWT，分辨率需为偶数，代码内有自动补齐）：
#   x ─DWT→ {LL, LH, HL, HH}（各 H/2 × W/2，Haar 正交，无学习参数）
#   三个细节子带 concat → 1×1(3c→h) → DW 3×3 → 1×1(h→3c) → 拆回
#   IDWT(0, LH', HL', HH') → 纯高频细节残差 HF（分辨率与 x 相同）
#   out = x + γ ⊙ HF         （γ 为可学习通道门控，零初始化）
#
# 用法（yaml，ultralytics 自动传入 c1；数字按缩放前写，与 FEM 行一致）：
#   [-1, 1, WDM, [256]]      # P3 头输出后（yolo26s 实际宽度 128）
#   [-1, 1, WDM, [512]]      # P4 头输出后（yolo26s 实际宽度 256）
# 只依赖 torch，不需要 import ultralytics（与 fem.py / pam.py 一致）。
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["WDM"]


# ---------------- 固定 Haar 正交小波（无学习参数） ----------------
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


class WDM(nn.Module):
    """
    Wavelet Detail Module（小波细节增强模块）

    用固定 Haar 小波把特征解耦为 {LL, LH, HL, HH}，只精炼三个高频
    细节子带，再与零 LL 一起逆变换，得到"纯高频细节残差"HF 并门控
    加回输入 → 定向强化细边界/细纹理（发纹、划痕、点蚀等）。
    高频细节直接影响高 IoU 段定位质量，因此对 mAP50-95 比对 mAP50
    更友好；门控零初始化保证起步为恒等映射，降低掉点风险。

    参数:
        c1: 输入通道数（ultralytics 自动传入；内部宽度一律以 c1 为准）
        c2: 兼容占位（yaml 第二参数），可用可不用，不影响内部宽度
        reduction: 精炼分支隐藏通道压缩率，默认 4

    输入/输出: [B, c, H, W] → [B, c, H, W]（分辨率不变）
    """

    def __init__(self, c1, c2=None, reduction=4):
        super().__init__()
        c = c1
        h = max(4, c // reduction)
        self.dwt = _DWT(c)
        self.idwt = _IDWT(c)
        # 高频细节精炼：3c → h → DW3x3 → 3c（输出仍为三个子带，可逆变换回去）
        self.refine = nn.Sequential(
            nn.Conv2d(c * 3, h, 1, bias=False),
            nn.BatchNorm2d(h),
            nn.SiLU(inplace=True),
            nn.Conv2d(h, h, 3, padding=1, groups=h, bias=False),
            nn.BatchNorm2d(h),
            nn.SiLU(inplace=True),
            nn.Conv2d(h, c * 3, 1, bias=False),
        )
        # 通道级输出门控：零初始化 → 初始输出 = 恒等映射
        self.gate = nn.Parameter(torch.zeros(c, 1, 1))

    def forward(self, x):
        h0, w0 = x.shape[-2:]
        if h0 % 2 or w0 % 2:  # 防御：奇数分辨率自动补齐再裁回
            x = F.pad(x, (0, w0 % 2, 0, h0 % 2))
        ll, lh, hl, hh = self.dwt(x)
        dlh, dhl, dhh = self.refine(torch.cat([lh, hl, hh], dim=1)).chunk(3, dim=1)
        # 纯高频细节残差：LL 置零，只让精炼后的细节子带参与逆变换
        detail = self.idwt(torch.cat([torch.zeros_like(ll), dlh, dhl, dhh], dim=1))
        out = x + self.gate * detail
        return out[..., :h0, :w0]


if __name__ == "__main__":
    torch.manual_seed(0)
    c = 128

    # 1) 完美重构校验：DWT + IDWT 不做任何修改时应无损还原
    x = torch.randn(2, c, 80, 80)
    dwt, idwt = _DWT(c), _IDWT(c)
    ll, lh, hl, hh = dwt(x)
    rec = idwt(torch.cat([ll, lh, hl, hh], dim=1))
    err = (rec - x).abs().max().item()
    print(f"完美重构误差: {err:.2e}")
    assert err < 1e-4, "Haar 正/逆变换不自洽！"

    # 2) 零初始化门控 → 输出应等于输入（恒等起步）
    m = WDM(c, c)
    y = m(x)
    print(f"零初始化恒等误差: {(y - x).abs().max().item():.2e}")
    assert torch.allclose(y, x, atol=1e-5), "门控零初始化应输出恒等！"

    # 3) 门控可学习 → 置非零后输出发生变化（梯度通路正常）
    m.gate.data.fill_(0.1)
    y2 = m(x)
    loss = y2.pow(2).mean()
    loss.backward()
    g = m.gate.grad.abs().sum().item()
    print(f"gate=0.1 后输出变化: {(y2 - x).abs().max().item():.4f}, gate 梯度量级: {g:.2f}")
    assert g > 0, "gate 无梯度，模块没有学习能力！"

    # 4) 参数量（yolo26s 的 P3 头实际宽度 128）
    n = sum(p.numel() for p in m.parameters())
    print(f"WDM(c={c}) 参数量: {n} ({n/1e3:.2f} K)")
    print("✅ 所有测试通过！")
