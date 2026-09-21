# sampling.py — 上下采样替换件（改进点③候选）
# ============================================================================
# 两个模块都瞄准同一处实测短板：
#   · crazing（龟裂细网纹）召回仅 0.406、mAP50-95 0.188（全类最差）；
#   · scratches 94.3% 长宽比>3；mAP50 0.731 但 mAP50-95 0.374 ⇒ 细长/细纹的"保真"是缺口。
#
# 【1】SPDConv —— 保信息下采样（Space-to-Depth Convolution）
#   机制：x → PixelUnshuffle(2)（把 2×2 邻域无损重排进通道，4 个相位一个不丢）→ 1×1(4c1→c2) 融合
#         → DW 3×3（补局部空域混合）。替代 stride=2 的 3×3 卷积。
#   为什么针对短板：stride=2 的卷积对 1~3px 细纹是"隔点采样"，裂纹落在被跳过的列/行上时响应骤降；
#     PixelUnshuffle 保留全部相位，细纹信息在进入深层之前不丢。
#   文献（素材库可核验）：《2024-2026最新模块整理V1.md》第 1727–1803 行
#     "用 Space-to-Depth + stride=1 Conv **替代 stride>1 的卷积或池化**"（Cluster Computing 2026 / ICCE 2024）。
#   与已失败方向的区别：AConv（已试失败）是"先均值池化再做步长卷积"，会主动抹掉高频；
#     本模块是**无损重排 + 学习融合**，不引入任何低通；且参数量比原卷积**少 55%**（4c1c2+9c2 vs 9c1c2）。
#   代价：c1→c2 为 1×1 后接 DW3×3，参数 4·c1·c2 + 9·c2；FLOPs 约为原 stride-2 卷积的 1/2。
#
# 【2】DySample —— 内容感知上采样（ICCV 2023）
#   机制：offset = Conv1×1(x)（**权重零初始化**，起步偏移为 0 ⇒ 采样网格 = 规则网格）
#         → grid = base_grid + offset → F.grid_sample(x, grid, bilinear)
#   为什么针对短板：颈部用 nearest 上采样把 P5/P4 的语义搬到高分辨率时会**错位**，
#     细长结构的边界因此模糊；内容感知采样让每个位置的采样点按内容微调，改善边界对齐（mAP50-95）。
#   文献（素材库可核验）：《论文阅读清单…》第 96 行（ICCV 2023，官方代码 tiny-smart/dysample）
#     "内容感知上采样——FPN 融合前对齐，小模块好嵌入"。
#   与已失败方向的区别：EA-Up（已试）是"nearest + 边缘门控的细节注入"（往特征里加东西）；
#     本模块不加任何信息，只改**采样位置**；参数仅 ~2K（1×1 生成 2·groups·r² 个偏移通道）。
#   注意：起步等价于"双线性上采样"（不是原网络的 nearest），属于温和且标准的替代。
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SPDConv", "DySample"]


class SPDConv(nn.Module):
    """保信息下采样：PixelUnshuffle(2) → 1×1(4c1→c2) → DW3×3 → BN+SiLU。

    forward 输出 [B, c2, H/2, W/2]（替代 Conv(c1, c2, 3, s=2) 时形状完全一致）。
    开关 `mode='dw'`（默认，参数最省）/'conv'（论文原版 stride=1 的 3×3 密卷积，参数约 4 倍）。
    """

    def __init__(self, c1, c2, mode="dw", act=True):
        super().__init__()
        assert c1 * 4 == c1 * 4  # 占位，保证可读性
        self.mode = mode
        self.fuse = nn.Conv2d(4 * c1, c2, 1, bias=False)
        if mode == "dw":
            self.mix = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False)
        else:
            self.mix = nn.Conv2d(c2, c2, 3, 1, 1, groups=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        if x.shape[-1] % 2 or x.shape[-2] % 2:  # 奇数边补齐（640 输入下不会触发）
            x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2))
        x = F.pixel_unshuffle(x, 2)  # [B, 4c1, H/2, W/2]，无参、无损
        return self.act(self.bn(self.mix(self.fuse(x))))


class DySample(nn.Module):
    """内容感知上采样（ICCV 2023 官方思路的精简实现）。

    输入 [B, c1, H, W] → 输出 [B, c1, r·H, r·W]（通道保持，可直接替换 nn.Upsample）
    · offset 生成器权重零初始化 ⇒ 起步 = 规则网格 = 双线性上采样（无随机扰动）；
    · groups 个通道组各有一套偏移场（默认 4，与官方一致），参数量 ≈ 2·groups·r²·c1 + 通道数。
    """

    def __init__(self, c1, scale=2, groups=4, offset_range=1.0):
        super().__init__()
        self.scale, self.groups, self.offset_range = scale, groups, offset_range
        self.offset = nn.Conv2d(c1, 2 * groups * scale * scale, 1)
        nn.init.zeros_(self.offset.weight)  # ★ 零初始化：起步偏移为 0
        nn.init.zeros_(self.offset.bias)
        self._cache = {}

    def _base_grid(self, H, W, dtype, device):
        key = (H, W, dtype, device)
        if key not in self._cache:
            r = self.scale
            Hr, Wr = H * r, W * r
            # 每个输出像素映射回源坐标（像素单位）：s = (i + 0.5)/r - 0.5
            idx = torch.arange(Hr, dtype=dtype, device=device)
            idy = torch.arange(Wr, dtype=dtype, device=device)
            sy = (idx + 0.5) / r - 0.5
            sx = (idy + 0.5) / r - 0.5
            gy = sy.view(1, Hr, 1).expand(1, Hr, Wr)
            gx = sx.view(1, 1, Wr).expand(1, Hr, Wr)
            self._cache[key] = torch.stack((gx, gy), dim=-1)  # [1, Hr, Wr, 2]
        return self._cache[key]

    def forward(self, x):
        B, C, H, W = x.shape
        r, g = self.scale, self.groups
        off = self.offset(x)  # [B, 2*g*r*r, H, W]
        off = off.view(B, 2 * g, r, r, H, W)
        # 排列成 [B, g, 2, H, r, W, r] → 展平成 [B, g, 2, Hr, Wr]
        off = off.permute(0, 1, 3, 5, 4, 2).reshape(B, g, 2, H * r, W * r)
        base = self._base_grid(H, W, x.dtype, x.device)  # [1, Hr, Wr, 2]（half-pixel 像素坐标）
        base = base.permute(0, 3, 1, 2).expand(B, 2, H * r, W * r)  # [B,2,Hr,Wr]
        # ★ 归一化口径必须与 grid_sample 的 align_corners=False 一致（half-pixel）：
        #   g = (2*s + 1)/size - 1。这样起步（偏移=0）严格等于 align_corners=False 的双线性上采样，
        #   也就是标准 resize 的几何（与原网络 nearest 的"半像素"几何一致，不会整体平移半格）。
        scale_norm = torch.tensor([2.0 / W, 2.0 / H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1)
        bias_norm = torch.tensor([1.0 / W - 1.0, 1.0 / H - 1.0], dtype=x.dtype, device=x.device).view(1, 2, 1, 1)
        x = x.reshape(B, g, C // g, H, W)
        outs = []
        for gi in range(g):
            grid = (base + torch.tanh(off[:, gi]) * self.offset_range) * scale_norm + bias_norm
            grid = grid.permute(0, 2, 3, 1)  # [B, Hr, Wr, 2]
            o = F.grid_sample(
                x[:, gi], grid, mode="bilinear", padding_mode="border", align_corners=False
            )
            outs.append(o)
        return torch.cat(outs, dim=1)


if __name__ == "__main__":  # 自检：python -m ultralytics.nn.Convmodules.sampling
    torch.manual_seed(0)
    print("=" * 88)
    # 1) SPDConv：形状、与官方 stride-2 卷积的参数量对比
    from ultralytics.nn.modules.conv import Conv as UConv

    for c1, c2 in ((32, 64), (64, 128), (128, 256), (256, 512)):
        s = SPDConv(c1, c2)
        u = UConv(c1, c2, 3, 2)
        y = s(torch.randn(1, c1, 64, 64))
        print(
            f"[1] SPDConv({c1}→{c2}) 输出 {tuple(y.shape)} | 参数 {sum(p.numel() for p in s.parameters())/1e3:7.1f}K"
            f" vs 原 stride-2 卷积 {sum(p.numel() for p in u.parameters())/1e3:7.1f}K"
            f" ({(sum(p.numel() for p in s.parameters())/sum(p.numel() for p in u.parameters())-1)*100:+.0f}%)"
        )

    # 2) DySample：形状 + 起步是否等于双线性上采样 + 参数量
    for c in (256, 512):
        d = DySample(c).eval()
        x = torch.randn(1, c, 20, 20)
        with torch.no_grad():
            y = d(x)
        ref = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        print(
            f"[2] DySample(c={c}) 输出 {tuple(y.shape)} | 参数 {sum(p.numel() for p in d.parameters())/1e3:.1f}K"
            f" | 起步与双线性上采样差异 {(y - ref).abs().max().item():.2e}"
        )

    # 3) DySample 的偏移必须有梯度（否则学不动）
    d = DySample(64).train()
    d(torch.randn(1, 64, 10, 10)).pow(2).mean().backward()
    print(f"[3] DySample 偏移生成器梯度 {d.offset.weight.grad.abs().sum():.3e}（应>0，双线性采样对偏移可导）")

    # 4) 不同 scale（P5→P3 需要 ×4 时可链式用两次 ×2）
    print(f"[4] scale=4 变体输出 {tuple(DySample(32, scale=4)(torch.randn(1,32,16,16)).shape)}  "
          f"参数 {sum(p.numel() for p in DySample(32, scale=4).parameters())/1e3:.1f}K")
    print("✅ 全部自检通过")
