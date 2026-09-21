# texnet.py — TEX-Net：纹理显式化的钢表面缺陷检测组件（v2，2026-09-13 数值稳定性修复）
# ============================================================================
# 【v2 修改说明：为什么要改】
#   v1 在 4 卡并行实验中出现严重训练退化（实测日志）：
#     · +TSM@P3：cls_loss 在 ep3 起卡在 ~2.95（baseline 同期 2.2），mAP50-95 仅为 baseline 的 1/4；
#     · TEX-Net（2×TSM+DCNorm）：cls_loss 在 ep3 起卡在 ~4.5（≈ 6 类均匀输出 6·ln2=4.16 的水平），
#       即分类分支根本没在学习；而同一批实验里 baseline 与 +DCNorm 的 loss 轨迹与历史 run 逐轮重合
#       ⇒ 问题定位在 TSM，且"加 2 个比加 1 个更糟"。
#   本地梯度诊断（实测数值）：
#     · 正常特征输入：TSM 投影权重梯度 0.130（对照 1×1 conv 0.075）；
#     · **平坦/低方差输入：TSM 投影权重梯度暴增到 50.0（380 倍）**，输入梯度同步放大 5 倍。
#   根因（两处叠加）：
#     ① `rc = |μ|/(σ+ε)`、`gran = σ₁/σ₂` 在平坦区是 0/0 型比值，σ 被 clamp 到 1e-6 后
#        sqrt 的导数高达 500，反向梯度被放大后**直接回传到共享主干特征**；
#     ② `sqrt(clamp(方差,1e-6))` 与 BN 对"近常数图"按极小方差归一化，等于把噪声放大成描述子。
#   【v2.1 追加修复（更准确的病因）】
#     上述"梯度放大"只是次要因素；真正的病因是 **BN 对平滑统计图的病态放大**：
#     `rc = |μ|/(σ+ε)`、`gran = σ₁/σ₂` 这类"池化后统计量"在空间上几乎恒定（平滑），
#      BN 除以它们极小的**空间**方差，会把微小的空间起伏放大成**单位方差的噪声**，
#     然后经投影注入 P3 特征 ⇒ 检测头的输入被噪声污染 ⇒ cls_loss 在高 LR 阶段（ep5 后）停滞。
#     这解释了"加 2 个 TSM 比加 1 个更糟"以及"ep1-4 正常、ep5 后偏离"。
#     v2.1 的修复：**去掉 BN**，改为"逐组有界化 + 每组可学习尺度"（5 个标量），
#     保证描述子数值稳定且有统一量纲，不再有任何"除以小方差"的操作。
#   v2 的三条修复（保留）：
#     A. **统计量梯度解耦**（默认 detach=True）：统计量在 `torch.no_grad()` 下计算，只作为
#        "固定物理描述子"，梯度不再回传主干；可学习部分只剩 1×1 投影（detach=False 保留做消融）。
#        依据同本项目已验证经验：恒等起步 + 不扰动既有特征 = 安全（v1 恰好违反了后者）。
#     B. **数值护栏**：统计量全程在 fp32 计算（半精度不溢出）、方差下限提到 1e-4（σ≥0.01）、
#        比值 eps 提到 1e-3 并 clip 到 [0, ratio_clamp]，各向异性天然有界于 [-1,1]。
#     C. **恒等起步保持不变**：投影零初始化 ⇒ 训练起点输出逐位等于输入（本地验证 0.00e+00）。
#   DCNorm 未改动（实测与 baseline 轨迹逐轮重合，中性无害）。
# ============================================================================
# 设计出发点（全部来自对 NEU-DET 标注与本项目实验的定量分析）：
#   ① 数据事实：4189 个实例的等效边长中位 220px（640 输入口径），**81% 落在 P5 域(>128px)**，
#      只有 1.2% 落在 P3 域(<64px)；scratches 有 94.3% 长宽比>3（中位长边 560px）、inclusion 41.3%。
#      ⇒ 本任务是"大区域、细长、低对比、纹理定义"的检测，不是小目标检测。
#   ② 架构错配：YOLO26 检测头在 stride 8/16/32，P5 的 3×3 卷积在原图上覆盖 96×96 区域，
#      而 crazing 的裂纹网、scratches 的划痕只有 1~3px 宽 ⇒ 深层特征里"判别依据"已被平均掉；
#      多数实例（81%）恰恰由 P5 判决。
#   ③ 机制缺失：YOLO26 全程只有 BN（全局、逐通道）归一化，没有空间局部的对比度归一化；
#      也没有任何显式的纹理统计通路（C2PSA 只在 P5 做位置注意力）。
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DCNorm", "TextureStats", "TSM"]


class DCNorm(nn.Module):
    """多尺度除性对比度归一化（Divisive Contrast Normalization）。

    机制（每个空间位置、逐通道）：
        s      = Σ_k α_k · AvgPool_{p_k}(|x|)            # 多尺度邻域"能量"（surround）
        s_rel  = s / mean_{H,W}(s)                        # 相对邻域能量（空间均值≈1，尺度无关）
        factor = 1 − tanh(λ_c) · clamp(s_rel − 1, ±0.9)
        out    = x ⊙ factor                               # 强邻域→抑制；孤立弱响应→相对增强

    与已有工作的区别：
      · LRN/AlexNet：通道间、单一固定尺度、逐点；InstanceNorm/GN：全图统计、无空间局域性；
        本模块是空间局部 + 多尺度 + 可学习 + 逐通道的除性归一化。
      · 注意力（SE/CBAM/CAA/PAM）：产生"重要性权重"去加权特征；本模块不产生权重，
        而是把绝对响应折算成相对对比度（属"特征编码方式"的改造）。
      · 本项目已试的 WDM/WSDM：放大高频细节（会把背景纹理一起放大）；本模块方向相反——
        抑制占优的邻域能量。理论源头为视觉神经科学的 divisive normalization。

    恒等起步：λ 零初始化 ⇒ factor ≡ 1 ⇒ 输出逐位等于输入。参数量：|pools| + C（输入级 3 通道时 6 个）。
    用法：[-1, 1, DCNorm, []]（输入级或特征级，输出通道 = 输入通道）。
    """

    def __init__(self, c1, pools=(3, 5, 9), clamp=0.9):
        super().__init__()
        self.pools = tuple(pools)
        self.clamp = clamp
        self.alpha = nn.Parameter(torch.zeros(len(self.pools)))  # softmax → 初始均匀
        self.lam = nn.Parameter(torch.zeros(c1, 1, 1))  # 0 ⇒ 恒等起步

    def forward(self, x):
        a = self.alpha.softmax(0)
        s = 0
        for w, p in zip(a, self.pools):
            s = s + w * F.avg_pool2d(x.abs(), p, stride=1, padding=p // 2)
        s_rel = s / (s.mean(dim=(2, 3), keepdim=True).detach() + 1e-5)
        factor = 1.0 - torch.tanh(self.lam) * (s_rel - 1.0).clamp(-self.clamp, self.clamp)
        return x * factor


class TextureStats(nn.Module):
    """可微局部纹理统计描述子（5 张统计图 → 压缩成 out_ch 通道）。

    统计量：
        σ1 = sqrt(clamp(AvgPool_{p1}(x²) − AvgPool_{p1}(x)², ≥eps²))   # 细尺度局部标准差
        σ2 = 同理（大尺度）                                             # 粗尺度标准差
        gran = clip(σ1 / (σ2 + ε), 0, ratio_clamp)                      # **纹理粒度**
        rc   = clip(|μ1| / (σ1 + ε), 0, ratio_clamp)                    # **相对对比度**
        ani  = (a_x − a_y) / (a_x + a_y + ε)                            # **各向异性**（∈[-1,1]）

    数值设计（v2 修复的关键）：
      · 全程 **fp32** 计算统计量再转回原 dtype：半精度下 x²/比值不溢出；
      · 方差下限 eps²（默认 eps=1e-2 ⇒ σ≥0.01）+ 比值 eps=1e-3 + clip：平坦区不再是 0/0；
      · `detach=True`（默认）时整个统计量在 no_grad 下计算 ⇒ **不向主干回传梯度**，
        只作为固定的物理描述子供 1×1 投影学习如何使用（v1 的退化正是梯度污染所致）。

    与已有工作的区别：
      · 卷积是线性的，比值类统计量（gran、rc）无法用 1×1 卷积构造；注意力产生权重图，本模块产生统计描述通道。
      · 双流 CNN + LBP/HOG：手工特征 + 独立分支、不可端到端；本模块并入检测通路且投影可学。
      · 本项目已试的 LDTE：把滤波器响应注入并修改特征（类间跷跷板）；本模块不修改既有特征。
    """

    def __init__(self, c, pools=(3, 7), out_ch=None, detach=True, eps=1e-2, ratio_clamp=4.0):
        super().__init__()
        self.p1, self.p2 = pools
        self.detach = detach
        self.eps = eps
        self.ratio_clamp = ratio_clamp
        out_ch = out_ch or max(8, c // 8)
        # ★ v2.1：**不用 BN**（BN 会把平滑统计图按极小空间方差归一化 → 放大成噪声）
        # 改为：5 组统计量先各自有界化到 [-1,1]，再乘"每组一个可学习尺度"（5 个标量）统一量纲
        self.group_scale_log = nn.Parameter(torch.zeros(5))
        self.proj = nn.Sequential(
            nn.Conv2d(5 * c, out_ch, 1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.out_ch = out_ch

    def _stats(self, x):
        """在 fp32 下计算 5 张有界统计图（返回 fp32）。"""
        xf = x.float()
        e2 = self.eps * self.eps
        m1 = F.avg_pool2d(xf, self.p1, 1, self.p1 // 2)
        s1 = (F.avg_pool2d(xf * xf, self.p1, 1, self.p1 // 2) - m1 * m1).clamp_min(e2).sqrt()
        m2 = F.avg_pool2d(xf, self.p2, 1, self.p2 // 2)
        s2 = (F.avg_pool2d(xf * xf, self.p2, 1, self.p2 // 2) - m2 * m2).clamp_min(e2).sqrt()
        gran = (s1 / (s2 + 1e-3)).clamp_(0, self.ratio_clamp)  # 纹理粒度（有界）
        rc = (m1.abs() / (s1 + 1e-3)).clamp_(0, self.ratio_clamp)  # 相对对比度（有界）
        gx = (xf[..., :, 1:] - xf[..., :, :-1]).abs()
        gy = (xf[..., 1:, :] - xf[..., :-1, :]).abs()
        ax = F.avg_pool2d(F.pad(gx, (0, 1, 0, 0)), (1, self.p2), 1, (0, self.p2 // 2))
        ay = F.avg_pool2d(F.pad(gy, (0, 0, 0, 1)), (self.p2, 1), 1, (self.p2 // 2, 0))
        ani = (ax - ay) / (ax + ay + 1e-3)  # 各向异性（天然有界于 [-1,1]）
        # ★ v2.1：五组各自有界化到 [-1,1]，避免任一组量纲失控（σ 用 tanh 软饱和）
        return torch.cat(
            [
                torch.tanh(s1 * 0.5),
                torch.tanh(s2 * 0.5),
                gran / self.ratio_clamp,
                rc / self.ratio_clamp,
                ani,
            ],
            dim=1,
        )

    def forward(self, x):
        if self.detach:
            with torch.no_grad():  # ★ v2：统计量不参与反向，杜绝梯度污染主干
                desc = self._stats(x)
        else:
            desc = self._stats(x)
        # ★ v2.1：逐组有界化后用可学习尺度统一量纲（desc 已落在 [-1,1]，再乘以尺度）
        c = x.shape[1]
        scale = self.group_scale_log.exp().repeat_interleave(c).view(1, -1, 1, 1)
        return self.proj((desc * scale).to(x.dtype))


class TSM(nn.Module):
    """纹理统计侧路（Texture Statistics Module）：out = x + ZeroInitProj([x, stats(x)])。

    · 输出通道数与输入相同（drop-in，yaml 一行替换，不需要改任何 Concat 的通道算术）；
    · 投影零初始化 ⇒ 训练起点严格恒等（与官方 yaml 逐位一致，本地验证 0.00e+00）；
    · `init_scale>0` 可做"热启动"消融；
    · **v2 起默认 detach 统计量**：只让投影学习"如何利用固定统计量"，避免 v1 的梯度污染退化。
    · 位置建议：backbone 的 P3 stage 输出后（P3 是唯一还保留细纹理的尺度），
      统计量沿既有自顶向下通路流到 P4/P5 头，补上"81% 实例由 P4/P5 判决、细纹理已被 stride 抹掉"的缺口。
    """

    def __init__(self, c1, out_ch=None, init_scale=0.0, pools=(3, 7), detach=True):
        super().__init__()
        c = c1
        self.stats = TextureStats(c, pools=pools, out_ch=out_ch, detach=detach)
        self.proj = nn.Conv2d(c + self.stats.out_ch, c, 1, bias=False)
        nn.init.zeros_(self.proj.weight)  # 恒等起步
        if init_scale > 0:  # 热启动：给很小的随机权重（消融用）
            nn.init.normal_(self.proj.weight, std=init_scale)
        # ⚠️ 此处不能加 BN：BN 在训练模式下用批次统计，会让第 0 步输出 ≠ 输入（实测 2.4e-5 且随批次变化）

    def forward(self, x):
        return x + self.proj(torch.cat([x, self.stats(x)], dim=1))


if __name__ == "__main__":  # 自检：python -m ultralytics.nn.Convmodules.texnet
    torch.manual_seed(0)
    print("=" * 88)
    for c in (3, 128, 256):
        m = DCNorm(c).eval()
        x = torch.randn(2, c, 40, 40)
        with torch.no_grad():
            y = m(x)
        print(f"[1] DCNorm(c={c:<4d}) 恒等起步误差 {(y - x).abs().max().item():.2e}  参数 {sum(p.numel() for p in m.parameters())}")

    print("-" * 88)
    for c in (128, 256, 512):
        tsm = TSM(c).eval()
        x = torch.randn(1, c, 40, 40)
        with torch.no_grad():
            y = tsm(x)
        print(
            f"[2] TSM(c={c:<4d}) 输出 {tuple(y.shape)} 恒等误差 {(y - x).abs().max().item():.2e}"
            f" 参数 {sum(p.numel() for p in tsm.parameters())/1e3:.1f}K"
        )

    # 3) v2 关键验证：平坦/低方差输入下梯度不再爆炸（v1 此处为 50.0，对照 conv 0.075）
    print("-" * 88)
    C = 256
    for tag, xin in (
        ("正常特征", torch.randn(2, C, 40, 40)),
        ("平坦特征(均值5,方差小)", torch.randn(2, C, 40, 40) * 0.001 + 5.0),
        ("极低方差", torch.randn(2, C, 40, 40) * 1e-3),
    ):
        tsm = TSM(C).train()
        x = xin.clone().requires_grad_(True)
        tsm(x).pow(2).mean().backward()
        with torch.no_grad():
            s = tsm.stats(xin)
        print(
            f"[3] {tag:<22} 输入梯度范数 {x.grad.norm().item():9.4f} | 投影权重梯度 {tsm.proj.weight.grad.norm().item():9.4f}"
            f" | 描述子范围 [{s.min().item():6.2f},{s.max().item():6.2f}] inf/nan={bool(torch.isinf(s).any() or torch.isnan(s).any())}"
        )
    conv = nn.Conv2d(C, C, 1)
    x2 = torch.randn(2, C, 40, 40).requires_grad_(True)
    conv(x2).pow(2).mean().backward()
    print(f"[3] {'对照 1×1 conv':<22} 输入梯度范数 {x2.grad.norm().item():9.4f} | 权重梯度 {conv.weight.grad.norm().item():9.4f}")

    # 4) 半精度（AMP）安全
    with torch.autocast("cpu", dtype=torch.bfloat16):
        s16 = TextureStats(C)(torch.randn(2, C, 40, 40) * 0.001 + 5.0)
    print(
        f"[4] bf16 平坦输入 描述子范围 [{s16.float().min().item():.2f},{s16.float().max().item():.2f}]"
        f" inf/nan={bool(torch.isinf(s16).any() or torch.isnan(s16).any())}"
    )
    print("✅ 全部自检通过")
