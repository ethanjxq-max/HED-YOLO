# sgde.py — SGDE：Semantic-Guided Detail Enhancement（语义引导细节增强模块）
# ==================================================================
# 研发背景（基于 LDTE 四轮实验的失败归因，2026-09-11）：
#   LDTE（方向纹理注入）四个配置全部未超 E 基座（最好 0.382 vs 0.388），
#   归因结论：在 1439 张小数据 + P3 共享特征图上，"带固定偏好的纹理注入"
#   （无论是方向选择 g、位置能量门控 h、还是多尺度 λ）都会造成类间跷跷板
#   —— 救 2~3 类、伤 2~3 类（v2 数据：inclusion +4.0 / scratches −4.0）。
#   设计教训：
#     ① 不要再"注入外部滤波器响应"（Gabor 核的中尺度响应与钢材背景轧制
#        纹理碰撞 → scratches/pitted 被噪声干扰）；
#     ② 不要再让模块自己学"什么纹理值得增强"（可学偏好 = 类间冲突源）；
#     ③ 应该回答"哪里有缺陷"——这是**语义**问题，不是纹理统计问题。
#
# 机制（语义引导的细节选择性增强）：
#   x = 浅层特征 [B,c,H,W]（P3 头，80×80）
#   s = 深层语义 [B,c_sem,h,w]（C2PSA 输出，20×20）
#     │
#     ├─ sem_proj: 1×1 降维 + BN + SiLU（语义编码）           [B,hidden,h,w]
#     ├─ 上采样到 (H,W) + dw3×3 + 1×1 → σ（空间门控）        [B,1,H,W]
#     │        "语义认为哪里有缺陷"
#     ▼
#   detail = x − blur5×5(x)（自身高通细节，无外部核、无参数）
#   out = x + γ ⊙ ( g ⊙ detail )     （γ 标量零初始化 = 恒等起步）
#
# 设计依据：
#   1) OverLoCK（CVPR 2025 Oral，论文清单 F4 🔴）："低频总览全局上下文引导
#      高频细节分支"——本模块是该思想在检测 neck 内的轻量实现；
#   2) Gold-YOLO（NeurIPS 2023，清单 K1 🔴）：跨层"收集-分发"融合——深层
#      语义分发到浅层；
#   3) 自有实验 E（编号14）：P5 大目标分支吃 C2PSA 语义值 +1.1 mAP50-95——
#      证明 C2PSA 全局语义对本任务是"有效信息"；本模块把同一语义源进一步
#      分发到 P3 小目标分支（与 E 修复构成"C2PSA 语义的多尺度供给"叙事）；
#   4) FreqFusion（TPAMI 2024，清单 F2 🔴）：选择性融合优于全局增强——
#      门控 g 实现"逐位置选择"；
#   5) 与 LDTE 的关键差异：不引入任何外部纹理核（detail 来自特征自身高通），
#      不做可学习通道偏好（只有 1 个标量 γ 控制全局幅度）——把"选择权"
#      完全交给语义门控，避开类间跷跷板。
#
# 用法（yaml，双输入：[-1]=P3 特征层，[11]=C2PSA 语义层）：
#   - [[-1, 11], 1, SGDE, [256]]      # P3 头 C3k2 后（yolo26s 实际 c=128, c_sem=512）
# 消融开关（yaml 透传）：
#   - [[-1, 11], 1, SGDE, [256, 8, 5, False]]   # use_gate=False（无语义引导，纯高通）
# 参数量（c=128, c_sem=512, hidden=64 实测）：≈ 33K
# 只依赖 torch，不需要 import ultralytics。
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SGDE"]


class SGDE(nn.Module):
    """
    Semantic-Guided Detail Enhancement（语义引导细节增强）

    参数:
        c1: 浅层特征通道（ultralytics 自动传入）
        c2: 输出通道（=c1，兼容占位）
        c_sem: 深层语义通道（tasks.py 的 SGDE 分支自动传入）
        reduction: 语义编码压缩率（512→64，默认 8）
        blur_k: 高通平滑核大小（默认 5）
        use_gate: 是否启用语义门控（默认 True；False=消融，纯高通增强）

    输入: x [B,c,H,W], s [B,c_sem,h,w]（h,w 可整除到 H,W）
          ultralytics 多输入层以 list 单参数调用 → forward(x=[feat, sem])
    输出: [B,c,H,W]
    """

    def __init__(self, c1, c2=None, c_sem=None, reduction=8, blur_k=5, use_gate=True):
        super().__init__()
        c = c1
        c_sem = c_sem or c2 or c
        self.use_gate = use_gate
        hidden = max(8, c_sem // reduction)

        # 语义编码：深层语义 → hidden 通道
        self.sem_proj = nn.Sequential(
            nn.Conv2d(c_sem, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        # 门控头：上采样后的语义 → 单通道空间门控（中性但非退化初始化，
        # 与 WSDM SpatialGate 同款：小随机权重 + 零偏置 → g≈0.5 有空间差异）
        self.gate_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        nn.init.normal_(self.gate_head[-1].weight, std=0.05)
        nn.init.zeros_(self.gate_head[-1].bias)

        # 细节提取：均值平滑的高通残差（无参数）
        self.blur = nn.AvgPool2d(kernel_size=blur_k, stride=1, padding=blur_k // 2)

        # 输出标量门控：零初始化 → 初始输出 = 恒等映射
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # ultralytics 多输入层调用约定：x = [浅层特征, 深层语义]（list）
        if isinstance(x, (list, tuple)):
            x_feat, s = x[0], x[1]
        else:  # 单输入时无语义引导（防御，不在 yaml 中使用）
            x_feat, s = x, x
        h, w = x_feat.shape[-2:]
        if s.shape[-2:] != (h, w):
            s_up = F.interpolate(self.sem_proj(s), size=(h, w), mode="bilinear", align_corners=False)
        else:
            s_up = self.sem_proj(s)

        if self.use_gate:
            g = torch.sigmoid(self.gate_head(s_up))  # [B,1,H,W]
        else:
            g = 1.0

        detail = x_feat - self.blur(x_feat)  # 特征自身高通细节
        return x_feat + self.gamma * (g * detail)


if __name__ == "__main__":
    torch.manual_seed(0)
    c, c_sem = 128, 512
    x = torch.randn(2, c, 80, 80)
    s = torch.randn(2, c_sem, 20, 20)

    # 1) 恒等起步：γ=0 → 输出 == 输入（list 输入接口，与 ultralytics 调用约定一致）
    m = SGDE(c, c, c_sem)
    y = m([x, s])
    print(f"零初始化恒等误差: {(y - x).abs().max().item():.2e}")
    assert torch.allclose(y, x, atol=1e-5), "γ 零初始化应输出恒等！"

    # 2) 学习通路：gamma、sem_proj、gate_head 全部有梯度
    m.gamma.data.fill_(0.5)
    y2 = m([x, s])
    y2.pow(2).mean().backward()
    grads = {
        "gamma": m.gamma.grad.abs().sum().item(),
        "sem_proj": m.sem_proj[0].weight.grad.abs().sum().item(),
        "gate_head": m.gate_head[-1].weight.grad.abs().sum().item(),
    }
    print(f"梯度量级: { {k: round(v, 5) for k, v in grads.items()} }")
    assert all(v > 0 for v in grads.values()), "存在无梯度的部件！"

    # 3) 门控确实随语义变化（不是常数图）
    with torch.no_grad():
        g1 = torch.sigmoid(m.gate_head(F.interpolate(m.sem_proj(s), size=(80, 80), mode="bilinear", align_corners=False)))
        g2 = torch.sigmoid(m.gate_head(F.interpolate(m.sem_proj(torch.randn_like(s)), size=(80, 80), mode="bilinear", align_corners=False)))
    print(f"门控: 均值 {g1.mean().item():.3f}, std {g1.std().item():.4f}, 换语义后差异 {(g1-g2).abs().mean().item():.4f}")
    assert g1.std().item() > 1e-3, "门控退化为常数图！"
    assert (g1 - g2).abs().mean().item() > 1e-3, "门控对语义不敏感！"

    # 4) use_gate=False 消融变体前向正常
    m2 = SGDE(c, c, c_sem, use_gate=False)
    assert m2([x, s]).shape == x.shape, "消融变体输出形状错误"

    # 5) 参数量
    n = sum(p.numel() for p in m.parameters())
    print(f"SGDE(c={c}, c_sem={c_sem}) 参数量: {n} ({n/1e3:.2f} K)")
    print("✅ 所有测试通过！")
