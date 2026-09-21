# analyze_ltdte.py — 训练后 LDTE 参数诊断（零成本，不占 GPU）
# 用法：python analyze_ltdte.py <best.pt 路径>
# 输出：gate 幅度分布（判断注入是否激活）
#       Δθ/σ/λ/长宽比 训练后取值（判断方向核是否学动/分化）
#       每个方向基重建核的能量方向谱（判断核是否保持方向性）
import math
import sys

import torch

from ultralytics import YOLO

ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/neu_db_fem_ltdte_p3/db_fem_ltdte_p3/weights/best.pt"
m = YOLO(ckpt).model

# 找 LDTE 层
ldte, idx = None, None
for i, layer in enumerate(m.model):
    if type(layer).__name__ == "LDTE":
        ldte, idx = layer, i
        break
if ldte is None:
    print("❌ 未找到 LDTE 层（检查模型是否包含 LDTE）")
    sys.exit(1)
print(f"找到 LDTE 层 @ layer {idx}，c={ldte.gabor.c}")

g = ldte.gabor
with torch.no_grad():
    # ---- 1) 输出门控 gate：注入强度的核心证据 ----
    gate = ldte.gate.flatten()
    print("\n[1] 输出门控 gate（0=恒等，|·|越大注入越强）")
    print(f"    mean={gate.mean().item():.4f}  std={gate.std().item():.4f}")
    print(f"    abs 分位数: p50={gate.abs().median().item():.4f} "
          f"p90={gate.abs().quantile(0.9).item():.4f} max={gate.abs().max().item():.4f}")
    print(f"    |gate|>0.05 通道占比: {(gate.abs() > 0.05).float().mean().item()*100:.1f}%")
    print(f"    |gate|>0.2  通道占比: {(gate.abs() > 0.2).float().mean().item()*100:.1f}%")

    # ---- 2) 方向偏移 Δθ（学了没有） ----
    dtheta = g.dtheta.flatten()
    print("\n[2] 方向偏移 Δθ（初始全 0；越大=方向自适应越强）")
    print(f"    {[round(v, 4) for v in dtheta.tolist()]}")

    # ---- 3) Gabor 物理参数分布（σ 尺度 / λ 波长 / 长宽比） ----
    sigma = torch.nn.functional.softplus(g.log_sigma) + 0.5
    lam = torch.nn.functional.softplus(g.log_lambda) + 1.0
    ratio = torch.nn.functional.softplus(g.log_ratio) + 0.1
    print("\n[3] Gabor 参数（训练后）")
    print(f"    σ 尺度    : mean={sigma.mean().item():.2f}  min={sigma.min().item():.2f}  max={sigma.max().item():.2f}")
    print(f"    λ 波长(px): mean={lam.mean().item():.2f}  min={lam.min().item():.2f}  max={lam.max().item():.2f}  "
          f"std={lam.std().item():.2f}")
    print(f"    长宽比     : mean={ratio.mean().item():.2f}  min={ratio.min().item():.2f}  max={ratio.max().item():.2f}")
    # 初始值对照（softplus(0)+0.5≈1.19 / softplus(ln4)+1≈2.61 / softplus(ln0.5)+0.1≈0.51
    #  — 若训练后相差大说明参数学动了）
    print(f"    (初始参考: σ≈1.19, λ≈2.61, 长宽比≈0.51 — 若相差大说明参数学动了)")

    # ---- 4) 方向核方向谱检查：FFT 主频方向角 vs 理论方向 ----
    print("\n[4] 方向核方向选择性（FFT 主频方向角，应≈理论值 θ+90°）")
    kernels = g._build_kernels("cpu", torch.float32)  # [c,1,k,k]
    c = kernels.shape[0]
    num_dirs = g.num_dirs
    k = g.k
    # 通道按 i%8 轮流分配方向 → 需先 view(c/N,N) 再转置分组，不能直接 view(N, c/N)
    # grouped: [N, c/N, 1, k, k]；组内 16 通道平均后得每方向基的代表核 [N, k, k]
    grouped = kernels.view(c // num_dirs, num_dirs, 1, k, k).permute(1, 0, 2, 3, 4)
    avg_kernels = grouped.mean(dim=1)[:, 0]  # [N, k, k]
    for d in range(num_dirs):
        kk = avg_kernels[d]  # [k,k]
        # 零填充到 32×32 提高频谱角分辨率
        spec = torch.fft.fftshift(torch.fft.fft2(kk, s=(32, 32))).abs()
        center = 16
        spec[center - 2:center + 3, center - 2:center + 3] = 0  # 去 DC 及低频邻域
        peak = (spec == spec.max()).nonzero()
        if len(peak) == 0:
            print(f"    θ基={d}: 频谱为空（核退化?）")
            continue
        p = peak[0]
        dy, dx = (p[0].item() - center), (p[1].item() - center)
        # 核沿 x_t=x·cosθ+y·sinθ 振荡 → 频峰方向角 = θ（atan2(行频率, 列频率)）
        ang = math.degrees(math.atan2(dy, dx)) % 180.0
        theo = math.degrees(math.pi * d / num_dirs) % 180.0
        directional = (spec.max().item() / (spec.sum().item() + 1e-9))
        print(f"    θ基={d}: 实测主频方向={ang:5.1f}° (理论 θ={theo:5.1f}°)  方向能量占比={directional:.2f}")
    # 方向核是否分化：各方向基平均核互相关应低
    flat = avg_kernels.view(num_dirs, -1)
    flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-6)
    corr = (flat @ flat.T).abs()
    off_diag = corr[~torch.eye(num_dirs, dtype=torch.bool)].mean().item()
    print(f"    方向基核互相关系数(off-diag mean): {off_diag:.3f}  (<0.3 说明方向分化好)")

    # ---- 5) 门控 MLP 是否激活 ----
    w1 = ldte.dir_gate[1].weight  # c→hidden
    print("\n[5] 方向门控 MLP: conv1 weight abs mean =", round(w1.abs().mean().item(), 6),
          " bias1 =", round(ldte.dir_gate[1].bias.abs().mean().item(), 4))

# ---- 判读建议 ----
print("\n===== 判读 =====")
print("1) gate 若几乎全 0 → 注入没激活 → 检查梯度/训练（正常 250ep 应有不少通道 |gate|>0.05）")
print("2) Δθ 若≈全 0 且 σ/λ 变化小 → 方向自由度没用上 → 考虑提高 LDTE 层学习率或去掉 sigma 下限")
print("3) 方向核若互相关系数高（>0.5）→ 核退化成各向同性 → 机制失效")
print("4) 若 gate 健康、核分化好 → 机制在工作，是'注入伤害非周期类'的类级问题 → 才考虑内部消融")
