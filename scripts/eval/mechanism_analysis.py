# mechanism_analysis.py — HDB / TSG-EM 的机理支撑实验（三部分，本地 CPU 也能跑）
# ============================================================================
# A. 缺陷形态的频谱先验：统计 NEU-DET 中"缺陷区域"与"背景区域"的功率谱形状与绝对能量
#    → 支撑"钢缺陷以中高频细节为主、背景以低频为主；低对比导致缺陷调制深度弱"这一前提
# B. 模块频率响应：对标准 Bottleneck 与 HDB 输入不同频率/方向的栅格信号，测输出增益
#    → 支撑"并行双分支保留高频、且 1×k / k×1 分支带来方向选择性"
# C. 有效感受野（ERF）：对基线模型与含 HDB/TSG-EM 的模型，用梯度法测 P3 层中心点的 ERF
#    → 支撑"并行大核分支扩大了低对比区域的有效上下文，而不牺牲细节分支的局部性"
# 输出：figures/fig_spectrum_prior.png、figures/fig_freq_response.png、figures/fig_erf.png
# 用法（服务器，用训练好的权重）：
#   python mechanism_analysis.py --data ultralytics/cfg/datasets/neu_det.yaml \
#       --baseline runs/ab_base/ab_base/weights/best.pt \
#       --ours runs/ab_db_fem/ab_db_fem/weights/best.pt
# ============================================================================
import argparse
import pathlib

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PATCH = 64          # 频谱分析用的 patch 尺寸（像素）
BINS = 16           # 径向频率分箱数


# ---------------------------------------------------------------- A. 频谱先验
def radial_spectrum(patch):
    """返回按半径归一化的功率谱（BINS 个频带）"""
    p = patch - patch.mean()
    win = np.outer(np.hanning(p.shape[0]), np.hanning(p.shape[1]))
    F2 = np.fft.fftshift(np.abs(np.fft.fft2(p * win)) ** 2)
    h, w = F2.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - cy) / (h / 2)) ** 2 + ((xx - cx) / (w / 2)) ** 2)  # 归一到 [0, ~1.4]
    r = r / r.max()
    out = np.zeros(BINS)
    for b in range(BINS):
        m = (r >= b / BINS) & (r < (b + 1) / BINS)
        if m.any():
            out[b] = F2[m].mean()
    return out


def analyze_spectrum(data_yaml, out_png, n_img=200, n_bg=4, seed=0):
    import yaml
    d = yaml.safe_load(open(data_yaml, encoding="utf-8"))
    root, val = pathlib.Path(d.get("path", ".")), str(d["val"])
    img_dir = pathlib.Path(val) if val.startswith("/") else root / val
    lab_dir = pathlib.Path(str(img_dir).replace("images", "labels"))
    from PIL import Image
    rng = np.random.default_rng(seed)
    spec_d, spec_b, e_d, e_b = [], [], [], []
    files = sorted(img_dir.glob("*.jpg"))[:n_img]
    for f in files:
        lp = lab_dir / (f.stem + ".txt")
        if not lp.exists():
            continue
        im = np.array(Image.open(f).convert("L"), dtype=np.float32)
        H, W = im.shape
        boxes = np.loadtxt(lp, ndmin=2)
        for row in boxes:
            _, cx, cy, bw, bh = row[:5]
            x1 = int((cx - bw / 2) * W); y1 = int((cy - bh / 2) * H)
            x2 = int((cx + bw / 2) * W); y2 = int((cy + bh / 2) * H)
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            crop = im[y1:y2, x1:x2]
            crop = np.array(Image.fromarray(crop.astype(np.uint8)).resize((PATCH, PATCH), Image.BILINEAR), np.float32)
            s = radial_spectrum(crop)
            spec_d.append(s / s.sum()); e_d.append(crop.std())
            for _ in range(n_bg):  # 背景：随机取不落进任何框的 patch
                for _try in range(20):
                    bx = rng.integers(0, max(1, W - PATCH)); by = rng.integers(0, max(1, H - PATCH))
                    ok = True
                    for r2 in boxes:
                        _, c2x, c2y, b2w, b2h = r2[:5]
                        if abs(bx + PATCH / 2 - c2x * W) < b2w * W / 2 + 8 and abs(by + PATCH / 2 - c2y * H) < b2h * H / 2 + 8:
                            ok = False; break
                    if ok:
                        break
                crop = im[by:by + PATCH, bx:bx + PATCH]
                if crop.shape != (PATCH, PATCH):
                    continue
                s = radial_spectrum(crop)
                spec_b.append(s / s.sum()); e_b.append(crop.std())
    D = np.mean(spec_d, axis=0); B = np.mean(spec_b, axis=0)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    x = (np.arange(BINS) + 0.5) / BINS
    ax[0].plot(x, D, "o-", color="#c0392b", label=f"Defect regions (std={np.mean(e_d):.1f})")
    ax[0].plot(x, B, "s--", color="#7f8c8d", label=f"Background (std={np.mean(e_b):.1f})")
    ax[0].set_xlabel("Normalized spatial frequency")
    ax[0].set_ylabel("Normalized power")
    ax[0].set_title("(a) Radial power spectrum (shape)", fontsize=10)
    ax[0].legend(fontsize=8, frameon=False); ax[0].grid(ls=":", alpha=0.5)
    rel = D / (B + 1e-12)
    ax[1].bar(x, rel, width=0.055, color="#c0392b", alpha=0.85)
    ax[1].axhline(1.0, color="k", lw=0.8, ls=":")
    ax[1].set_xlabel("Normalized spatial frequency")
    ax[1].set_ylabel("Defect / Background")
    ax[1].set_title("(b) Relative spectral energy ratio", fontsize=10)
    ax[1].grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)
    print(f"[A] saved {out_png} | 缺陷 patch 数 {len(spec_d)}，背景 patch 数 {len(spec_b)}")
    print(f"    缺陷平均灰度标准差 {np.mean(e_d):.2f} vs 背景 {np.mean(e_b):.2f}（对比度指标）")
    return D, B


# ------------------------------------------------------- B. 模块频率响应
def grating(freq, theta, size=128, device="cpu"):
    y, x = torch.meshgrid(torch.arange(size, dtype=torch.float32), torch.arange(size, dtype=torch.float32), indexing="ij")
    k = 2 * np.pi * freq
    g = torch.sin(k * (x * np.cos(theta) + y * np.sin(theta)))
    return g[None, None].to(device)


def module_gain(module, freqs, thetas, size=128, device="cpu"):
    module.eval()
    gains = np.zeros((len(thetas), len(freqs)))
    with torch.no_grad():
        for i, th in enumerate(thetas):
            for j, fr in enumerate(freqs):
                x = grating(fr, th, size, device).repeat(1, 8, 1, 1)  # 8 通道便于复用同一模块
                y = module(x)
                gains[i, j] = (y.std().item() + 1e-8) / (x.std().item() + 1e-8)
    return gains


def analyze_response(out_png, weights=None):
    from ultralytics.nn.modules.block import Bottleneck
    from ultralytics.nn.Convmodules.repblock import Bottleneck_MRB  # noqa: F401  (确保自定义模块已注册)
    from ultralytics.nn.Convmodules.db import Bottleneck_DB
    torch.manual_seed(0)
    m_std = Bottleneck(8, 8, shortcut=True, k=(3, 3), e=1.0).eval()
    m_hdb = Bottleneck_DB(8, 8, shortcut=True, k=(3, 3), e=1.0).eval()
    if weights:  # 可选：从训练权重里取同名子模块（名称不匹配则保持随机初始化）
        try:
            ck = torch.load(weights, map_location="cpu", weights_only=False)
            sd = (ck["model"] if isinstance(ck, dict) and "model" in ck else ck).float().state_dict()
            for m, key in ((m_hdb, "branch_a")):
                pass
            print("[B] 提示：本脚本默认用随机初始化的同构模块测'结构本身'的频率响应；"
                  "若要与训练权重对应，请按需替换 map 名称。")
        except Exception as e:
            print("[B] 权重加载跳过:", type(e).__name__)
    freqs = np.linspace(0.02, 0.45, 15)
    thetas = [0.0, np.pi / 2, np.pi / 4]     # 0°(水平), 90°(垂直), 45°
    g_std = module_gain(m_std, freqs, thetas)
    g_hdb = module_gain(m_hdb, freqs, thetas)
    names = ["0° (horizontal)", "90° (vertical)", "45° (diagonal)"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for i in range(3):
        ax[i].plot(freqs, g_std[i], "o--", color="#7f8c8d", label="Serial bottleneck (3×3→3×3)")
        ax[i].plot(freqs, g_hdb[i], "s-", color="#c0392b", label="HDB (parallel)")
        ax[i].set_title(names[i], fontsize=10); ax[i].set_xlabel("Spatial frequency (cycles/px)")
        ax[i].grid(ls=":", alpha=0.5)
    ax[0].set_ylabel("Output/input amplitude gain"); ax[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)
    print(f"[B] saved {out_png}")
    for i, nm in enumerate(names):
        print(f"    {nm:<16} 高频段(≥0.30)平均增益  串行 {g_std[i][-5:].mean():.3f}  并行 {g_hdb[i][-5:].mean():.3f}")


# ------------------------------------------------------- C. 有效感受野（ERF）
def erf_map(model, imgsz=320):
    """用 forward hook 抓取 stride=8 的特征层（P3），以其中心点响应回传梯度得到 ERF。

    不依赖模型前向的返回格式（ultralytics 在不同模式下返回 tuple / dict 不一致）。
    """
    net = model.model if hasattr(model, "model") else model
    net.eval()
    captured = {}

    def make_hook():
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, (list, tuple)) else out
            if hasattr(o, "shape") and o.dim() == 4 and o.shape[-1] == imgsz // 8:
                captured.setdefault("f", o)
        return hook

    handles = [lyr.register_forward_hook(make_hook()) for lyr in net]
    x = torch.zeros(1, 3, imgsz, imgsz, requires_grad=True)
    forward = getattr(model, "_predict_once", None) or getattr(model, "predict", None) or net
    try:
        forward(x)      # ⚠ 必须用 DetectionModel 的前向（会按 m.f 路由多输入层）；直接调用 nn.Sequential 会在 Concat 报错
    finally:
        for h in handles:
            h.remove()
    if "f" not in captured:
        raise RuntimeError("未捕获到 stride=8 的特征层")
    f = captured["f"]
    loss = f[0, :, f.shape[-2] // 2, f.shape[-1] // 2].sum()
    model.zero_grad()
    loss.backward()
    g = x.grad.detach().abs().sum(1)[0].numpy()
    g = g / max(g.max(), 1e-12)
    area = float((g > 0.1).sum())        # 有效感受野面积（>10% 峰值）
    return g, area


def analyze_erf(out_png, baseline_ckpt, ours_ckpt, imgsz=320):
    from ultralytics import YOLO
    res = {}
    for tag, ck in (("baseline", baseline_ckpt), ("ours", ours_ckpt)):
        try:
            y = YOLO(ck)
            g, a = erf_map(y.model, imgsz)
            res[tag] = (g, a)
        except Exception as e:
            print(f"[C] {tag} 失败: {type(e).__name__}: {str(e)[:60]}")
    if not res:
        print("[C] 无结果，跳过"); return
    fig, ax = plt.subplots(1, len(res), figsize=(4.2 * len(res), 4))
    ax = np.atleast_1d(ax)
    for a_, (tag, (g, area)) in zip(ax, res.items()):
        im = a_.imshow(g, cmap="inferno"); a_.set_title(f"{tag}\nERF area(>0.1) = {area:.0f} px", fontsize=9)
        a_.axis("off"); fig.colorbar(im, ax=a_, fraction=0.046)
    fig.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)
    print(f"[C] saved {out_png} | " + "  ".join(f"{k}: {v[1]:.0f} px" for k, v in res.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ultralytics/cfg/datasets/neu_det.yaml")
    ap.add_argument("--baseline", default="", help="基线权重（ERF 用）")
    ap.add_argument("--ours", default="", help="本文模型权重（ERF 用）")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--nimg", type=int, default=200)
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    analyze_spectrum(a.data, f"{a.out}/fig_spectrum_prior.png", n_img=a.nimg)
    analyze_response(f"{a.out}/fig_freq_response.png")
    if a.baseline and a.ours:
        analyze_erf(f"{a.out}/fig_erf.png", a.baseline, a.ours, a.imgsz)
    else:
        print("[C] 未提供 --baseline/--ours，跳过 ERF")
