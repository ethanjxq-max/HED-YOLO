# make_paper_figures.py — 一键生成论文用图（数据已内置，本地/服务器都能跑）
# ============================================================================
# 生成（默认输出到 ./figures/，300 dpi，英文标注可直接进论文）：
#   fig1_ablation.png      消融链柱状图（NEU-DET 与 GC10-DET 双面板：baseline→+DB→+FEM→最终）
#   fig2_per_class.png     逐类对比（NEU-DET，baseline vs 最终，mAP50 与 mAP50-95 双面板）
#   fig3_acc_efficiency.png 精度–复杂度散点（GFLOPs vs mAP50-95，标注端到端延迟；含本文模型）
#   fig4_cpu_onnx.png      CPU-ONNX 延迟对比（含"NMS-free 输出 300 框"标注）
#   fig5_head_paths.png    检测头路径消融（NMS-free vs 含 NMS：精度与延迟）
# 依赖：pip install matplotlib（>=3.5）
# 用法：python make_paper_figures.py            # 输出到 ./figures/
#       python make_paper_figures.py --out myfig
# 说明：图里的数字与《论文写作素材_总表与结论汇总.md》第十一~十三节完全一致；
#       若后续实验有更新，改本文件顶部的数据字典即可（每个字典都写了来源）。
# ============================================================================
import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- 数据（唯一事实来源）
# 主消融：yaml = <name>.yaml；数字来自同协议 250ep/seed42 实测
ABL = {
    "NEU-DET": {
        "rows": ["Baseline", "+C3k2-DB", "+FEM", "+DB+FEM\n(ours)"],
        "mAP50": [0.711, 0.727, 0.727, 0.751],
        "mAP50-95": [0.371, 0.373, 0.384, 0.388],
    },
    "GC10-DET": {
        "rows": ["Baseline", "+C3k2-DB", "+FEM", "+DB+FEM\n(ours)"],
        "mAP50": [0.633, 0.671, 0.673, 0.674],
        "mAP50-95": [0.333, 0.337, 0.352, 0.348],
    },
}
# 逐类（NEU-DET，六类顺序固定）
PER_CLASS = {
    "classes": ["crazing", "inclusion", "patches", "pitted", "rolled", "scratches"],
    "base_50": [0.534, 0.809, 0.891, 0.805, 0.480, 0.746],
    "ours_50": [0.554, 0.812, 0.912, 0.789, 0.574, 0.864],
    "base_95": [0.206, 0.430, 0.590, 0.453, 0.209, 0.340],
    "ours_95": [0.203, 0.435, 0.578, 0.429, 0.270, 0.414],
}
# 精度–复杂度（GFLOPs, mAP50-95, GPU 端到端 ms, Params(M)）—— 来源：第十二节 Table 4
SCATTER = {
    "YOLOv8s": (14.41, 0.377, 2.54, 11.17),
    "YOLOv9s": (13.78, 0.397, 2.37, 7.32),
    "YOLOv10s": (12.55, 0.352, 3.96, 8.13),
    "YOLOv11s": (10.86, 0.386, 2.49, 9.46),
    "YOLO26s": (11.26, 0.371, 2.84, 9.95),
    "Ours": (12.50, 0.388, 2.73, 12.89),
}
# CPU-ONNX 延迟（ms / FPS）—— 来源：第十三节 Table 7
CPU = {  # 延迟 ms / FPS（按延迟升序）
    "YOLO26s": (76.0, 13.15),          # 基座，全部模型中最快
    "MRB-folded": (83.4, 12.00),
    "YOLOv10s": (84.9, 11.77),
    "Ours": (87.2, 11.47),             # 本文模型：仅比 v10s 高 2.7%
    "YOLOv8s": (95.2, 10.51),
    "YOLOv11s": (109.1, 9.17),
    "YOLOv9s": (139.6, 7.16),
}
# 头部路径消融（同一基座：NMS-free vs 含 NMS）
HEAD_PATHS = {
    "labels": ["YOLO26s\n(NMS-free)", "YOLO26s\n(w/ NMS)", "Ours\n(NMS-free)", "Ours\n(w/ NMS)"],
    "mAP50-95": [0.371, 0.402, 0.388, 0.394],
    "ms": [2.84, 2.64, 2.73, 2.94],
}

C_OURS, C_ALT, C_GREY = "#c0392b", "#2c7fb8", "#95a5a6"


def save(fig, out, name):
    p = pathlib.Path(out) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print("saved", p)


def fig_ablation(out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (ds, d) in zip(axes, ABL.items()):
        x = np.arange(len(d["rows"]))
        w = 0.36
        ax.bar(x - w / 2, d["mAP50"], w, label="mAP@0.5", color=C_GREY)
        ax.bar(x + w / 2, d["mAP50-95"], w, label="mAP@0.5:0.95", color=C_OURS, alpha=0.85)
        for xi, (a, b) in enumerate(zip(d["mAP50"], d["mAP50-95"])):
            ax.text(xi - w / 2, a + 0.006, f"{a:.3f}", ha="center", fontsize=7)
            ax.text(xi + w / 2, b + 0.006, f"{b:.3f}", ha="center", fontsize=7, color=C_OURS)
        ax.set_xticks(x)
        ax.set_xticklabels(d["rows"], fontsize=8)
        ax.set_ylim(0, 0.85)
        ax.set_ylabel("Accuracy")
        ax.set_title(ds, fontsize=10)
        ax.grid(axis="y", ls=":", alpha=0.5)
        ax.legend(fontsize=8, frameon=False)
    save(fig, out, "fig1_ablation.png")


def fig_per_class(out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(len(PER_CLASS["classes"]))
    w = 0.36
    for ax, key in zip(axes, ("50", "95")):
        b, o = PER_CLASS[f"base_{key}"], PER_CLASS[f"ours_{key}"]
        ax.bar(x - w / 2, b, w, label="Baseline", color=C_GREY)
        ax.bar(x + w / 2, o, w, label="Ours", color=C_OURS, alpha=0.85)
        for xi, (bb, oo) in enumerate(zip(b, o)):
            d = oo - bb
            if abs(d) >= 0.02:  # 只标出明显变化，避免噪声视觉化
                ax.text(xi + w / 2, oo + 0.012, f"{d:+.3f}", ha="center", fontsize=7,
                        color=C_OURS if d > 0 else "black")
        ax.set_xticks(x)
        ax.set_xticklabels(PER_CLASS["classes"], rotation=18, fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("mAP@0.5" if key == "50" else "mAP@0.5:0.95")
        ax.set_title(f"Per-class performance ({'mAP@0.5' if key == '50' else 'mAP@0.5:0.95'})", fontsize=10)
        ax.grid(axis="y", ls=":", alpha=0.5)
        ax.legend(fontsize=8, frameon=False)
    save(fig, out, "fig2_per_class.png")


def fig_acc_efficiency(out):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for name, (gf, m, ms, pm) in SCATTER.items():
        ours = name == "Ours"
        ax.scatter(gf, m, s=170 if ours else 90, color=C_OURS if ours else C_ALT,
                   marker="*" if ours else "o", zorder=3, edgecolor="white")
        ax.annotate(f"{name}\n{pm:.1f}M, {ms:.2f} ms", (gf, m), textcoords="offset points",
                    xytext=(6, -12 if not ours else 6), fontsize=8,
                    color=C_OURS if ours else "#333333")
    ax.set_xlabel("GFLOPs @640")
    ax.set_ylabel("mAP@0.5:0.95 (NEU-DET)")
    ax.set_title("Accuracy–complexity trade-off", fontsize=10)
    ax.grid(ls=":", alpha=0.5)
    ax.set_xlim(min(g for g, *_ in SCATTER.values()) - 1, max(g for g, *_ in SCATTER.values()) + 1.4)
    save(fig, out, "fig3_acc_efficiency.png")


def fig_cpu(out):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    names = list(CPU)
    ms = [CPU[n][0] for n in names]
    colors = [C_OURS if n == "Ours" else (C_GREY if n == "YOLO26s" else C_ALT) for n in names]
    bars = ax.bar(names, ms, color=colors, alpha=0.9)
    for b, n in zip(bars, names):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{CPU[n][0]:.1f}",
                ha="center", fontsize=8)
    ax.set_ylabel("CPU latency (ms, ONNX Runtime)")
    ax.set_title("CPU inference latency (FP32, batch=1, 640, 8 threads)", fontsize=10)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.tick_params(axis="x", labelsize=8)
    ax.axhline(CPU["YOLO26s"][0], color=C_GREY, ls="--", lw=0.8)
    ax.annotate("NMS-free graphs output 300 boxes (ready to deploy);\n"
                "v8/v9/v11 output 8400 raw predictions (need NMS, not counted here)",
                xy=(0.02, 0.92), xycoords="axes fraction", fontsize=7.5, va="top")
    save(fig, out, "fig4_cpu_onnx.png")


def fig_head_paths(out):
    fig, ax1 = plt.subplots(figsize=(6.6, 3.8))
    x = np.arange(len(HEAD_PATHS["labels"]))
    b = ax1.bar(x, HEAD_PATHS["mAP50-95"], 0.5,
                color=[C_GREY, C_GREY, C_OURS, C_OURS], alpha=0.85)
    for xi, v in zip(x, HEAD_PATHS["mAP50-95"]):
        ax1.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    ax1.set_ylabel("mAP@0.5:0.95")
    ax1.set_ylim(0, 0.48)
    ax1.set_xticks(x)
    ax1.set_xticklabels(HEAD_PATHS["labels"], fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(x, HEAD_PATHS["ms"], "o--", color="#e67e22", lw=1.2, ms=5, label="GPU end-to-end (ms)")
    ax2.set_ylabel("GPU end-to-end latency (ms)", color="#e67e22")
    ax2.tick_params(axis="y", colors="#e67e22")
    ax1.set_title("Detection-head path: NMS-free vs with-NMS", fontsize=10)
    ax1.grid(axis="y", ls=":", alpha=0.4)
    save(fig, out, "fig5_head_paths.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    fig_ablation(a.out)
    fig_per_class(a.out)
    fig_acc_efficiency(a.out)
    fig_cpu(a.out)
    fig_head_paths(a.out)
    print("\n全部完成 → ", pathlib.Path(a.out).resolve())
