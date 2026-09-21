# make_robustness_curve.py — 由 robustness_results.csv 画鲁棒性衰减曲线（论文 Fig.7）
# ============================================================================
# 输出：figures/fig7_robustness.png（2×2 子图：高斯噪声 / 椒盐噪声 / 模糊+运动模糊 / 压缩+光照）
#       每个子图两条线：YOLO26s (baseline) 与 Ours；纵轴 mAP@0.5:0.95
# 用法：python make_robustness_curve.py            # 默认读 ./robustness_results.csv
#       python make_robustness_curve.py --csv robustness_results.csv --out figures/fig7_robustness.png
# 说明：x 轴用"退化强度等级"（同类内按严重度递增），刻度标签写明实际参数值（σ / p% / k / q / 亮度倍数）
# ============================================================================
import argparse
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (子图标题, [(csv 列名, x 轴刻度标签), ...])
PANELS = [
    ("Gaussian noise", [("gauss10", "σ=10"), ("gauss20", "σ=20"), ("gauss30", "σ=30")]),
    ("Salt-and-pepper noise", [("sp05", "0.5%"), ("sp10", "1%"), ("sp20", "2%")]),
    ("Blur", [("blur3", "k=3"), ("blur5", "k=5"), ("blur7", "k=7"),
              ("motion5", "m=5"), ("motion9", "m=9"), ("motion13", "m=13")]),
    ("Compression / illumination", [("jpeg30", "q=30"), ("jpeg15", "q=15"),
                                    ("bright07", "×0.7"), ("bright13", "×1.3")]),
]
SERIES = [("base", "YOLO26s (baseline)", "#7f8c8d", "o", "-"),
          ("ours", "Ours (DB+FEM+SAFR)", "#c0392b", "s", "-")]
CLEAN_ROW = "clean"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="robustness_results.csv")
    ap.add_argument("--out", default="figures/fig7_robustness.png")
    a = ap.parse_args()

    rows = {r["variant"]: r for r in csv.DictReader(open(a.csv, encoding="utf-8"))}
    clean = {k: float(rows[CLEAN_ROW][k]) for k, *_ in SERIES}

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    for ax, (title, items) in zip(axes.ravel(), PANELS):
        xs = range(len(items))
        for key, label, color, mk, ls in SERIES:
            ys = [float(rows[c][key]) for c, _ in items]
            ax.plot(xs, ys, marker=mk, color=color, lw=1.6, ms=5, ls=ls, label=label)
            if key in clean:  # 清洁基线参考线（虚线）
                ax.axhline(clean[key], color=color, lw=0.8, ls=":", alpha=0.7)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([t for _, t in items], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("mAP@0.5:0.95")
        ax.grid(ls=":", alpha=0.45)
        ax.set_ylim(bottom=0)
    axes[0][0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Robustness under image degradations (NEU-DET val, dotted lines = clean reference)",
                 fontsize=11)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=300)
    print("saved", out.resolve())

    # 顺手打印"优势随退化增强"的证据行（论文正文可直接引用）
    print("\n各退化下 ours − base 的 mAP50-95 差值（正=ours 更好）：")
    for _, items in PANELS:
        for c, lab in items:
            d = float(rows[c]["ours"]) - float(rows[c]["base"])
            print(f"  {c:<10}{lab:>7}  {d:+.4f}")


if __name__ == "__main__":
    main()
