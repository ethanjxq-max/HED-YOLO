# make_qualitative_grid.py — 把 baseline / ours 的检测可视化拼成论文用对照图
# ============================================================================
# 输入：两个目录，文件名需一一对应（都是同一批 val 图跑出来的）
#   --base figures_qual/base     ← runs/vis_ab_base/val/*.jpg
#   --ours figures_qual/ours     ← runs/vis_ab_db_fem/val/*.jpg
# 输出：figures/fig_qualitative.png（上排 Baseline，下排 Ours；列标题显示文件名前缀=类别）
# 用法：
#   python make_qualitative_grid.py --base figures_qual/base --ours figures_qual/ours \
#       --out figures/fig_qualitative.png --n 3
#   # 指定具体文件（推荐：挑 rolled / scratches / crazing 各一张）：
#   python make_qualitative_grid.py --base ... --ours ... --names rolled-in_scale_241.jpg scratches_100.jpg crazing_120.jpg
# 依赖：pip install pillow
# ============================================================================
import argparse
import pathlib

from PIL import Image, ImageDraw, ImageFont

EXTS = (".jpg", ".jpeg", ".png")


def load_common(base_dir, ours_dir, names, n):
    base_dir, ours_dir = pathlib.Path(base_dir), pathlib.Path(ours_dir)
    if names:
        pairs = [(nm, base_dir / nm, ours_dir / nm) for nm in names]
    else:
        common = sorted(p.name for p in base_dir.glob("*") if p.suffix.lower() in EXTS
                        and (ours_dir / p.name).exists())
        if not common:
            raise SystemExit(f"两个目录里没有同名图片：{base_dir} vs {ours_dir}")
        pairs = [(nm, base_dir / nm, ours_dir / nm) for nm in common[:n]]
    missing = [nm for nm, b, o in pairs if not (b.exists() and o.exists())]
    pairs = [(nm, b, o) for nm, b, o in pairs if b.exists() and o.exists()]
    if missing:
        print("⚠️ 跳过不存在的文件：", missing)
    print(f"使用 {len(pairs)} 对：", [nm for nm, _, _ in pairs])
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--out", default="figures/fig_qualitative.png")
    ap.add_argument("--n", type=int, default=3, help="不指定 --names 时取前 n 张")
    ap.add_argument("--names", nargs="*", default=None, help="显式指定文件名（两边同名）")
    ap.add_argument("--colw", type=int, default=520, help="每列缩放后的宽度")
    ap.add_argument("--label", nargs=2, default=["Baseline", "Ours"])
    a = ap.parse_args()

    pairs = load_common(a.base, a.ours, a.names, a.n)
    if not pairs:
        raise SystemExit("没有可用图片对")

    def prep(p):
        im = Image.open(p).convert("RGB")
        h = int(im.height * a.colw / im.width)
        return im.resize((a.colw, h), Image.LANCZOS)

    cols = [prep(p) for _, _, p in pairs]
    rowh = max(c.height for c in cols)
    pad, head = 8, 34
    W = pad + len(cols) * (a.colw + pad)
    H = head + 2 * (rowh + head) + pad
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        fsmall = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = fsmall = ImageFont.load_default()

    def put(im, col, row, label):
        x = pad + col * (a.colw + pad)
        y = head + row * (rowh + head)
        canvas.paste(im, (x, y))
        draw.text((x + 4, y - head + 6), f"{label}: {pairs[col][0]}", fill="black", font=fsmall)

    for row, key in enumerate(("base", "ours")):
        for col, (_, b, o) in enumerate(pairs):
            put(prep(b if key == "base" else o), col, row, a.label[row])

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, dpi=(300, 300))
    print("saved", out.resolve())


if __name__ == "__main__":
    main()
