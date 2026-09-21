# robustness_test.py — 噪声/退化鲁棒性实验（不训练，只做 val；单卡几分钟跑完）
# ============================================================================
# 做什么：把 val 图先缩放到推理输入尺寸（默认 640×640），再施加 6 类退化（每类 2~3 个强度），
#         然后用 baseline 与本文模型分别 val，得到"逐退化的 mAP"，看谁掉得少。
# 退化类型（贴近钢带产线成像工况）：
#   gauss  高斯噪声 σ=10/20/30（传感器噪声）
#   sp     椒盐噪声 p=0.5%/1%/2%（坏点、粉尘）
#   blur   高斯模糊 k=3/5/7（离焦）
#   motion 水平运动模糊 k=5/9/13（产线振动）
#   bright 亮度偏移 ±30%（光照波动）
#   jpeg   JPEG 压缩 q=30/15（传输/存储压缩）
# 输出：
#   robust/<退化名>/images/val/*.jpg + labels/val/（复制原标签）+ data.yaml
#   robustness_results.md / robustness_results.csv（表格，可直接进论文）
# 用法（服务器项目根目录）：
#   python robustness_test.py
#   python robustness_test.py --models base=runs/ab_base/ab_base/weights/best.pt ours=runs/ab_db_fem/ab_db_fem/weights/best.pt mrb=runs/r5_mrb/r5_mrb/weights/best.pt
#   python robustness_test.py --imgsz 640 --data ultralytics/cfg/datasets/neu_det.yaml
# 依赖：Pillow, numpy, ultralytics（服务器已有）
# ============================================================================
import argparse
import io
import json
import pathlib
import shutil

import numpy as np
import yaml
from PIL import Image, ImageFilter

CLEAN = "clean"
CORRUPTIONS = {
    "gauss10": ("gauss", 10), "gauss20": ("gauss", 20), "gauss30": ("gauss", 30),
    "sp05": ("sp", 0.005), "sp10": ("sp", 0.01), "sp20": ("sp", 0.02),
    "blur3": ("blur", 3), "blur5": ("blur", 5), "blur7": ("blur", 7),
    "motion5": ("motion", 5), "motion9": ("motion", 9), "motion13": ("motion", 13),
    "bright07": ("bright", 0.7), "bright13": ("bright", 1.3),
    "jpeg30": ("jpeg", 30), "jpeg15": ("jpeg", 15),
}


def corrupt(arr: np.ndarray, kind: str, lv, rng: np.random.Generator) -> np.ndarray:
    """arr: HWC uint8 RGB 0-255"""
    if kind == "gauss":
        return np.clip(arr.astype(np.float32) + rng.normal(0, lv, arr.shape), 0, 255).astype(np.uint8)
    if kind == "sp":  # 椒盐
        out = arr.copy()
        m = rng.random(arr.shape[:2])
        out[m < lv / 2] = 0
        out[m > 1 - lv / 2] = 255
        return out
    if kind == "bright":
        return np.clip(arr.astype(np.float32) * lv, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr)
    if kind == "blur":
        return np.array(im.filter(ImageFilter.GaussianBlur(lv)))
    if kind == "motion":
        # 水平方向运动模糊 = 水平盒式滤波（PIL 的 Kernel 只支持 3×3/5×5，故自行实现，支持任意奇数核）
        k = int(lv)
        pad = k // 2
        a = np.pad(arr.astype(np.float32), ((0, 0), (pad, pad), (0, 0)), mode="edge")
        c = np.cumsum(a, axis=1)
        c = np.concatenate([np.zeros((a.shape[0], 1, a.shape[2]), np.float32), c], axis=1)
        return np.clip((c[:, k:, :] - c[:, :-k, :]) / k, 0, 255).astype(np.uint8)
    if kind == "jpeg":
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=int(lv))
        buf.seek(0)
        return np.array(Image.open(buf).convert("RGB"))
    raise ValueError(kind)


def build_variants(data_yaml: str, root: pathlib.Path, imgsz: int, seed: int = 42):
    d = yaml.safe_load(open(data_yaml, encoding="utf-8"))
    p = pathlib.Path(d.get("path", "."))
    val_dir = (pathlib.Path(d["val"]) if str(d["val"]).startswith("/") else p / d["val"])
    lab_dir = pathlib.Path(str(val_dir).replace("images", "labels"))
    imgs = sorted([q for q in val_dir.glob("*") if q.suffix.lower() in (".jpg", ".png", ".jpeg")])
    print(f"[数据] val {len(imgs)} 张（{val_dir}） 标签 {lab_dir}")

    ext = ".jpg"
    tasks = [(CLEAN, None, None)] + [(k, v[0], v[1]) for k, v in CORRUPTIONS.items()]
    for name, kind, lv in tasks:
        out = root / name
        (out / "images" / "val").mkdir(parents=True, exist_ok=True)
        if not (out / "labels" / "val").exists():
            shutil.copytree(lab_dir, out / "labels" / "val")
        rng = np.random.default_rng(seed)
        for q in imgs:
            im = Image.open(q).convert("RGB")
            if im.size != (imgsz, imgsz):
                im = im.resize((imgsz, imgsz), Image.LANCZOS)
            a = np.array(im)
            if kind is not None:
                a = corrupt(a, kind, lv, rng)
            Image.fromarray(a).save(out / "images" / "val" / f"{q.stem}{ext}", quality=95)
        (out / "data.yaml").write_text(
            f"path: {out.resolve().as_posix()}\ntrain: images/val\nval: images/val\n"
            + yaml.safe_dump({"names": d["names"]}, allow_unicode=True, sort_keys=False),
            encoding="utf-8")
        print(f"  ✔ 生成 {name}")
    return [t[0] for t in tasks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ultralytics/cfg/datasets/neu_det.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--root", default="robust")
    ap.add_argument("--out", default="robustness_results.md")
    ap.add_argument("--skip-build", action="store_true", help="复用已生成的 robust/ 目录")
    ap.add_argument("--models", nargs="*", default=[
        "base=runs/ab_base/ab_base/weights/best.pt",
        "ours=runs/ab_db_fem/ab_db_fem/weights/best.pt",
    ], help="名字=权重路径，可多给几个")
    a = ap.parse_args()

    from ultralytics import YOLO

    root = pathlib.Path(a.root)
    names = [t[0] for t in [(CLEAN, None, None)] + [(k, v[0], v[1]) for k, v in CORRUPTIONS.items()]]
    if not a.skip_build:
        names = build_variants(a.data, root, a.imgsz)

    models = dict(m.split("=", 1) for m in a.models)
    res = {m: {} for m in models}
    for mname, ckpt in models.items():
        if not pathlib.Path(ckpt).exists():
            print(f"⚠️ 跳过 {mname}：找不到 {ckpt}")
            continue
        yolo = YOLO(ckpt)
        for v in names:
            dy = root / v / "data.yaml"
            mt = yolo.val(data=str(dy), split="val", imgsz=a.imgsz, verbose=False, plots=False)
            res[mname][v] = {"mAP50": float(mt.box.map50), "mAP50-95": float(mt.box.map)}
            print(f"  {mname:<6} {v:<10} mAP50 {mt.box.map50:.4f}  mAP50-95 {mt.box.map:.4f}")

    # ---------------- 输出表格（论文可直接用）
    keys = [k for k in res if res[k]]
    lines = ["| Corruption | Severity | " + " | ".join(f"{k} mAP50-95" for k in keys) + " | Δ (base−ours) |",
             "|---|---|" + "---|" * (len(keys) + 1)]
    rows = []
    for v in names:
        kind = "clean" if v == CLEAN else CORRUPTIONS[v][0]
        lv = "-" if v == CLEAN else str(CORRUPTIONS[v][1])
        vals = [res[k][v]["mAP50-95"] for k in keys]
        gap = (vals[0] - vals[1]) if len(vals) > 1 else float("nan")
        lines.append(f"| {kind} | {lv} | " + " | ".join(f"{x:.4f}" for x in vals) + f" | {gap:+.4f} |")
        rows.append({"variant": v, **{k: res[k][v] for k in keys}})
    # 相对衰减（以 clean 为基准）
    lines += ["", "| Corruption | " + " | ".join(f"{k} 相对衰减 %" for k in keys) + " |", "|---|" + "---|" * len(keys)]
    for v in names[1:]:
        dec = [100 * (res[k][CLEAN]["mAP50-95"] - res[k][v]["mAP50-95"]) / res[k][CLEAN]["mAP50-95"] for k in keys]
        lines.append(f"| {CORRUPTIONS[v][0]} {CORRUPTIONS[v][1]} | " + " | ".join(f"{d:.1f}" for d in dec) + " |")
    txt = "\n".join(lines)
    pathlib.Path(a.out).write_text(txt + "\n", encoding="utf-8")
    pathlib.Path(a.out).with_suffix(".csv").write_text(
        "variant," + ",".join(keys) + "\n" + "\n".join(
            r["variant"] + "," + ",".join(f"{r[k]['mAP50-95']:.4f}" for k in keys) for r in rows), encoding="utf-8")
    print("\n" + txt)
    print(f"\n✅ 已保存 {a.out} 与 {pathlib.Path(a.out).with_suffix('.csv')}")


if __name__ == "__main__":
    main()
