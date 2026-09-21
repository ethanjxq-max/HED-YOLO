# convert_gc10det.py — GC10-DET（archive：1~10 类文件夹 + lable/*.xml）→ YOLO 检测格式
# ============================================================================
# 输入（服务器上的 archive 目录，默认 ./archive）：
#   archive/1..10/*.jpg          2312 张 2048×1000 灰度图
#   archive/lable/*.xml          2294 个 VOC 标注（含 2 个无目标、1 个错标 "d"）
# 输出（默认 ./GC10-DET）：
#   GC10-DET/images/{train,val}/*.jpg
#   GC10-DET/labels/{train,val}/*.txt     （YOLO: cls cx cy w h，归一化）
#   GC10-DET/gc10det.yaml                 （data 配置，path 用绝对路径）
# 规则：
#   · 类别名归一化：10_yaozhed → 10_yaozhe；错标 "d" 丢弃；只保留 10 个标准类
#   · 2 个无目标 XML、6 张无标注图片：直接不纳入（不做背景图）
#   · 按"图片主类（所在文件夹）"分层切分，val 比例 0.2（保证每类都出现在 val）
# 用法：
#   cd /path/to/HED-YOLO
#   python convert_gc10det.py --src archive --dst GC10-DET
# ============================================================================
import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

# 10 个标准类别（顺序即 YOLO 类别 id）
CLASSES = ["1_chongkong", "2_hanfeng", "3_yueyawan", "4_shuiban", "5_youban",
           "6_siban", "7_yiwu", "8_yahen", "9_zhehen", "10_yaozhe"]
ALIAS = {"10_yaozhed": "10_yaozhe"}  # 拼写变体合并
DROP = {"d"}                          # 错标，丢弃


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="archive", help="解压后的 archive 目录")
    ap.add_argument("--dst", default="GC10-DET", help="输出数据集目录")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    assert src.is_dir(), f"找不到 {src}"
    lab_dir = src / "lable"
    assert lab_dir.is_dir(), f"找不到 {lab_dir}"

    cls_id = {c: i for i, c in enumerate(CLASSES)}
    rng = random.Random(a.seed)

    # 1) 建立 文件名 → (图片路径, 主类文件夹)
    img_by_name, folder_of = {}, {}
    for d in sorted(src.iterdir()):
        if not d.is_dir() or d.name == "lable":
            continue
        for p in d.glob("*.jpg"):
            img_by_name[p.name] = p
            folder_of[p.name] = d.name
    print(f"[扫描] 图片 {len(img_by_name)} 张，来自 {len(set(folder_of.values()))} 个类文件夹")

    # 2) 解析 XML → YOLO 行；按主类分层分组
    per_file = defaultdict(list)
    stats = Counter()
    skipped_nobox, dropped_labels = 0, Counter()
    for x in sorted(lab_dir.glob("*.xml")):
        r = ET.parse(x).getroot()
        fn = (r.findtext("filename") or "").strip()
        if fn not in img_by_name:
            stats["xml_无对应图片"] += 1
            continue
        size = r.find("size")
        W, H = int(size.findtext("width")), int(size.findtext("height"))
        lines = []
        for o in r.findall("object"):
            nm = (o.findtext("name") or "").strip()
            nm = ALIAS.get(nm, nm)
            if nm in DROP:
                dropped_labels[nm] += 1
                continue
            if nm not in cls_id:
                dropped_labels[nm] += 1
                continue
            b = o.find("bndbox")
            x1, y1 = float(b.findtext("xmin")), float(b.findtext("ymin"))
            x2, y2 = float(b.findtext("xmax")), float(b.findtext("ymax"))
            x1, x2 = max(0.0, min(x1, x2)), min(float(W), max(x1, x2))
            y1, y2 = max(0.0, min(y1, y2)), min(float(H), max(y1, y2))
            if x2 - x1 < 2 or y2 - y1 < 2:
                stats["退化框丢弃"] += 1
                continue
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            w, h = (x2 - x1) / W, (y2 - y1) / H
            lines.append(f"{cls_id[nm]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            stats[nm] += 1
        if not lines:
            skipped_nobox += 1
            continue
        per_file[fn] = lines

    print(f"[解析] 有标注图片 {len(per_file)} 张；无目标/空标注跳过 {skipped_nobox}；丢弃错标 {dict(dropped_labels)}")

    # 3) 分层切分（按主类文件夹）
    by_folder = defaultdict(list)
    for fn in per_file:
        by_folder[folder_of[fn]].append(fn)
    train, val = [], []
    for fo, fns in sorted(by_folder.items()):
        rng.shuffle(fns)
        k = int(round(len(fns) * a.val_ratio))
        val += fns[:k]
        train += fns[k:]
    print(f"[切分] train {len(train)} 张 / val {len(val)} 张（val 比例 {a.val_ratio}）")

    # 4) 落盘
    for split, files in (("train", train), ("val", val)):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        for fn in files:
            shutil.copy2(img_by_name[fn], dst / "images" / split / fn)
            (dst / "labels" / split / fn.replace(".jpg", ".txt")).write_text(
                "\n".join(per_file[fn]) + "\n", encoding="utf-8")

    yaml_txt = (f"# GC10-DET（钢带表面缺陷，10 类）— 由 convert_gc10det.py 生成\n"
                f"path: {dst.resolve().as_posix()}\n"
                f"train: images/train\nval: images/val\n"
                f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    (dst / "gc10det.yaml").write_text(yaml_txt, encoding="utf-8")

    # 5) 校验：每类实例数（train/val）
    cnt = {s: Counter() for s in ("train", "val")}
    for s, files in (("train", train), ("val", val)):
        for fn in files:
            for ln in per_file[fn]:
                cnt[s][CLASSES[int(ln.split()[0])]] += 1
    print("\n[校验] 类别                 train   val")
    for c in CLASSES:
        print(f"       {c:<20} {cnt['train'][c]:>5} {cnt['val'][c]:>5}")
    print(f"       {'合计':<18} {sum(cnt['train'].values()):>5} {sum(cnt['val'].values()):>5}")
    print(f"\n✅ 完成：{dst}/  （data 配置：{dst}/gc10det.yaml）")
    miss = [c for c in CLASSES if cnt["val"][c] == 0]
    if miss:
        print(f"⚠️ 以下类在 val 中为 0（需调 val-ratio 或忽略该类指标）：{miss}")


if __name__ == "__main__":
    main()
