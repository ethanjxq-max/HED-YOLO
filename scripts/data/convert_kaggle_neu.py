# -*- coding: utf-8 -*-
"""
Kaggle 版 NEU-DET → YOLO 格式转换（适配 ultralytics-main 教程第2章目标结构）

源结构（Kaggle 下载，已按官方 1440/360 划分，图片按类别分子目录）:
  archive/NEU-DET/
  ├── train/
  │   ├── annotations/*.xml          ← 标注平铺
  │   └── images/{crazing,inclusion,patches,pitted_surface,rolled-in_scale,scratches}/
  └── validation/
      ├── annotations/*.xml
      └── images/{6类}/

输出结构:
  <out>/
  ├── images/train/   ← 1439 张
  ├── images/val/     ← 360 张
  ├── labels/train/
  └── labels/val/
  <yaml>              ← 生成的数据配置文件

用法:
  python convert_kaggle_neu.py --src archive/NEU-DET --out datasets/NEU-DET --yaml datasets/neu_det.yaml
注意: Kaggle 打包时 crazing_240 的图片与标注被拆到两个集（图在train、xml在validation），
      本脚本自动跳过无法配对的样本，不影响其余 1799 张的完整性。
"""
import os
import shutil
import xml.etree.ElementTree as ET

# ===== 配置（默认值为占位路径，请用命令行参数指定实际位置）=====
SRC = "archive/NEU-DET"              # Kaggle 解压后的 NEU-DET 根目录（含 train/ 与 validation/）
OUT = "datasets/NEU-DET"             # 输出数据集目录
YAML_PATH = "datasets/neu_det.yaml"  # 生成的数据配置文件路径

# 六个类别按字母序编号（与教程 convert_neu.py / neu_det.yaml 完全一致）
CLASSES = ["crazing", "inclusion", "patches",
           "pitted_surface", "rolled-in_scale", "scratches"]
CLASS2ID = {name: i for i, name in enumerate(CLASSES)}

EXT = ".jpg"


def collect_images(img_root):
    """把 6 个类别子目录的图片收集成 {basename: 绝对路径} 索引"""
    index = {}
    for cls_dir in os.listdir(img_root):
        cls_path = os.path.join(img_root, cls_dir)
        if not os.path.isdir(cls_path):
            continue
        for fn in os.listdir(cls_path):
            name, ext = os.path.splitext(fn)
            if ext.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            index[name] = os.path.join(cls_path, fn)
    return index


def convert_xml(xml_path, img_w, img_h):
    """解析一个 XML，返回 YOLO 格式标注行列表（与教程 convert_one 一致）"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASS2ID:
            continue
        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)
        # 防御：坐标越界/非法的框直接跳过（NEU-DET 偶发宽高为0的框）
        if xmax <= xmin or ymax <= ymin:
            continue
        x_c = (xmin + xmax) / 2 / img_w
        y_c = (ymin + ymax) / 2 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        lines.append(f"{CLASS2ID[name]} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
    return lines


def convert_split(src_split, out_split):
    """转换一个划分（train 或 validation）"""
    xml_dir = os.path.join(SRC, src_split, "annotations")
    img_root = os.path.join(SRC, src_split, "images")
    out_img = os.path.join(OUT, "images", out_split)
    out_lbl = os.path.join(OUT, "labels", out_split)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    img_index = collect_images(img_root)
    n_img = n_lbl = n_skip = n_empty = 0
    for fn in sorted(os.listdir(xml_dir)):
        name, ext = os.path.splitext(fn)
        if ext.lower() != ".xml":
            continue
        # 1) 找对应图片（xml 在平铺目录，图在类别子目录）
        img_src = img_index.get(name)
        if img_src is None:
            print(f"  [跳过] 有标注无图片: {name}")
            n_skip += 1
            continue
        # 2) 解析 xml → txt
        xml_path = os.path.join(xml_dir, fn)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)
        lines = convert_xml(xml_path, img_w, img_h)
        if not lines:
            print(f"  [跳过] 无有效目标框: {name}")
            n_empty += 1
            continue
        # 3) 复制图片 + 写 txt
        shutil.copy2(img_src, os.path.join(out_img, name + EXT))
        with open(os.path.join(out_lbl, name + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        n_img += 1
        n_lbl += 1
    print(f"[{src_split} -> {out_split}] 图片 {n_img} 张 / 标注 {n_lbl} 个"
          f"（跳过 {n_skip + n_empty}: 无配对 {n_skip}, 空框 {n_empty}）")
    return n_img, n_lbl


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Kaggle 版 NEU-DET → YOLO 格式转换")
    ap.add_argument("--src", default=SRC, help="Kaggle 解压后的 NEU-DET 根目录（含 train/ 与 validation/）")
    ap.add_argument("--out", default=OUT, help="输出数据集目录")
    ap.add_argument("--yaml", default=YAML_PATH, help="生成的数据配置文件路径")
    _a = ap.parse_args()
    SRC, OUT, YAML_PATH = _a.src, _a.out, _a.yaml

    if os.path.exists(OUT):
        # 输出目录已存在则先确认是空的（避免覆盖已有转换结果）
        remain = sum(len(fs) for _, _, fs in os.walk(OUT))
        if remain:
            print(f"警告: {OUT} 非空（{remain} 个文件），已中止，请检查后手动删除再运行。")
            raise SystemExit(1)
    print("开始转换 Kaggle 版 NEU-DET → YOLO 格式 ...")
    n_tr, _ = convert_split("train", "train")
    n_va, _ = convert_split("validation", "val")
    print(f"完成: train {n_tr} 张 / val {n_va} 张（共 {n_tr + n_va} 张）")

    # 生成数据配置文件（与教程 2.5 节一致）
    yaml_text = f"""# NEU-DET 数据配置
path: {OUT.replace(chr(92), '/')}   # 数据集根目录（绝对路径，注意用正斜杠）
train: images/train         # 训练图片目录（相对 path）
val: images/val             # 验证图片目录

# 六个类别，顺序必须与 convert 脚本中 CLASSES 一致！
names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
"""
    with open(YAML_PATH, "w") as f:
        f.write(yaml_text)
    print(f"已生成数据配置: {YAML_PATH}")
