# map_pretrained.py — 把官方 COCO 预训练权重"按语义"迁移到本项目的自定义结构
# ============================================================================
# 为什么需要它：
#   ultralytics 自带的迁移（model=xxx.yaml pretrained=yolo26s.pt）只按"层号 + 参数名 + 形状"
#   取交集。我们的 yaml 有三处偏差，导致它只能搬到约 14% 的权重、且有少量错位：
#     ① DB/FEM/DADC 插入使后续层号整体偏移（如官方 SPPF@9 → 我们的 SPPF@10）；
#     ② C3k2 → C3k2_DB 后，官方 Bottleneck 的 cv1/cv2 变成了 branch_a.cv1/cv2；
#     ③ DADC 的卷积参数名去掉了 .conv.（DeformConv2d.weight 而非 Conv.conv.weight）。
#   本脚本用"模块族对齐 + 参数名归一化"处理这三件事，把可迁移比例从 ~14% 提升到 ~80%。
#
# 用法（在工程根目录，即含 ultralytics/ 的那一层）：
#   python map_pretrained.py --weights yolo26s.pt \
#       --cfg ultralytics/cfg/models/26/yolo26s_db_fem_dadc_p3p4p5.yaml \
#       --out weights/pretrain_dadc.pt
# 然后用产出的权重训练（它的结构就是你的自定义结构，不需要再写 pretrained=）：
#   yolo detect train data=ultralytics/cfg/datasets/neu_det.yaml \
#     model=weights/pretrain_dadc.pt epochs=250 imgsz=640 batch=24 device=0 workers=4 seed=42 scale=0.5 \
#     project=.../runs/neu_dadc_pretrain name=dadc_pretrain
# ============================================================================
import argparse
import re
from datetime import datetime
from pathlib import Path

import torch

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import __version__


def fam(name: str) -> str:
    """模块族名：C3k2_DB_DADC → C3k2（用于跨结构的层对齐）。"""
    return re.sub(r"_(DB_DADC|DB|DADC)$", "", name)


def align_layers(src_model, dst_model):
    """按模块族序列做层对齐，返回 [(src_idx, dst_idx), ...]（处理插入/删除造成的偏移）。"""
    import difflib

    a = [fam(type(m).__name__) for m in src_model.model]
    b = [fam(type(m).__name__) for m in dst_model.model]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    pairs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":  # 只有族名一致的层才配对；insert/delete 自然跳过
            pairs.extend((i1 + k, j1 + k) for k in range(i2 - i1))
    return pairs, a, b


def candidate_keys(src_key: str, i_src: int, i_dst: int, dst_typename: str):
    """给一个源参数名，产出所有可能的迁移目标名（按优先级）。"""
    prefix = f"model.{i_src}."
    if not src_key.startswith(prefix):
        return []
    suffix = src_key[len(prefix) :]
    base = f"model.{i_dst}."
    cands = [base + suffix]

    # 规则②：C3k2 → C3k2_DB(_DADC)，内部 Bottleneck 变成 branch_a
    #  官方键：m.0.cv1.conv.weight / m.0.0.cv1.conv.weight（attn 分支下多一层）
    #  目标键：m.0.branch_a.cv1.conv.weight / m.0.0.branch_a.cv1.conv.weight
    if "DB" in dst_typename and re.search(r"(?:^|\.)m\.\d+\.", suffix):
        db_suffix = re.sub(r"\.(cv\d)\.", r".branch_a.\1.", suffix, count=1)
        cands.append(base + db_suffix)
        # 规则③：DADC 的 DeformConv2d 参数名去掉 .conv.
        cands.append(base + db_suffix.replace(".conv.", ".", 1))
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="官方预训练权重，如 yolo26s.pt")
    ap.add_argument("--cfg", required=True, help="目标自定义结构 yaml")
    ap.add_argument("--out", required=True, help="输出权重路径，如 weights/pretrain_xxx.pt")
    ap.add_argument("--nc", type=int, default=6, help="类别数（NEU-DET = 6）")
    ap.add_argument("--ch", type=int, default=3)
    args = ap.parse_args()

    # ---------- 1) 读源权重 ----------
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    src_model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    src_sd = src_model.float().state_dict()
    print(f"[源] {args.weights}: {type(src_model).__name__}, {len(src_sd)} 个参数张量")

    # ---------- 2) 建目标结构 ----------
    dst_model = DetectionModel(args.cfg, ch=args.ch, nc=args.nc, verbose=False)
    dst_sd = dst_model.state_dict()
    print(f"[目标] {args.cfg}: {len(dst_model.model)} 层, {len(dst_sd)} 个参数张量")

    # ---------- 3) 层对齐 + 逐张量迁移 ----------
    pairs, _, dst_fams = align_layers(src_model, dst_model)
    src_types = [type(m).__name__ for m in src_model.model]
    dst_types = [type(m).__name__ for m in dst_model.model]
    print(f"[对齐] 官方 {len(src_types)} 层 ↔ 目标 {len(dst_types)} 层，族名配对成功 {len(pairs)} 层")

    new_sd = dict(dst_sd)
    moved, skipped, misplaced = 0, [], []
    for i_src, i_dst in pairs:
        for k, v in src_sd.items():
            if not k.startswith(f"model.{i_src}."):
                continue
            done = False
            for k2 in candidate_keys(k, i_src, i_dst, dst_types[i_dst]):
                if k2 in new_sd and new_sd[k2].shape == v.shape:
                    new_sd[k2] = v.clone()
                    moved += 1
                    done = True
                    # 语义校验：目标层与源层的族名必须一致
                    if fam(src_types[i_src]) != dst_fams[i_dst]:
                        misplaced.append((i_src, src_types[i_src], i_dst, dst_types[i_dst]))
                    break
            if not done:
                skipped.append(k)

    dst_model.load_state_dict(new_sd, strict=False)
    total = len(dst_sd)
    print(f"[结果] 迁移 {moved}/{total} 项（{moved / total * 100:.1f}%）；未迁移 {len(skipped)} 项")
    if misplaced:
        print(f"  ⚠️ 语义错位 {len(misplaced)} 项（需人工确认）：{misplaced[:5]}")
    else:
        print("  ✅ 无语义错位（每个被迁移的张量都落在同族层上）")
    # 未迁移项按层归类（便于判断哪些是合理的新增层）
    by_layer = {}
    for k in skipped:
        m = re.match(r"model\.(\d+)\.", k)
        if m:
            by_layer.setdefault(int(m.group(1)), []).append(k)
    show = sorted(by_layer.items())[:12]
    print("  未迁移项按层（官方层号: 数量/示例）：")
    for i, ks in show:
        print(f"    官方层 {i:>2} [{src_types[i]:<10s}] : {len(ks):>3} 项  e.g. {ks[0]}")

    # ---------- 4) 保存为新结构的权重 ----------
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": dst_model, "date": datetime.now().isoformat(), "version": __version__}, args.out)
    print(f"\n[输出] 已写出 {args.out}")
    print("训练命令示例：")
    print(
        f"  yolo detect train data=ultralytics/cfg/datasets/neu_det.yaml model={args.out} "
        f"epochs=250 imgsz=640 batch=24 device=0 workers=4 seed=42 scale=0.5 "
        f"project=runs/neu_pretrain name=$(basename {args.out} .pt)"
    )


if __name__ == "__main__":
    main()
