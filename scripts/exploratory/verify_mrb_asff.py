# verify_mrb_asff.py — MRB / ASFF 上线前本地验证（CPU 可跑，无需 GPU）
# 用法：cd 项目根目录 && python verify_mrb_asff.py
import copy

import torch

from ultralytics.nn.Convmodules.repblock import RepConvDiverse, rep_fuse_model
from ultralytics.nn.tasks import DetectionModel

CFG = "ultralytics/cfg/models/26/"
MODELS = [
    ("yolo26s", "基线"),
    ("yolo26s_mrb", "基线 + MRB"),
    ("yolo26s_db_fem_11_13", "E 基座（DB+FEM@11,13）"),
    ("yolo26s_db_fem_11_13_mrb", "E + MRB"),
    ("yolo26s_db_fem_11_13_asff", "E + ASFF"),
]


def build(name):
    return DetectionModel(f"{CFG}{name}.yaml", ch=3, nc=6, verbose=False)


def _flat(y):
    """模型输出可能是 tuple/dict 嵌套，递归取出所有张量用于比对"""
    out = []

    def rec(o):
        if isinstance(o, torch.Tensor):
            out.append(o)
        elif isinstance(o, dict):
            rec(list(o.values()))
        elif isinstance(o, (list, tuple)):
            for v in o:
                rec(v)

    rec(y)
    return out


print("=" * 78)
print("一、建模 + 参数量（np 为 ultralytics 统计的模块参数量）")
print("=" * 78)
models = {}
for name, note in MODELS:
    m = build(name)
    n_rep = sum(1 for x in m.modules() if isinstance(x, RepConvDiverse))
    p = sum(x.numel() for x in m.parameters())
    models[name] = m
    print(f"{name:<28} {note:<24} 训练参数 {p/1e6:8.4f} M   层数 {len(m.model):3d}   MRB 块 {n_rep}")

print()
print("=" * 78)
print("二、MRB 折叠（推理态）：参数量与逐位等价性")
print("=" * 78)
for name in ("yolo26s_mrb", "yolo26s_db_fem_11_13_mrb"):
    md = copy.deepcopy(models[name]).double().eval()
    torch.manual_seed(0)
    x = torch.randn(1, 3, 128, 128, dtype=torch.float64)
    with torch.no_grad():
        y1 = _flat(md(x))
    n, p_before, p_after = rep_fuse_model(md)
    with torch.no_grad():
        y2 = _flat(md(x))
    err = max((a - b).abs().max().item() for a, b in zip(y1, y2))
    ref_p = sum(p.numel() for p in models["yolo26s"].parameters())
    print(f"{name:<28} 折叠 {n} 处：{p_before/1e6:.4f}M → {p_after/1e6:.4f}M   "
          f"（基线 {ref_p/1e6:.4f}M，相对基线 {(p_after/ref_p-1)*100:+.2f}%）")
    print(f"{'':<28} 折叠前/后输出最大差（float64）: {err:.2e}")
    assert err < 1e-9, "折叠不等价！"

print()
print("=" * 78)
print("三、前向形状 + 梯度连通性（640×640，与训练一致）")
print("=" * 78)
for name, _ in MODELS:
    m = models[name]
    m.train()
    torch.manual_seed(0)
    x = torch.randn(1, 3, 640, 640)
    y = m(x)
    ts = _flat(y)
    shapes = [tuple(t.shape) for t in ts]
    loss = sum(t.float().pow(2).mean() for t in ts)
    loss.backward()
    nograd = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    print(f"{name:<28} 输出张量 {shapes}  无梯度参数张量数 {len(nograd)}")
    assert not nograd, f"{name} 存在无梯度参数：{nograd[:5]}"

print()
print("✅ verify_mrb_asff.py 全部通过：5 个模型均可建模、可反向、MRB 折叠精确等价。")
