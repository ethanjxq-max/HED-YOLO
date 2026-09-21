# verify_mgf.py — 本地离线校验：新结构 yaml 的层号引用 / 通道 / 参数量 / FLOPs / 前向形状
# 用法：python verify_mgf.py           （在 yolo26_project/yolo26_project 目录下运行，CPU 即可）
# 作用：不占服务器 GPU，先在本地把结构问题（层号引用错、通道不匹配、分辨率错）全部暴露出来。
import sys
import torch

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops

CFG = "ultralytics/cfg/models/26/"
MODELS = [
    ("E 基座（现有最好，0.743/0.388）", "yolo26s_db_fem_11_13.yaml"),
    ("MGF-P4（新增）", "yolo26s_db_fem_mgf_p4.yaml"),
    ("MGF-P4+P5（新增）", "yolo26s_db_fem_mgf_p4p5.yaml"),
    ("SBE-P4 / FEM@P4（备选）", "yolo26s_db_fem_sbe_p4.yaml"),
]

ref = None
for title, name in MODELS:
    print("=" * 100)
    print(f"[{title}]  {name}")
    print("=" * 100)
    model = DetectionModel(CFG + name, ch=3, nc=6, verbose=True)
    n_p = sum(p.numel() for p in model.parameters())
    n_layers = len(model.model)
    try:
        flops = get_flops(model, 640)
    except Exception as e:  # noqa: BLE001
        flops = float("nan")
        print("FLOPs 计算失败:", e)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 640, 640))

    def _shapes(o):
        if isinstance(o, torch.Tensor):
            return [tuple(o.shape)]
        if isinstance(o, dict):
            return {k: _shapes(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_shapes(v) for v in o]
        return type(o).__name__

    shapes = _shapes(out)
    print(f"\n>>> {name}")
    print(f"    层数 {n_layers} | 参数量 {n_p/1e6:.4f} M | GFLOPs {flops:.2f}")
    print(f"    640×640 前向输出形状: {shapes}")
    if ref is None:
        ref = n_p
        print(f"    （基准）参数量 {n_p/1e6:.4f} M")
    else:
        print(f"    参数量相对基座: {(n_p - ref)/1e3:+.1f} K ({(n_p - ref)/ref*100:+.2f}%)")
    print()
