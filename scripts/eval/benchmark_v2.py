# benchmark_v2.py — 参数量 / GFLOPs / 端到端延迟 / FPS（修正版：不手动搬设备，交给 ultralytics 自己管）
# ============================================================================
# 上一版报 "Expected all tensors to be on the same device" 的原因：脚本先手动 `model.cuda()/half()`，
# 又让 predict 自己选设备，两边不一致；thop 在 CUDA 模型上喂 CPU 张量也报同错。
# 本版顺序：① 先 predict 一次预热（此时 ultralytics 会把模型/缓冲区都放到正确设备与 dtype）
#           ② 再测端到端延迟（含预处理+后处理，含 NMS）与 FPS
#           ③ 用已就绪的模型测纯前向（FP16, batch=1）
#           ④ GFLOPs 用 CPU 副本单独算（避免设备冲突）
# 用法：
#   pip install thop
#   python benchmark_v2.py --data ultralytics/cfg/datasets/neu_det.yaml --imgsz 640
#   python benchmark_v2.py --nimg 0        # 只测前向/GFLOPs，跳过端到端
# ============================================================================
import argparse
import copy
import glob
import time

import torch
import yaml

from ultralytics import YOLO

# 名字 | 模型路径 | 备注（.pt 是 runs/<name>/<name>/weights/best.pt，两层目录）
MODELS = [
    ("YOLO26s (baseline)",        "runs/ab_base/ab_base/weights/best.pt",              "基座 0.711/0.371"),
    ("Ours final (DB+FEM+SAFR)",  "runs/ab_db_fem/ab_db_fem/weights/best.pt",          "本文最终 0.751/0.388"),
    ("YOLO26s (NMS path)",        "runs/r6_noe2e/r6_noe2e/weights/best.pt",            "同网络含 NMS 版 0.757/0.402"),
    ("YOLO26s +MRB (folded)",     "runs/r5_mrb/r5_mrb/weights/best.pt",                "重参数化"),
    ("YOLOv11s",                  "ultralytics/cfg/models/11/yolo11s.yaml",           "对比基线 0.758/0.386"),
    ("YOLOv8s",                   "ultralytics/cfg/models/v8/yolov8s.yaml",           "对比基线 0.743/0.377"),
    ("YOLOv9s",                   "ultralytics/cfg/models/v9/yolov9s.yaml",           "对比基线 0.741/0.397"),
    ("YOLOv10s",                  "ultralytics/cfg/models/v10/yolov10s.yaml",         "对比基线 0.697/0.352"),
]


def gflops_cpu(model, imgsz):
    """在 CPU 副本上算 GFLOPs，避免与 CUDA 设备冲突。"""
    try:
        from thop import profile

        m = copy.deepcopy(model).float().cpu().eval()
        macs, _ = profile(m, inputs=(torch.zeros(1, 3, imgsz, imgsz),), verbose=False)
        return macs / 1e9
    except Exception as e:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iters", type=int, default=200, help="纯前向迭代次数")
    ap.add_argument("--nimg", type=int, default=300, help="端到端测速图片数（0=跳过）")
    ap.add_argument("--warm", type=int, default=30, help="端到端预热图片数")
    ap.add_argument("--data", default="ultralytics/cfg/datasets/neu_det.yaml")
    ap.add_argument("--source", default=None)
    a = ap.parse_args()

    src = a.source
    if src is None:
        d = yaml.safe_load(open(a.data, encoding="utf-8"))
        root, val = d.get("path", "."), str(d["val"])
        src = val if val.startswith("/") else f"{root}/{val}"
    files = sorted(glob.glob(f"{src}/*.jpg")) or sorted(glob.glob(f"{src}/*.png"))
    files = files[: max(a.nimg + a.warm, 1)]
    print(f"# 测速图片目录：{src}（{len(files)} 张）")
    print(f"# GPU: {torch.cuda.get_device_name(0)} | imgsz={a.imgsz} | FP16 | batch=1（前向）\n")
    print(f"{'模型':<28}{'Params(M)':>10}{'GFLOPs':>9}{'fwd ms':>9}{'e2e ms':>9}{'FPS':>8}  备注")
    print("-" * 96)

    for name, path, note in MODELS:
        row = [name]
        try:
            yo = YOLO(path)
        except Exception as e:
            print(f"{name:<28}{'—':>10}{'—':>9}{'—':>9}{'—':>9}{'—':>8}  跳过({type(e).__name__})")
            continue

        params = sum(p.numel() for p in yo.model.parameters()) / 1e6
        gf = gflops_cpu(yo.model, a.imgsz)

        # ① 预热：交给 ultralytics 决定设备与 dtype（第一次会建 predictor/编译）
        e2e = fps = float("nan")
        if a.nimg > 0 and files:
            try:
                for _ in yo.predict(source=files[: a.warm], imgsz=a.imgsz, half=True,
                                    verbose=False, stream=True, save=False):
                    pass
                t0 = time.perf_counter()
                n = 0
                for _ in yo.predict(source=files, imgsz=a.imgsz, half=True,
                                    verbose=False, stream=True, save=False):
                    n += 1
                dt = time.perf_counter() - t0
                e2e, fps = dt / n * 1000, n / dt
            except Exception as e:
                print(f"    (端到端测速失败：{type(e).__name__}: {str(e)[:70]})")

        # ② 纯前向：此时 yo.model 已在正确设备/dtype 上
        fwd = float("nan")
        try:
            m = yo.model.eval()
            dev = next(m.parameters()).device
            x = torch.zeros(1, 3, a.imgsz, a.imgsz, device=dev)
            x = x.half() if next(m.parameters()).dtype == torch.float16 else x
            with torch.no_grad():
                for _ in range(20):
                    m(x)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(a.iters):
                    m(x)
                torch.cuda.synchronize()
                fwd = (time.perf_counter() - t0) / a.iters * 1000
        except Exception as e:
            print(f"    (前向测速失败：{type(e).__name__}: {str(e)[:70]})")

        dev = next(yo.model.parameters()).device
        print(f"{name:<28}{params:>10.3f}{gf:>9.2f}{fwd:>9.2f}{e2e:>9.2f}{fps:>8.1f}  {note}")
        del yo
        torch.cuda.empty_cache()

    print("\n# 论文注明：GPU 型号、imgsz、FP16、batch=1；e2e 含预处理+网络+后处理（非 NMS-free 模型含 NMS）")


if __name__ == "__main__":
    main()
