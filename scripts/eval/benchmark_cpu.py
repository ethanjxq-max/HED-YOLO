# benchmark_cpu.py — CPU 单图延迟对比（YOLO26 的官方卖点之一就是 CPU 推理更快：
#                  官方论文报 YOLO26n 在 Intel Xeon CPU 的 ONNX 推理比 YOLO11n 快 43%）
# ============================================================================
# 为什么要在 CPU 上测：GPU 上各家模型都被"喂饱"了，差异被并行度掩盖（我们实测 2.2~2.9 ms 互有胜负）；
#   而工业现场常常是 CPU/边缘设备部署，CPU 上的差异主要来自**算子数量、访存、后处理**——
#   这恰好是 YOLO26 的设计改动（去掉 DFL、更轻的分类分支、NMS-free 省掉后处理）最能体现的地方。
#
# 测什么（全部 CPU、FP32、batch=1、imgsz=640）：
#   ① 纯网络前向延迟 ms（warmup 后取均值）  ② 端到端 predict 延迟 ms 与 FPS（含 letterbox 与后处理，含 NMS）
# 用法：
#   python benchmark_cpu.py --data ultralytics/cfg/datasets/neu_det.yaml --imgsz 640 --iters 20
# ============================================================================
import argparse
import glob
import os
import time

import torch
import yaml

from ultralytics import YOLO

# 名字 | 模型路径 | 是否先折叠重参数化分支 | 备注
MODELS = [
    ("YOLO26s (baseline)",       "runs/ab_base/ab_base/weights/best.pt",       False, "基座 0.711/0.371"),
    ("Ours final (NMS-free)",    "runs/ab_db_fem/ab_db_fem/weights/best.pt",   False, "本文最终 0.751/0.388"),
    ("Ours final (NMS path)",    "runs/r7_E/r7_E/weights/best.pt",             False, "本文最终含 NMS 0.742/0.394"),
    ("YOLO26s (NMS path)",       "runs/r6_noe2e/r6_noe2e/weights/best.pt",     False, "同网络含 NMS 0.757/0.402"),
    ("Ours +MRB (folded)",       "runs/r5_mrb/r5_mrb/weights/best.pt",         True,  "重参数化折叠后"),
    ("YOLOv11s",                 "ultralytics/cfg/models/11/yolo11s.yaml",     False, "对比基线 0.758/0.386"),
    ("YOLOv8s",                  "ultralytics/cfg/models/v8/yolov8s.yaml",     False, "对比基线 0.743/0.377"),
    ("YOLOv9s",                  "ultralytics/cfg/models/v9/yolov9s.yaml",     False, "对比基线 0.741/0.397"),
    ("YOLOv10s",                 "ultralytics/cfg/models/v10/yolov10s.yaml",   False, "对比基线 0.697/0.352"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iters", type=int, default=20, help="纯前向迭代次数（CPU 上别设太大）")
    ap.add_argument("--nimg", type=int, default=20, help="端到端测速图片数（0=跳过）")
    ap.add_argument("--threads", type=int, default=0, help="0=用全部核心")
    ap.add_argument("--data", default="ultralytics/cfg/datasets/neu_det.yaml")
    a = ap.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    d = yaml.safe_load(open(a.data, encoding="utf-8"))
    root, val = d.get("path", "."), str(d["val"])
    src = val if val.startswith("/") else f"{root}/{val}"
    files = sorted(glob.glob(f"{src}/*.jpg"))[: a.nimg]

    print(f"# CPU: {os.cpu_count()} 核，torch 使用 {torch.get_num_threads()} 线程 | imgsz={a.imgsz} | FP32 | batch=1")
    print(f"# 端到端测速图片：{len(files)} 张（{src}）\n")
    print(f"{'模型':<26}{'前向 ms':>10}{'端到端 ms':>11}{'FPS':>8}  备注")
    print("-" * 84)

    for name, path, fuse, note in MODELS:
        try:
            yo = YOLO(path)
        except Exception as e:
            print(f"{name:<26}{'—':>10}{'—':>11}{'—':>8}  跳过({type(e).__name__})")
            continue
        if fuse:
            try:
                from ultralytics.nn.Convmodules.repblock import rep_fuse_model

                n, b, af = rep_fuse_model(yo.model)
                note += f"（折叠 {n} 处：{b/1e6:.3f}→{af/1e6:.3f} M）"
            except Exception as e:
                note += f"（折叠失败：{type(e).__name__}）"

        # ① 纯前向
        fwd = float("nan")
        try:
            m = yo.model.cpu().eval().float()
            x = torch.zeros(1, 3, a.imgsz, a.imgsz)
            with torch.no_grad():
                for _ in range(2):
                    m(x)
                t0 = time.perf_counter()
                for _ in range(a.iters):
                    m(x)
                fwd = (time.perf_counter() - t0) / a.iters * 1000
        except Exception as e:
            print(f"    (前向失败：{type(e).__name__}: {str(e)[:60]})")

        # ② 端到端（含 letterbox 与后处理/NMS）
        e2e = fps = float("nan")
        if a.nimg and files:
            try:
                for _ in yo.predict(source=files[:2], imgsz=a.imgsz, device="cpu", verbose=False, stream=True):
                    pass
                t0 = time.perf_counter()
                n = 0
                for _ in yo.predict(source=files, imgsz=a.imgsz, device="cpu", verbose=False, stream=True):
                    n += 1
                dt = time.perf_counter() - t0
                e2e, fps = dt / n * 1000, n / dt
            except Exception as e:
                print(f"    (端到端失败：{type(e).__name__}: {str(e)[:60]})")

        print(f"{name:<26}{fwd:>10.1f}{e2e:>11.1f}{fps:>8.2f}  {note}")

    print("\n# 论文注明：CPU 型号与核数、线程数、FP32、batch=1、imgsz；延迟含/不含前后处理要写清")


if __name__ == "__main__":
    main()
