# benchmark_onnx_cpu.py — ONNX Runtime CPU 延迟（与 YOLO26 官方"CPU ONNX 快 43%"同一测量口径）
# ============================================================================
# 为什么要单独测 ONNX：官方那句"YOLO26n 比 YOLO11n 在 Xeon CPU 上快 43%"是 **ONNX Runtime + CPU** 口径。
# 我们上一版用 PyTorch eager + FP32 测出来各家都在 57~68 ms（差 <15%），原因是**固定的 Python 预处理/
# 后处理开销（letterbox、decode、collate）占了 60~70%**，把网络本身的差异盖住了。
# 本脚本只测**网络图本身**（ONNX，无 Python 前后处理），这才是架构差异能显出来的口径。
#
# 依赖：pip install onnxruntime onnx
# 用法：
#   python benchmark_onnx_cpu.py --imgsz 640 --iters 30 --threads 0
# 输出：每个模型导出 .onnx（放在权重旁）并测 CPU 单图延迟 ms 与 FPS
# ============================================================================
import argparse
import os
import time

import numpy as np
import torch

from ultralytics import YOLO

MODELS = [
    ("YOLO26s (baseline)",    "runs/ab_base/ab_base/weights/best.pt",     "0.711/0.371"),
    ("Ours final",            "runs/ab_db_fem/ab_db_fem/weights/best.pt", "0.751/0.388"),
    ("Ours +MRB (folded)",    "runs/r5_mrb/r5_mrb/weights/best.pt",       "0.713/0.371"),
    ("YOLOv11s",              "ultralytics/cfg/models/11/yolo11s.yaml",   "0.758/0.386"),
    ("YOLOv8s",               "ultralytics/cfg/models/v8/yolov8s.yaml",   "0.743/0.377"),
    ("YOLOv9s",               "ultralytics/cfg/models/v9/yolov9s.yaml",   "0.741/0.397"),
    ("YOLOv10s",              "ultralytics/cfg/models/v10/yolov10s.yaml", "0.697/0.352"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--warm", type=int, default=3)
    a = ap.parse_args()

    import onnxruntime as ort

    if a.threads:
        os.environ["OMP_NUM_THREADS"] = str(a.threads)
        ort.set_default_logger_severity(3)
    so = ort.SessionOptions()
    if a.threads:
        so.intra_op_num_threads = a.threads
    print(f"# CPU {os.cpu_count()} 核 | onnxruntime {ort.__version__} | imgsz={a.imgsz} | FP32 | batch=1")
    print(f"{'模型':<24}{'ONNX ms':>10}{'FPS':>8}  备注")
    print("-" * 66)

    for name, path, note in MODELS:
        try:
            yo = YOLO(path)
            if "mrb" in path.lower():
                from ultralytics.nn.Convmodules.repblock import rep_fuse_model

                rep_fuse_model(yo.model)  # 折叠为单卷积后再导出（推理态）
                note += "（已折叠）"
            onnx_path = yo.export(format="onnx", imgsz=a.imgsz, half=False, simplify=True, dynamic=False, verbose=False)
        except Exception as e:
            print(f"{name:<24}{'—':>10}{'—':>8}  导出失败({type(e).__name__}: {str(e)[:40]})")
            continue
        try:
            sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
            inp = sess.get_inputs()[0]
            x = np.zeros((1, 3, a.imgsz, a.imgsz), dtype=np.float32)
            for _ in range(a.warm):
                sess.run(None, {inp.name: x})
            t0 = time.perf_counter()
            for _ in range(a.iters):
                sess.run(None, {inp.name: x})
            ms = (time.perf_counter() - t0) / a.iters * 1000
            print(f"{name:<24}{ms:>10.1f}{1000/ms:>8.2f}  {note}")
        except Exception as e:
            print(f"{name:<24}{'—':>10}{'—':>8}  推理失败({type(e).__name__}: {str(e)[:40]})")
        finally:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n# 论文注明：CPU 型号/核数、onnxruntime 版本、FP32、batch=1、imgsz、是否 dynamic；"
          "ONNX 图内不含 Python 前后处理（NMS-free 与含 NMS 的差别在部署侧另述）")


if __name__ == "__main__":
    main()
