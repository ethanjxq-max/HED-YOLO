# benchmark_models.py — Table 4 用：参数量 / GFLOPs / 延迟 / FPS 同机同口径测量
# ============================================================================
# 为什么需要它：论文的"SOTA 对比表"必须同机、同口径测 FPS，不能引用各家论文里的数字。
#   本脚本对一串模型（yaml 或 best.pt 都行）统一测：
#     · Params(M)、GFLOPs（thop 口径）
#     · 纯网络前向延迟（FP16, batch=1, imgsz 默认 640, warmup+100 次取均值）
#     · 端到端 predict 延迟（含预处理 + 后处理；对非 e2e 模型**含 NMS**）与 FPS
#     · 对 YOLO26 系列额外测 **end2end=True（NMS-free）vs False（含 NMS）** 两条路径
# 用法（在服务器项目根目录）：
#   python benchmark_models.py --data ultralytics/cfg/datasets/neu_det.yaml
#   python benchmark_models.py --imgsz 640 --iters 100 --nimg 200 --half
# 说明：FPS 用 val 集图片跑端到端（不落盘），这是审稿人认可的口径；纯前向延迟用于横向比模型本体。
# ============================================================================
import argparse
import time

import torch

torch.backends.cudnn.benchmark = True  # 输入尺寸固定时自动选最快卷积算法

from ultralytics import YOLO

# ── 要对比的模型：名字 | 模型路径（yaml 或 .pt）| 备注
MODELS = [
    # 本文的模型（若某个 run 目录名不同，改这里）
    ("YOLO26s (baseline)", "ultralytics/cfg/models/26/yolo26s.yaml", "baseline（也可换成 runs/ab_base/ab_base/weights/best.pt）"),
    ("Ours final (DB+FEM+SAFR)", "runs/ab_db_fem/ab_db_fem/weights/best.pt", "本文最终模型"),
    # 效率变体（论文轻量化一节）
    ("Ours +MRB (folded)", "runs/r5_mrb/r5_mrb/weights/best.pt", "重参数化折叠（第5轮 r5_mrb）"),
    ("Ours +SPD (lossless down)", "runs/samp_yolo26s_spd_all_s42/samp_yolo26s_spd_all_s42/weights/best.pt", "无损下采样"),
    # 对比模型（按你们服务器上实际存在的路径填；v9s/v10s 若来自别的仓库，测不了就留空）
    ("YOLO26s (NMS path)", "runs/r6_noe2e/r6_noe2e/weights/best.pt", "同一网络的含 NMS 路径（测 NMS 开销）"),
    ("YOLOv11s", "ultralytics/cfg/models/11/yolo11s.yaml", "对比基线"),
    ("YOLOv8s", "ultralytics/cfg/models/v8/yolov8s.yaml", "对比基线"),
    ("YOLOv9s", "ultralytics/cfg/models/v9/yolov9s.yaml", "对比基线（若不存在请改成你实际用的路径）"),
    ("YOLOv10s", "ultralytics/cfg/models/v10/yolov10s.yaml", "对比基线（若不存在请改成你实际用的路径）"),
]


# ⚠ 运行前先 `ls runs/` 核对下面每个 .pt 的目录名（本项目是 runs/<name>/<name>/weights/best.pt 两层）
def gflops_of(model, imgsz):
    try:
        from thop import profile

        macs, _ = profile(model.model, inputs=(torch.randn(1, 3, imgsz, imgsz),), verbose=False)
        return macs / 1e9
    except Exception as e:  # thop 缺失时不影响其它指标
        print(f"    (GFLOPs 跳过：{type(e).__name__})")
        return float("nan")


def forward_latency(yolo, imgsz, iters, half, batch=1):
    dev = next(yolo.model.parameters()).device
    m = yolo.model.eval()
    x = torch.randn(batch, 3, imgsz, imgsz, device=dev)
    if half:
        m.half()
        x = x.half()
    with torch.no_grad():
        for _ in range(20):
            m(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            m(x)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1000
    return ms


def e2e_latency(yolo, source, imgsz, half, nimg):
    import glob

    files = sorted(glob.glob(f"{source}/*.jpg"))[:nimg]
    assert files, f"在 {source} 下没找到图片"
    t0 = time.perf_counter()
    n = 0
    for _ in yolo.predict(source=files, imgsz=imgsz, half=half, verbose=False, stream=True, save=False):
        n += 1
    dt = time.perf_counter() - t0
    return dt / n * 1000, n / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--nimg", type=int, default=200)
    ap.add_argument("--half", action="store_true", default=True)
    ap.add_argument("--data", default="ultralytics/cfg/datasets/neu_det.yaml")
    ap.add_argument("--source", default=None, help="端到端测速用图片目录（默认从 data 的 val 推导）")
    a = ap.parse_args()

    src = a.source
    if src is None:
        import yaml

        d = yaml.safe_load(open(a.data, encoding="utf-8"))
        root = d.get("path", ".")
        src = f"{root}/{d['val']}" if not str(d["val"]).startswith("/") else str(d["val"])
    print(f"# 测速图片目录：{src}\n")

    print(f"{'模型':<28}{'Params(M)':>10}{'GFLOPs':>9}{'fwd ms':>9}{'e2e ms':>9}{'FPS':>8}  备注")
    print("-" * 92)
    for name, path, note in MODELS:
        try:
            yolo = YOLO(path)
        except Exception as e:
            print(f"{name:<28}{'—':>10}{'—':>9}{'—':>9}{'—':>9}{'—':>8}  跳过（{type(e).__name__}: {str(e)[:40]}）")
            continue
        yolo.model.cuda().eval()
        params = sum(p.numel() for p in yolo.model.parameters()) / 1e6
        gf = gflops_of(yolo, a.imgsz)
        fwd = forward_latency(yolo, a.imgsz, a.iters, a.half)
        try:
            e2e_latency(yolo, src, a.imgsz, a.half, min(a.nimg, 20))  # 第一遍：预热/建 predictor，丢弃
            e2e, fps = e2e_latency(yolo, src, a.imgsz, a.half, a.nimg)
        except Exception as e:
            e2e, fps = float("nan"), float("nan")
            print(f"    (端到端测速失败：{type(e).__name__}: {str(e)[:60]})")
        print(f"{name:<28}{params:>10.3f}{gf:>9.2f}{fwd:>9.2f}{e2e:>9.2f}{fps:>8.1f}  {note}")

        # YOLO26 系列：额外测 NMS-free(True) vs 含 NMS(False) 两条推理路径
        head = yolo.model.model[-1] if hasattr(yolo.model, "model") else None
        if head is not None and hasattr(head, "end2end"):
            for flag in (True, False):
                head.end2end = flag
                try:
                    e2e2, fps2 = e2e_latency(yolo, src, a.imgsz, a.half, a.nimg)
                    print(f"    └ end2end={flag!s:<5} e2e {e2e2:>7.2f} ms   FPS {fps2:>6.1f}   "
                          f"（{'NMS-free' if flag else '含 NMS'}）")
                except Exception as e:
                    print(f"    └ end2end={flag} 测速失败（{type(e).__name__}）——跳过，NMS 开销请看 YOLO26s (NMS path) 那行")
            head.end2end = True

    print("\n# 记录进论文时请注明：GPU 型号、imgsz、FP16、batch=1（纯前向）与端到端含前后处理（FPS）")


if __name__ == "__main__":
    main()
