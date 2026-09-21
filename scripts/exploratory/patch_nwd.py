# patch_nwd.py — 给服务器上的 ultralytics/utils/loss.py 打 NWD 混合损失补丁
# 用法：放到工程根目录，执行 python3 patch_nwd.py
# 幂等：已打过补丁会自动跳过；自带编译验证。
import pathlib
import py_compile
import sys

P = pathlib.Path("ultralytics/utils/loss.py")
if not P.exists():
    print("❌ 找不到 ultralytics/utils/loss.py，请在工程根目录运行")
    sys.exit(1)

s = P.read_text(encoding="utf-8")

if "wasserstein_loss" in s:
    print("✅ 已包含 NWD 补丁，无需重复打补丁")
else:
    # ---------- 补丁 1：NWD 函数与常量（插在 class BboxLoss 之前）----------
    anchor1 = "class BboxLoss(nn.Module):"
    assert s.count(anchor1) == 1, f"锚点1出现 {s.count(anchor1)} 次（预期1）"
    block1 = '''# ===================== 自研扩展：NWD 混合回归损失 =====================
# 背景：NEU-DET 细目标类（crazing/scratches）的框精度（mAP50-95）是最大短板，
#       而结构侧 13 组改进全部呈"块状类涨 / 细目标类崩"的类间零和 —— 损失函数维度从未探索。
# 机制：把 bbox 建模为二维高斯分布（均值=中心，协方差=diag((w/2)^2,(h/2)^2)），
#       以归一化 Wasserstein 距离度量相似度：NWD = exp(-sqrt(W2^2)/C)，与 CIoU 按权重混合。
#       对中心对齐/尺寸偏差给出更平滑的梯度，对小/细目标更均衡。
# 参考文献：A Normalized Gaussian Wasserstein Distance for Tiny Object Detection (arXiv:2110.13389)
# 控制：环境变量 YOLO_NWD_W（混合权重，默认 0 = 关闭，历史实验与基线不受影响）、
#       YOLO_NWD_C（常数，默认 0.1，适配 ultralytics 的归一化坐标尺度）。
import os

_NWD_W = float(os.environ.get("YOLO_NWD_W", "0"))
_NWD_C = float(os.environ.get("YOLO_NWD_C", "0.1"))


def wasserstein_loss(pred_bboxes: torch.Tensor, target_bboxes: torch.Tensor, C: float = 0.1) -> torch.Tensor:
    """归一化 Wasserstein 距离相似度（输入 xyxy 归一化坐标，返回 [0,1]，越大越相似）。"""
    p_cx = (pred_bboxes[..., 0] + pred_bboxes[..., 2]) * 0.5
    p_cy = (pred_bboxes[..., 1] + pred_bboxes[..., 3]) * 0.5
    p_w = (pred_bboxes[..., 2] - pred_bboxes[..., 0]).clamp_min(1e-7)
    p_h = (pred_bboxes[..., 3] - pred_bboxes[..., 1]).clamp_min(1e-7)
    t_cx = (target_bboxes[..., 0] + target_bboxes[..., 2]) * 0.5
    t_cy = (target_bboxes[..., 1] + target_bboxes[..., 3]) * 0.5
    t_w = (target_bboxes[..., 2] - target_bboxes[..., 0]).clamp_min(1e-7)
    t_h = (target_bboxes[..., 3] - target_bboxes[..., 1]).clamp_min(1e-7)
    w2 = (p_cx - t_cx) ** 2 + (p_cy - t_cy) ** 2 + ((p_w - t_w) * 0.5) ** 2 + ((p_h - t_h) * 0.5) ** 2
    return torch.exp(-torch.sqrt(w2 + 1e-9) / C)


'''
    s = s.replace(anchor1, block1 + anchor1, 1)

    # ---------- 补丁 2：BboxLoss.forward 里混合 NWD ----------
    # 注意：RotatedBboxLoss 有相似结构（但用 rbox2dist），锚点必须包含 bbox2dist 以保证唯一
    anchor2 = """        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)"""
    assert s.count(anchor2) == 1, f"锚点2出现 {s.count(anchor2)} 次（预期1）"
    block2 = """        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # ✅ 自研扩展：NWD 混合回归损失（YOLO_NWD_W 控制，默认 0 = 关闭，历史行为不变）
        if _NWD_W > 0:
            nwd = wasserstein_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], C=_NWD_C)
            loss_nwd = ((1.0 - nwd) * weight).sum() / target_scores_sum
            loss_iou = (1.0 - _NWD_W) * loss_iou + _NWD_W * loss_nwd

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)"""
    s = s.replace(anchor2, block2, 1)

    P.write_text(s, encoding="utf-8")
    print("✅ 补丁写入完成")

# ---------- 验证 ----------
py_compile.compile(str(P), doraise=True)
s2 = P.read_text(encoding="utf-8")
ok = ("wasserstein_loss" in s2) and ("_NWD_W" in s2) and ("loss_nwd" in s2)
print("编译通过 | NWD 函数存在:", "wasserstein_loss" in s2, "| 混合逻辑存在:", "loss_nwd" in s2)
if not ok:
    print("❌ 补丁内容不完整，请检查")
    sys.exit(1)
print("完成。启用训练示例：YOLO_NWD_W=0.5 yolo detect train ...")
print("（不设 YOLO_NWD_W 时训练行为与历史完全一致，可放心跑其他实验）")
