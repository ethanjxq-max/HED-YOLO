# headplus.py — 为 YOLO26 的**原生 NMS-free 推理头**做结构增强（不改变端到端范式、不恢复 DFL）
# ==================================================================================
# 为什么要"只动推理头"（2026-09-14，基于第 6/7 轮诊断 + 官方 YOLO26 设计文档）：
#
#   YOLO26 的四个立身之本（docs/en/models/yolo26.md）：
#     ① 原生端到端、NMS-free 推理（一对一头直接出框，无 NMS 后处理）
#     ② DFL-free 回归（reg_max=1，L1+CIoU，无分布分箱、导出更简单）
#     ③ 训练配方：MuSGD（Muon+SGD 混合）、Progressive Loss（监督从一对多逐步转向一对一）、
#        STAL（小目标感知标签分配，保证小目标的正样本覆盖）
#     ④ 任务头扩展与部署效率（论文报 YOLO26n CPU ONNX 比 YOLO11n 快 43%）
#   → 因此"关掉 end2end"或"恢复 reg_max=16"都是**把 YOLO26 改回 YOLOv8/11**，论文里不能这么写。
#     第 6 轮的 0.757/0.402 只作为**诊断结论**（量化 NMS-free 头在钢缺陷小数据上的代价 3.2 分），
#     它同时指出了第三创新点应该落在哪里：**推理头本身**。
#
#   代码级证据（ultralytics/nn/modules/head.py, utils/loss.py）：
#     · 一对一（推理）头训练时读的是 **detach 后的特征**（x_detach，梯度不回骨干）——设计如此；
#     · 一对一分支配置 **每 GT 只有 1 个正样本**（E2ELoss: one2one = loss_fn(model, tal_topk=7, tal_topk2=1)，
#       tal.py 中 topk2 即"每 GT 保留的最终正样本数"）；
#     · 一对多监督权重按 Progressive Loss 从 0.8 衰减到 0.1。
#   ⇒ 推理头只能在"冻结特征 + 稀疏监督"下自己调权重，这就是它在细粒度缺陷上掉 3.2 分的最可能原因。
#
#   本文件给出的两个**结构**干预（均为"新模块 / 新网络结构"，不碰损失函数、不改端到端范式）：
#     A) HeadAdapter（DRA，推理头细化适配器）：逐尺度的轻量适配器（DW3×3 + 1×1 + 恒等残差，
#        输出投影零初始化 ⇒ 起步严格等于官方行为），**只插在一对一推理头的输入路径上**，
#        只由一对一损失训练（detach 保证不污染骨干，YOLO26 的原始设计不变），
#        让推理头能"重塑"特征而不只是"重加权"冻结特征。推理时开销 ~0.05M 参数/3 尺度。
#     B) wide（推理头容量重分配）：只把**一对一推理头**的 box/cls 分支通道加倍
#        （box: ch/4 → ch/2，cls: ch → 2ch），一对多（训练用）分支逐字不动 ——
#        用来区分"推理头欠拟合是容量问题还是分配问题"。
#
#   用法（yaml，把 Detect 换掉即可；ch 由 tasks.py 传入）：
#     - [[17, 20, 24], 1, DetectPlus, [nc, True, False, 1]]   # adapt=True
#     - [[17, 20, 24], 1, DetectPlus, [nc, False, True, 1]]   # wide=True（推理头加宽）
#     - [[17, 20, 24], 1, DetectPlus, [nc, False, False, 3]]  # o2o_topk=3（**仅诊断**：分配侧对照）
#   第 4 个参数 o2o_topk 通过 head.o2o_topk 属性被 utils/loss.py 的 E2ELoss 读取（默认 1 = 官方设定）。
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect

__all__ = ["HeadAdapter", "DetectPlus"]


class HeadAdapter(nn.Module):
    """推理头细化适配器：x → x + out(SiLU(pw(dw(x))))，输出投影零初始化（起步 = 恒等）。

    只服务于 NMS-free 推理头，参数量 ~ (9c + c·h + h·c)/尺度（h = c/2）。
    """

    def __init__(self, c, hidden=None, init_scale=0.0):
        super().__init__()
        hidden = hidden or max(8, c // 2)
        self.dw = Conv(c, c, 3, 1, g=c)  # 深度可分离 3×3（局部细化）
        self.pw = nn.Conv2d(c, hidden, 1, bias=True)  # 通道压缩
        self.out = nn.Conv2d(hidden, c, 1, bias=True)  # 输出投影（零初始化）
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        if init_scale:
            nn.init.normal_(self.out.weight, std=init_scale)

    def forward(self, x):
        return x + self.out(F.silu(self.pw(self.dw(x))))


class DetectPlus(Detect):
    """YOLO26 Detect 的"推理头增强"变体（保持 NMS-free 端到端推理与 DFL-free 回归不变）。

    参数（yaml 顺序）：[nc, adapt, wide, o2o_topk]，其余 (reg_max, end2end, ch) 由 tasks.py 传入。
      adapt   : 启用推理头细化适配器（只作用于一对一头）
      wide    : 推理头 box/cls 分支加宽（只作用于一对一头）
      o2o_topk: 一对一分支配置每 GT 的正样本数（官方 = 1；>1 仅用于**诊断**分配假设）
    """

    def __init__(self, nc=80, adapt=False, wide=False, wide_o2m=False, o2o_topk=1, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, reg_max, end2end, ch)
        self.o2o_topk = int(o2o_topk)
        self.use_adapt = bool(adapt)
        self.use_wide = bool(wide)
        self.use_wide_o2m = bool(wide_o2m)

        if self.use_adapt:
            self.adapt = nn.ModuleList(HeadAdapter(x) for x in ch)

        # ---- 只加宽一对一（推理）头：推理时有成本（第 9 轮实测 −0.010，已放弃该方向）----
        if self.use_wide:
            c2, c3 = self._wide_channels(ch)
            self.one2one_cv2, self.one2one_cv3 = self._make_branches(ch, c2, c3)

        # ---- 只加宽一对多（辅助训练）头：**推理时零成本**（该分支不参与推理）----
        # 机制：一对多的梯度是唯一能塑造骨干/颈部特征的路径（一对一头读 detach 特征），
        #      给它更大容量 = 用零推理开销换取更好的共享特征。
        if self.use_wide_o2m:
            c2, c3 = self._wide_channels(ch)
            self.cv2, self.cv3 = self._make_branches(ch, c2, c3)

    def _wide_channels(self, ch):
        return max(16, ch[0] // 2, self.reg_max * 4), max(ch[0] * 2, min(self.nc, 100))

    def _make_branches(self, ch, c2, c3):
        box = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        if self.legacy:
            cls = nn.ModuleList(
                nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
            )
        else:  # 与官方 YOLO26 一致的深度可分离 cls 分支结构
            cls = nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        return box, cls

    def forward_head(self, x, box_head=None, cls_head=None):
        """仅在一对一（推理）头的调用上施加适配器；一对多（训练）头读原始特征。"""
        if self.use_adapt and box_head is getattr(self, "one2one_cv2", None):
            x = [a(xi) for a, xi in zip(self.adapt, x)]
        return super().forward_head(x, box_head, cls_head)

    def extra_repr(self):
        return (f"adapt={self.use_adapt}, wide={self.use_wide}, wide_o2m={self.use_wide_o2m}, "
                f"o2o_topk={self.o2o_topk}, reg_max={self.reg_max}")


if __name__ == "__main__":
    torch.manual_seed(0)
    from ultralytics.nn.tasks import DetectionModel

    # 1) 建模 + 参数量（三种开关）
    for kw in ("adapt=True", "wide=True", "o2o_topk=3", "默认(等价官方)"):
        y = "yolo26s_db_fem_11_13_dra" if "adapt" in kw else (
            "yolo26s_db_fem_11_13_wide_o2o" if "wide" in kw else (
                "yolo26s_db_fem_11_13_o2o3" if "o2o" in kw else "yolo26s_db_fem_11_13"))
        m = DetectionModel(f"ultralytics/cfg/models/26/{y}.yaml", ch=3, nc=6, verbose=False)
        d = m.model[-1]
        p = sum(x.numel() for x in m.parameters())
        ref = sum(x.numel() for x in DetectionModel(
            "ultralytics/cfg/models/26/yolo26s_db_fem_11_13.yaml", ch=3, nc=6, verbose=False).parameters())
        print(f"{y:<34} 参数 {p/1e6:.4f} M（E 基座 {ref/1e6:.4f} M，{p-ref:+.0f}）  {d}")

    # 2) adapt 起步等价性：零初始化 ⇒ adapter 输出与恒等一致 ⇒ 整模型输出 == E 基座
    import copy as _copy

    m1 = DetectionModel("ultralytics/cfg/models/26/yolo26s_db_fem_11_13.yaml", ch=3, nc=6, verbose=False).eval()
    m2 = _copy.deepcopy(m1)
    # 把 m2 的对一检测头换成带 adapter 的版本（模拟 yaml 换模块后的初始化状态）
    m2.model[-1] = DetectPlus(6, True, False, 1, 1, True, (128, 256, 512)).eval()
    m2.model[-1].stride = m1.model[-1].stride
    m2.model[-1].no, m2.model[-1].nc, m2.model[-1].nl = m1.model[-1].no, 6, 3
    x = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        y1 = _flat = m1(x)
        y2 = m2(x)
    d1 = [_flat] if not isinstance(y1, (list, tuple)) else list(y1)
    d2 = [y2] if not isinstance(y2, (list, tuple)) else list(y2)
    err = max((a - b).abs().max().item() for a, b in zip(d1, d2))
    print(f"adapt 零初始化起步：与 E 基座输出最大差 {err:.2e}（应 <1e-4，证明起步严格等价）")
    assert err < 1e-4, "adapter 起步未等价于基线"

    # 3) 可学性：adapter 的输出投影与内部卷积都要有梯度
    ad = HeadAdapter(128)
    ad(torch.randn(2, 128, 20, 20)).pow(2).mean().backward()
    g = {n: float(p.grad.abs().sum()) for n, p in ad.named_parameters() if p.grad is not None}
    print("adapter 各参数梯度:", {k: round(v, 5) for k, v in g.items()}, "（out 权重梯度非零 → 零初始化不会卡死）")
    assert g["out.weight"] > 0 and g["dw.conv.weight"] > 0
    print("✅ headplus.py 全部测试通过！")
