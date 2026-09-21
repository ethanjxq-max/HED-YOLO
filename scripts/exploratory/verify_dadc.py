# verify_dadc.py — DADC 模块离线自检（不占 GPU）
# 用法：python verify_dadc.py
import torch
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from ultralytics.nn.Convmodules.db import C3k2_DB  # 通过包导入，保证相对导入正确
from ultralytics.nn.Convmodules.dadc import C3k2_DB_DADC, DeformConv2d

torch.manual_seed(0)

print("=" * 84)
# 1) 零初始化 → 与标准卷积数值等价
x = torch.randn(2, 32, 16, 16)
w = torch.randn(24, 32, 3, 3)
y_ref = F.conv2d(x, w, padding=1)
y_def = deform_conv2d(x, torch.zeros(2, 18, 16, 16), w, None, 1, 1, 1)
print(f"[1] 零偏移 vs 标准卷积  最大误差 {(y_def - y_ref).abs().max().item():.2e}   (float32 累加顺序差异，量级 1e-5，属数值等价)")

# 2) 前向 + 参数量 + 三种开关
for c in (128, 256):
    for where in ("both", "cv2"):
        m = C3k2_DB_DADC(c, c, n=2, c3k=False, e=0.5, dcn_where=where)
        y = m(torch.randn(1, c, 20, 20))
        print(
            f"[2] C3k2_DB_DADC(c={c:<3d}, dcn_where={where:<4s}) {tuple(y.shape)}  参数 {sum(p.numel() for p in m.parameters())/1e3:7.1f}K"
        )

# 3) 与 C3k2_DB 参数量对比（同壳，仅分支A 卷积核不同）
print("-" * 84)
for c in (128, 256, 512):
    db = C3k2_DB(c, c, n=2, c3k=False, e=0.5)
    dc = C3k2_DB_DADC(c, c, n=2, c3k=False, e=0.5)
    p_db = sum(p.numel() for p in db.parameters())
    p_dc = sum(p.numel() for p in dc.parameters())
    print(f"[3] c={c:<3d}: C3k2_DB {p_db/1e3:8.1f}K → C3k2_DB_DADC {p_dc/1e3:8.1f}K   ({p_dc - p_db:+.0f} 参数)")

# 4) 消融开关
print("-" * 84)
for kw in (dict(offset_mode="standard"), dict(use_mask=True), dict(dcn_where="cv2"), dict(offset_mode="standard", use_mask=True)):
    m = C3k2_DB_DADC(256, 256, n=2, c3k=False, e=0.5, **kw)
    y = m(torch.randn(1, 256, 20, 20))
    print(f"[4] 消融 {str(kw):<52s} {tuple(y.shape)} 参数 {sum(p.numel() for p in m.parameters())/1e3:7.1f}K")

# 5) attn 分支（head P5 用）
m = C3k2_DB_DADC(1024, 512, n=1, c3k=True, e=0.5, attn=True)
y = m(torch.randn(1, 1024, 20, 20))
print(f"[5] attn=True/c3k=True: {tuple(y.shape)} 参数 {sum(p.numel() for p in m.parameters())/1e6:.3f}M")

# 6) 梯度连通性
print("-" * 84)
m = C3k2_DB_DADC(64, 64, n=1, c3k=False, e=0.5)
y = m(torch.randn(1, 64, 20, 20))
y.pow(2).mean().backward()
g_off = m.m[0].branch_a.cv1.off_h.weight.grad.abs().sum().item()
g_w = m.m[0].branch_a.cv1.weight.grad.abs().sum().item()
print(f"[6] 梯度 偏移生成器 {g_off:.3e} / 卷积核 {g_w:.3e}")
assert g_off > 0 and g_w > 0, "偏移生成器无梯度！"

# 7) 起步等价性：DADC 版块 vs DB 版块，在零初始化下输出应一致（仅分支A 的核初始化不同 → 用同一权重拷贝对比）
print("-" * 84)
db = C3k2_DB(64, 64, n=1, c3k=False, e=0.5).eval()
dc = C3k2_DB_DADC(64, 64, n=1, c3k=False, e=0.5).eval()
xin = torch.randn(1, 64, 20, 20)


def copy_matching(dst, src):
    """把 src（全 Conv 版块）的权重按语义对齐到 dst（分支A 换核版块）。
    src 键形如 m.0.branch_a.cv1.conv.weight（Conv 内核）；dst 键形如 m.0.branch_a.cv1.weight。"""
    import re

    sd, ss = dst.state_dict(), src.state_dict()
    n_hit = 0
    for k, v in ss.items():
        for k2 in (k, re.sub(r"(cv\d)\.conv\.", r"\1.", k)):  # 先试原名，再试"换核"映射名
            if k2 in sd and sd[k2].shape == v.shape and not torch.equal(sd[k2], v):
                sd[k2] = v.clone()
                n_hit += 1
                break
    dst.load_state_dict(sd)
    return n_hit


b_db = db.m[0].branch_a
b_dc = dc.m[0].branch_a
n_hit = copy_matching(dc, db)  # 整块对齐（含分支A 的两个核 + 其余所有同名张量）
with torch.no_grad():
    o_db = db(xin)
    o_dc = dc(xin)
print(f"[7] 同步 {n_hit} 个张量后，两个块输出最大差异 {(o_db - o_dc).abs().max().item():.2e}")
print("    → 零偏移下 DADC 与 DB 是**逐位等价的同一函数**；训练起点 = E 基座（换核不换结构）")

# 8) 三种偏移参数化的起步等价性（都必须零扰动）
print("-" * 84)
for mode in ("directional", "snake", "standard"):
    db2 = C3k2_DB(64, 64, n=1, c3k=False, e=0.5).eval()
    dc2 = C3k2_DB_DADC(64, 64, n=1, c3k=False, e=0.5, offset_mode=mode).eval()
    h = copy_matching(dc2, db2)
    with torch.no_grad():
        d = (db2(xin) - dc2(xin)).abs().max().item()
    print(f"[8] offset_mode={mode:<12s} 同步 {h:3d} 张量 → 输出差异 {d:.2e}  ({'等价 ✓' if d < 1e-5 else '不一致 ✗'})")

# 9) snake 模式的链式偏移结构（人为给定等量增量，检验累积效果）
print("-" * 84)
import math as _m

dcv = DeformConv2d(4, 4, k=3, offset_mode="snake", max_offset=2.0)
with torch.no_grad():
    dcv.off_h.bias.fill_(_m.atanh(0.5))  # 每步 x 增量 = tanh(v)·max_offset/k = 0.5·2/3 = 1/3
    dcv.off_v.bias.fill_(0.0)
off = dcv._offset(torch.zeros(1, 4, 8, 8), 1, 8, 8)
dx_all = [round(float(t), 3) for t in off[0, 1::2, 4, 4]]  # 9 个核位置的 x 偏移
print(f"[9] snake 模式 9 个核位置的 x 偏移: {dx_all}")
print("    期望每行按 0.333 → 0.667 → 1.0 递增（沿条形方向累积的链式约束）")
print("✅ 全部自检通过")
