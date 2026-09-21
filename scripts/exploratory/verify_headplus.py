# verify_headplus.py — DetectPlus（DRA 适配器 / 推理头加宽 / o2o_topk 诊断）上线前验证
# 用法：cd 项目根目录 && python verify_headplus.py
import copy

import torch

from ultralytics.nn.tasks import DetectionModel

CFG = "ultralytics/cfg/models/26/"


def build(name, nc=6):
    return DetectionModel(f"{CFG}{name}.yaml", ch=3, nc=nc, verbose=False)


def flat(y):
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
print("一、参数量（全部保持 end2end=True + reg_max=1：NMS-free 与 DFL-free 都不动）")
print("=" * 78)
E = build("yolo26s_db_fem_11_13")
p0 = sum(p.numel() for p in E.parameters())
for n in ("yolo26s_db_fem_11_13", "yolo26s_db_fem_11_13_dra", "yolo26s_db_fem_11_13_wide_o2o",
          "yolo26s_db_fem_11_13_o2o3"):
    m = build(n)
    d = m.model[-1]
    p = sum(x.numel() for x in m.parameters())
    print(f"{n:<32} {p/1e6:8.4f} M ({p-p0:+8.0f})  end2end={d.end2end}  reg_max={d.reg_max}  {d.extra_repr()}")

print()
print("=" * 78)
print("二、DRA 起步等价性：同权重 + 适配器零初始化 ⇒ 输出必须与 E 基座逐位一致")
print("=" * 78)
dra = build("yolo26s_db_fem_11_13_dra")
hidx = len(E.model) - 1
sd = {k: v for k, v in E.state_dict().items() if not k.startswith(f"model.{hidx}.adapt")}
res = dra.load_state_dict(sd, strict=False)
bad = [k for k in res.missing_keys if ".adapt." not in k]
print(f"未匹配键：{len(res.missing_keys)} 个（其中非适配器的：{bad}）  意外键：{res.unexpected_keys}")
assert not bad, f"有权重没搬过去：{bad[:5]}"
E.eval(), dra.eval()
x = torch.randn(1, 3, 320, 320)
with torch.no_grad():
    y1, y2 = flat(E(x)), flat(dra(x))
err = max((a - b).abs().max().item() for a, b in zip(y1, y2))
print(f"最大输出差 = {err:.2e}  （应 <1e-4）")
assert err < 1e-4, "DRA 起步不等价于基线"

print()
print("=" * 78)
print("三、适配器可学性 + 只由一对一损失训练（骨干不受影响）")
print("=" * 78)
dra.train()
for p in dra.parameters():
    p.grad = None
out = dra(torch.randn(1, 3, 320, 320))
loss = sum(t.float().pow(2).mean() for t in flat(out))
loss.backward()
ad = dra.model[hidx].adapt
g_ad = sum(q.grad.abs().sum().item() for q in ad.parameters() if q.grad is not None)
g_backbone = dra.model[2].cv1.conv.weight.grad
print(f"适配器梯度量级 = {g_ad:.4f}（>0 表示可学）；骨干第 2 层梯度是否为空 = {g_backbone is None}（训练前向应非空）")
assert g_ad > 0, "适配器无梯度"

print()
print("=" * 78)
print("四、o2o_topk 诊断开关（E2ELoss 通过 head.o2o_topk 读取，官方默认 = 1）")
print("=" * 78)
for n in ("yolo26s_db_fem_11_13", "yolo26s_db_fem_11_13_o2o3"):
    m = build(n)
    print(f"{n:<32} head.o2o_topk = {getattr(m.model[-1], 'o2o_topk', 1)}"
          f"   → E2ELoss 会用 tal_topk2 = {int(getattr(m.model[-1], 'o2o_topk', 1))}")
print()
print("✅ verify_headplus.py 全部通过")
