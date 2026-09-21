# HED-YOLO（中文简要说明）

> 论文配套代码：**面向 CPU 钢表面检测的 NMS-free 端到端检测：量化并补偿 YOLO26 的精度代价**
>
> 完整说明（安装、数据准备、复现步骤、模型清单）请看 [README.md](README.md)。

本项目基于 **YOLO26s**（Ultralytics 8.4.137），在**不改变其原生 NMS-free 端到端推理与无 DFL 回归范式**的前提下，提出三项特征侧结构改进：

| # | 贡献 | 简称 | 代码位置 |
|---|---|---|---|
| 1 | 异构双分支瓶颈块（局部细节分支 ‖ 大核可分离注意力上下文分支） | **HDB** | `ultralytics/nn/Convmodules/db.py` |
| 2 | 纹理—结构—全局增强模块（三分支 + 通道注意力加权残差注入） | **TSG-EM** | `ultralytics/nn/Convmodules/fem.py` |
| 3 | 尺度自适应融合引用策略（重新指定颈部两个融合节点的引用对象） | **SAFR** | `ultralytics/cfg/models/26/yolo26s_db_fem_11_13.yaml` |



## 主要结果

训练协议（所有数字均在此协议下取得）：从零训练（不用预训练权重），`epochs=250 imgsz=640 batch=24 workers=4 seed=42 scale=0.5 cache=False`，`end2end=True reg_max=1`，优化器解析为 MuSGD。

| 数据集 | 配置 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| NEU-DET（6 类，1440/360） | YOLO26s 基线 | 0.711 | 0.371 |
| | **HED-YOLO** | **0.751** | **0.388** |
| GC10-DET（10 类，1834/458） | YOLO26s 基线 | 0.633 | 0.333 |
| | **HED-YOLO** | **0.674** | **0.348** |

参数量 9.95 M → 12.89 M，GFLOPs 11.26 → 12.50。

## 安装

本仓库**只包含本项目自己写的代码**，不含官方 Ultralytics 源码，因此以"覆盖层"的方式装到对应上游版本上：

```bash
pip install ultralytics==8.4.137
git clone <本仓库> && cd HED-YOLO
./apply.sh
```

`apply.sh` 会先校验已安装的上游版本号，不匹配则直接报错退出，避免覆盖到错误的版本；`DRY_RUN=1 ./apply.sh` 只预览将要复制的文件而不改动任何东西。

参考环境：Python ≥ 3.9、PyTorch ≥ 2.0（训练需 CUDA）。

## 复现

```bash
# 先修改 ultralytics/cfg/datasets/neu_det.yaml 里的 path
bash scripts/train/run_ablation3_3gpu.sh     # 主消融链
bash scripts/train/run_gc10det_4gpu.sh       # GC10-DET 跨数据集
python scripts/eval/robustness_test.py       # 16 种图像退化
```

⚠️ `workers` 会影响数据加载与增强的随机流，**保持 4** 才能复现论文数字。

## 许可

本仓库是 Ultralytics 的衍生作品，整体以 **AGPL-3.0** 发布。学术使用需保留许可证并引用上游；商业闭源需向 Ultralytics 申请商业许可。详见 [NOTICE.md](NOTICE.md) 与 [LICENSE](LICENSE)。数据集（NEU-DET、GC10-DET）与模型权重均未包含在本仓库中。
