# HED-YOLO

Research code for the paper:

> **NMS-free end-to-end detection for CPU-based steel surface inspection: quantifying and compensating YOLO26's accuracy cost**

HED-YOLO is built on **YOLO26s** (Ultralytics 8.4.137) and keeps its native **NMS-free end-to-end inference** and **DFL-free regression** unchanged. Three structural improvements are proposed for steel surface defect detection:

| # | Contribution | Abbrev. | Where in this repo |
|---|---|---|---|
| 1 | Heterogeneous dual-branch bottleneck (local-detail branch ‖ large-kernel separable-attention context branch) | **HDB** | [`ultralytics/nn/Convmodules/db.py`](ultralytics/nn/Convmodules/db.py) |
| 2 | Texture–Structure–Global enhancement module (three complementary branches + channel-attention weighted residual injection) | **TSG-EM** | [`ultralytics/nn/Convmodules/fem.py`](ultralytics/nn/Convmodules/fem.py) |
| 3 | Scale-adaptive fusion referencing (re-assigning the reference object of the two neck fusion nodes) | **SAFR** | [`yolo26s_db_fem_11_13.yaml`](ultralytics/cfg/models/26/yolo26s_db_fem_11_13.yaml) (yaml wiring only, no new parameters) |

The final model config is `yolo26s_db_fem_11_13.yaml`.

---

## Results

Protocol for **all** numbers below: from scratch (no pretrained weights), `epochs=250 imgsz=640 batch=24 workers=4 seed=42 scale=0.5 cache=False`, `end2end=True reg_max=1`, optimizer resolved to MuSGD.

### Detection accuracy

| Dataset | Config | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| NEU-DET (6 classes, 1440/360) | YOLO26s (baseline) | 0.711 | 0.371 |
| | + HDB | 0.727 | 0.373 |
| | + TSG-EM | 0.727 | 0.384 |
| | **HED-YOLO (all three)** | **0.751** | **0.388** |
| GC10-DET (10 classes, 1834/458) | YOLO26s (baseline) | 0.633 | 0.333 |
| | **HED-YOLO** | **0.674** | **0.348** |

Parameters 9.95 M → 12.89 M; GFLOPs 11.26 → 12.50.

---

## Installation

This repository contains **only the code written for this project**, not the upstream Ultralytics sources. It is applied as an overlay on a matching upstream release:

```bash
pip install ultralytics==8.4.137
git clone <this-repo> && cd HED-YOLO
./apply.sh
```

`apply.sh` verifies the installed upstream version and refuses to run against a mismatch, so the overlay cannot silently land on the wrong release. Use `DRY_RUN=1 ./apply.sh` to list the files it would copy without changing anything.

To do it by hand instead:

```bash
cp -r ultralytics/. "$(python -c 'import ultralytics,os;print(os.path.dirname(ultralytics.__file__))')/"
```

Verified with Python ≥ 3.9 and PyTorch ≥ 2.0 (CUDA for training). See [`MODIFICATIONS.md`](MODIFICATIONS.md) for the complete list of changes relative to upstream.

---

## Datasets

Both datasets are **public** and are **not redistributed** here.

**NEU-DET** — 6 classes, 1800 images @ 200×200 grayscale.
Prepare `images/{train,val}` and `labels/{train,val}` (1440/360 split), then set `path` in `ultralytics/cfg/datasets/neu_det.yaml`. A converter for the Kaggle/Roboflow variant is provided:

```bash
python scripts/data/convert_kaggle_neu.py --help
```

**GC10-DET** — 10 classes, 2300 images @ 2048×1000 grayscale.
Official source: <https://github.com/lvxiaomingld/GC10-DET>

```bash
python scripts/data/convert_gc10det.py --src /path/to/archive --dst /path/to/GC10-DET
```

The converter stratifies the split by image main-class folder (val ratio 0.2, seed 42) so that every class appears in the validation set.

---

## Reproducing the paper

```bash
# edit `path` in ultralytics/cfg/datasets/neu_det.yaml first
bash scripts/train/run_ablation3_3gpu.sh      # baseline / +HDB / +HDB+TSG-EM+SAFR
bash scripts/train/run_gc10det_4gpu.sh        # GC10-DET cross-dataset runs
```

The scripts auto-detect the repository root, or you can override it with `ROOT=/your/path`.

Equivalent single run:

```bash
yolo detect train \
  model=ultralytics/cfg/models/26/yolo26s_db_fem_11_13.yaml \
  data=ultralytics/cfg/datasets/neu_det.yaml \
  epochs=250 imgsz=640 batch=24 workers=4 seed=42 scale=0.5 cache=False
```

> `workers` affects the augmentation random stream. Keep it at 4 to reproduce the reported numbers.

The training shell scripts assume Linux with CUDA (they use `nohup`); the evaluation and figure scripts run on any platform.

Evaluation and figure scripts:

```bash
python scripts/eval/benchmark_models.py       # multi-detector comparison
python scripts/eval/robustness_test.py        # 16 image degradations
python scripts/eval/mechanism_analysis.py     # spectral prior + ERF control
python scripts/figures/make_paper_figures.py
python scripts/figures/make_robustness_curve.py
python scripts/figures/make_qualitative_grid.py
```

---

## Model configurations used in the paper

All under `ultralytics/cfg/models/26/`:

| File | Role |
|---|---|
| `yolo26s.yaml` | baseline (stock YOLO26s) |
| `yolo26s_db.yaml` | + HDB only |
| `yolo26s_fem_11_13.yaml` | + TSG-EM only |
| **`yolo26s_db_fem_11_13.yaml`** | **HED-YOLO (final)** |
| `yolo26s_db_fem_11_13_fix.yaml` | SAFR control: P4 references the refined output |
| `yolo26s_db_fem.yaml` | SAFR control: P5 references SPPF |
| `yolo26s_db_fem_11_13_noe2e.yaml` | head-path analysis (one-to-many + NMS) |
| `yolo26s_mrb.yaml`, `yolo26s_spd_all.yaml` | lightweight variants (MRB, SPD) |
| `yolo26s_*_gc10.yaml` | GC10-DET runs |

Relative to `yolo26s.yaml`, the final config changes three things: `C3k2` → `C3k2_DB` at backbone P3/P4/P5 and at the P5 head block (**HDB**); one `FEM [1024, 4]` inserted after backbone C2PSA and one after the P5 head block (**TSG-EM**); and the two neck fusion nodes take different reference objects — the P4 node takes the raw first-level cross-scale mix, the P5 node the attention-refined C2PSA output (**SAFR**).

---

## Repository layout

```
HED-YOLO/
├── ultralytics/            overlay files; paths mirror the upstream package
│   ├── nn/Convmodules/     custom modules (HDB, TSG-EM and explored variants)
│   ├── nn/SPPmodules/      custom SPP variants
│   ├── nn/tasks.py         modified: registers the custom modules
│   ├── nn/modules/head.py  modified: comments only
│   ├── utils/loss.py       modified: experimental loss term + o2o_topk hook
│   ├── utils/torch_utils.py, optim/muon.py   modified: small fixes
│   └── cfg/                model configs and dataset configs
├── scripts/
│   ├── data/               dataset preparation
│   ├── train/              training and ablation scripts
│   ├── eval/               benchmarks, robustness test, mechanism analysis
│   ├── figures/            figure generation
│   └── exploratory/        explored-but-unused directions
├── figures/                output directory for the figure scripts
├── apply.sh                applies the overlay to an installed ultralytics
├── MODIFICATIONS.md        exact diff relative to upstream 8.4.137
└── NOTICE.md, LICENSE      attribution, AGPL-3.0
```

`scripts/exploratory/` holds code for directions that were explored and rejected; it is kept for transparency. The loss file additionally contains an experimental anisotropic regression term that is disabled by default — see [`MODIFICATIONS.md`](MODIFICATIONS.md).

---

## License and citation

This repository is a derivative work of [Ultralytics](https://github.com/ultralytics/ultralytics) and is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md).

If you use this code, please cite the paper and the upstream project:

```bibtex
@article{qu2026hedyolo,
  title  = {NMS-free end-to-end detection for CPU-based steel surface inspection:
            quantifying and compensating YOLO26's accuracy cost},
  author = {Qu, Jinxin},
  year   = {2026}
}
```

```bibtex
@software{ultralytics2026yolo26,
  title  = {Ultralytics YOLO26: Unified real-time end-to-end vision models},
  author = {Ultralytics},
  year   = {2026},
  url    = {https://github.com/ultralytics/ultralytics}
}
```
