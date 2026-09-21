# Modifications relative to upstream Ultralytics

**Upstream baseline:** [Ultralytics](https://github.com/ultralytics/ultralytics) **8.4.137**

**Modification surface:** five upstream files touched, two new module directories, plus new model and dataset configs.

This document lists exactly what was changed so that the code can be checked without diffing the whole tree. To reproduce the diff:

```bash
git clone --branch v8.4.137 https://github.com/ultralytics/ultralytics.git upstream
diff -r upstream/ultralytics ./ultralytics
```

---

## 1. Modified upstream files

### 1.1 `ultralytics/nn/tasks.py`

Registers the custom modules with the Ultralytics model parser:

- adds `from .Convmodules import *` and `from .SPPmodules import *` at module level;
- adds four frozensets inside the `parse_model()` body, next to the upstream `base_modules` / `repeat_modules`: `conv_modules` and `conv_repeat_modules` (custom convolution-type modules, the latter taking an `n` argument), `same_ch_modules` (output channels equal input channels, so no width scaling), and `spp_modules` (custom SPP variants);
- adds `parse_model` branches for the multi-input and head-variant modules.

One correctness detail: `C3k2_MRB` forces `legacy = False` so that MRB replaces only the bottleneck while keeping YOLO26's default depthwise-separable classification branch, avoiding a confound of roughly 1.3 M extra parameters.

### 1.2 `ultralytics/utils/loss.py`

Two independent additions.

**(a) An experimental anisotropic regression term (self-developed, disabled by default).** Adds a Wasserstein-distance term mixed into CIoU, with an anisotropic variant that modulates the centre term by the target's own aspect ratio. It is controlled entirely by environment variables, so the default behaviour is bit-identical to upstream:

| Variable | Default | Meaning |
|---|---|---|
| `YOLO_NWD_W` | `0` (off) | mixing weight of the Wasserstein term |
| `YOLO_NWD_C` | `0.1` | constant, adapted to Ultralytics' normalized coordinates |
| `YOLO_NWD_GAMMA` | `0` (off) | anisotropy strength |

**This term is not used in the paper** and does not affect any reported number. It is kept because it was developed during this project.

**(b) `o2o_topk` override for the one-to-one branch.** The per-GT positive-sample count of the one-to-one (inference) branch is read from a head attribute when present. Stock `Detect` has no such attribute, so behaviour is unchanged. It supports the diagnostic in **§3.2 of the paper**, where increasing positives per target from 1 to 3 collapses mAP@0.5:0.95 to 0.255.

### 1.3 `ultralytics/nn/modules/head.py`

**Comments only.** Adds explanatory comments documenting the YOLO26 head internals. No statement was added, removed or altered, so this file has **no functional difference** from upstream.

### 1.4 `ultralytics/utils/torch_utils.py`

EMA update guard. Models such as RT-DETR produce `float64` buffers during training, and `torch._foreach_lerp_` requires both operands to share a dtype:

```python
# before
if v.dtype.is_floating_point:
# after
if v.dtype in (torch.float16, torch.float32, torch.bfloat16) and msd[k].dtype == v.dtype:
```

The skipped entries are non-learnable buffers and do not affect the EMA.

### 1.5 `ultralytics/optim/muon.py`

`view` → `reshape`, functionally equivalent for contiguous tensors and safe for non-contiguous inputs:

```python
# before
m = u.view(len(u), -1) if u.ndim > 2 else u
# after
m = u.reshape(len(u), -1) if u.ndim > 2 else u
```

---

## 2. New directories

### 2.1 `ultralytics/nn/Convmodules/`

| File | Exported classes | Status |
|---|---|---|
| `db.py` | `Bottleneck_DB`, `C3k_DB`, `C3k2_DB` | **used in the paper — HDB (§4.2)** |
| `fem.py` | `FEM` (`TextureEnhance`, `StructurePreserve`, `GlobalContext`) | **used in the paper — TSG-EM (§4.3)** |
| `repblock.py` | `RepConvDiverse`, `Bottleneck_MRB`, `C3k_MRB`, `C3k2_MRB`, `rep_fuse_model` | **used in the paper — MRB lightweight variant (§5.11)** |
| `sampling.py` | `SPDConv`, `DySample` | **`SPDConv` used in the paper (§5.11)** |
| others | head-path, cross-scale fusion and single-module variants | explored, not used in the paper |

`db.py` keeps the official `C3k2` shell and interface signature (C2f topology, `cv1`/`cv2`), replacing only the inner bottleneck with the parallel dual-branch block, so a config can swap `C3k2` → `C3k2_DB` without touching any other argument.

### 2.2 `ultralytics/nn/SPPmodules/`

Custom SPP variants, explored and not used in the paper.

### 2.3 `ultralytics/cfg/`

- `models/26/` — the paper configs plus every explored variant for YOLO26s
- `models/11/`, `models/v9/` — cross-check runs for YOLOv11s and YOLOv9s
- `datasets/neu_det.yaml` (modified: `path` templated to `/path/to/NEU-DET`) and `datasets/gc10_det.yaml` (new)

The paper-relevant subset is listed in the README.

---

## 3. Deliberately unchanged

- **Detection head structure and training paradigm.** `end2end=True` and `reg_max=1` (native NMS-free inference and DFL-free regression) are preserved; the paper's central claim is that accuracy is raised without leaving that paradigm.
- **Loss function used in the reported experiments.** The experimental regression term is present but disabled, so all reported numbers were produced by the stock Ultralytics loss.
- **Version string.** `ultralytics.__version__` remains `8.4.137`.

---

## 4. Reproducibility notes

- Protocol for every reported number: `epochs=250 imgsz=640 batch=24 workers=4 seed=42 scale=0.5 cache=False`, from scratch (no pretrained weights), `optimizer=auto` → MuSGD.
- `workers` changes the data-loading and augmentation random stream — keep it at 4.
- HED-YOLO builds to 12.89 M parameters.
