#!/usr/bin/env bash
# ============================================================================
# 第 9 轮：在 YOLO26 **原生 NMS-free 端到端头**上做结构增强（不改范式、不恢复 DFL）
#
# 为什么这样设计（2026-09-14 方向修正）：
#   YOLO26 的立身之本 = ①NMS-free 端到端推理 ②DFL-free 回归 ③MuSGD + Progressive Loss + STAL
#   配方 ④部署效率（官方论文 arXiv:2606.03748，docs/en/models/yolo26.md）。
#   → 第 6 轮的 end2end=False（0.757/0.402）与第 8 轮设想的 reg_max=16 都是**把 YOLO26 改回
#     YOLOv8/11**，不能作为论文协议；它们只保留为"诊断/分析行"（量化 NMS-free 头在小数据上的代价）。
#   → 真实结论：**差距在推理头本身**（一对一头训练时读 detach 特征 + 每 GT 仅 1 个正样本 +
#     一对多监督权重 0.8→0.1 衰减）。因此第三创新点应该落在"推理头的结构增强"上。
#
# 本轮 4 卡（协议与历史完全一致：end2end=True / reg_max=1 / 250ep / batch24 / seed42 / cache=False）：
#   GPU0 r9_E     yolo26s_db_fem_11_13              本轮参照（E 基座，历史 0.743/0.388）
#   GPU1 r9_dra   yolo26s_db_fem_11_13_dra          +DRA 推理头细化适配器（只作用于一对一头，+0.355M）
#   GPU2 r9_wide  yolo26s_db_fem_11_13_wide_o2o     +推理头容量重分配（只加宽一对一头，+0.612M）
#   GPU3 r9_o2o3  yolo26s_db_fem_11_13_o2o3         +每 GT 正样本 1→3（**仅诊断分配假设**）
#
# 用法：
#   cd /path/to/HED-YOLO
#   chmod +x run_round9_headplus_4gpu.sh && bash run_round9_headplus_4gpu.sh
#   监控：tail -f logs/r9_round_*.log
# ============================================================================
set -u
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # 自动定位仓库根目录；也可用 ROOT=/your/path bash xxx.sh 覆盖
cd "$ROOT" || { echo "找不到项目目录 $ROOT"; exit 1; }
mkdir -p logs runs

DATA=ultralytics/cfg/datasets/neu_det.yaml
CFG=ultralytics/cfg/models/26
TAIL="epochs=250 imgsz=640 batch=24 workers=8 seed=42 scale=0.5 cache=False"
STAMP=$(date +%m%d_%H%M)
ROUND_LOG="logs/r9_round_${STAMP}.log"

if ! grep -q "reshape(len(u), -1)" ultralytics/optim/muon.py 2>/dev/null; then
  echo "⚠️ 未检测到 muon.py 补丁（view → reshape），有概率训练中途崩，建议先打补丁；10 秒后继续。"
  sleep 10
fi
for f in yolo26s_db_fem_11_13_dra.yaml yolo26s_db_fem_11_13_wide_o2o.yaml yolo26s_db_fem_11_13_o2o3.yaml; do
  [ -f "$CFG/$f" ] || { echo "❌ 缺少 $CFG/$f（需上传）"; exit 1; }
done
grep -q "DetectPlus" ultralytics/nn/tasks.py || { echo "❌ tasks.py 未注册 DetectPlus（需上传 tasks.py）"; exit 1; }

JOBS=(
  "r9_E|0|${CFG}/yolo26s_db_fem_11_13.yaml|E 基座（同轮参照）"
  "r9_dra|1|${CFG}/yolo26s_db_fem_11_13_dra.yaml|+DRA 推理头细化适配器"
  "r9_wide|2|${CFG}/yolo26s_db_fem_11_13_wide_o2o.yaml|+推理头容量重分配"
  "r9_o2o3|3|${CFG}/yolo26s_db_fem_11_13_o2o3.yaml|+每 GT 正样本 1→3（诊断）"
)

{
  echo "================ [$(date '+%F %T')] 第9轮（原生 NMS-free 头上的结构增强）开始 ================"
  echo "协议：end2end=True / reg_max=1 / 250ep / batch24 / imgsz640 / seed42 / cache=False"
  for j in "${JOBS[@]}"; do
    IFS='|' read -r NAME GPU YAML NOTE <<< "$j"
    printf "[%s] GPU%s ← %-10s %s\n" "$(date '+%H:%M:%S')" "$GPU" "$NAME" "$NOTE"
  done
  for j in "${JOBS[@]}"; do
    IFS='|' read -r NAME GPU YAML NOTE <<< "$j"
    nohup yolo detect train data="$DATA" model="$YAML" device="$GPU" $TAIL \
      project="$ROOT/runs/$NAME" name="$NAME" > "logs/${NAME}.log" 2>&1 &
    echo "  started $NAME (pid=$!) → logs/${NAME}.log"
  done

  echo "---- 90 秒启动自检 ----"
  sleep 90
  FAIL=0
  for j in "${JOBS[@]}"; do
    IFS='|' read -r NAME GPU YAML NOTE <<< "$j"
    LOG="logs/${NAME}.log"
    if grep -qE "Traceback|ModuleNotFoundError|KeyError|TypeError|OutOfMemoryError|AssertionError" "$LOG"; then
      echo "  ❌ $NAME 启动报错："; tail -6 "$LOG"; FAIL=1
    elif grep -qE "Starting training for" "$LOG"; then
      echo "  ✅ $NAME 运行中"
    else
      echo "  ⏳ $NAME 启动中（未见报错）"
    fi
  done
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
  [ "$FAIL" = "1" ] && echo "⚠️ 有任务启动失败：pkill -f 'yolo detect train'，然后看 logs/ 对应 *.log 末尾"
} 2>&1 | tee -a "$ROUND_LOG"

while pgrep -f "yolo detect train" > /dev/null; do sleep 60; done

{
  echo
  echo "================ [$(date '+%F %T')] 训练结束，统一验证（best.pt）================"
  for j in "${JOBS[@]}"; do
    IFS='|' read -r NAME GPU YAML NOTE <<< "$j"
    OUT="logs/${NAME}_val.txt"
    yolo detect val model="$ROOT/runs/$NAME/$NAME/weights/best.pt" data="$DATA" device=0 split=val > "$OUT" 2>&1
    echo
    echo "=== $NAME  （$NOTE）"
    grep -E "^ +(all|crazing|inclusion|patches|pitted_surface|rolled-in_scale|scratches) +[0-9]" "$OUT" | tail -7
  done

  echo
  echo "================ 判定门（单 seed，本轮内部对 r9_E）================"
  echo "  ✅ 明显涨点：mAP50-95 ≥ r9_E + 0.015，或 crazing/scratches 的 mAP50-95 各 +2.0 且聚合不跌"
  echo "     → 该结构成为第三创新点（论文协议保持 NMS-free + DFL-free 不变）"
  echo "  ⚠️ 边缘：+0.006~+0.015 → 单独复跑一次"
  echo "  ❌ ≤ +0.006 → 放弃该结构"
  echo
  echo "  诊断行 r9_o2o3 的读法（它本身不能写进论文，属分配侧改动）："
  echo "     · 若不涨 → 差距在'推理头容量/表达'侧 → DRA / wide 方向正确，继续加码结构"
  echo "     · 若大涨 → 差距在'稀疏监督（分配）'侧 → 论文改走「头部分析 + 轻量化/鲁棒性」主线"
  echo "       （MRB 折叠 -5.4% 参数/FLOPs、SPD -10.6% 参数、NMS-free 头代价 3.2 分的机制分析）"
} 2>&1 | tee -a "$ROUND_LOG"

echo "全部完成。总日志：$ROUND_LOG"
