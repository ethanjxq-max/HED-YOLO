#!/usr/bin/env bash
# ============================================================================
# 第 12 轮：第三创新点候选 = **EP 增强金字塔**（把已验证的 FEM 从 P5 扩到 P3/P4/P5）
#            4 卡 = 完整逐级消融阶梯（P3 / P4 / P3+P4）+ 单模块行（baseline+EP）
#            全部单 seed 42，协议与主消融链逐字一致；不跑 500 epoch
#
# 依据（第 11 轮）：FEM 单模块行 = 0.727/**0.384**（对 baseline 0.711/0.371，+1.6 mAP50 / +1.3 mAP50-95）
#   → 两个模块单独都有效（DB 偏 mAP50，FEM 偏 mAP50-95），但"再加容量/换尺度/换头"的候选全部失败
#     （P2 四尺度 0.377、P3P4 加深 0.380、辅助头加宽 0.372）。
#   → 剩下的正确打法是本项目历史上唯一 4/4 成功的模式：**已验证组件的尺度扩展**
#     （DB→更多层、FEM→更多位置、宽 P4、C2PSA@P5 都属于这一类）。EP 就是"FEM→更多位置"。
#
# 本轮 4 卡：
#   GPU0 r12_ep      yolo26s_db_fem_ep.yaml   参照 0.751/0.388  → EP 完整版（FEM@P3+P4+P5）
#   GPU1 r12_fem_p3  yolo26s_db_fem_p3.yaml   参照 0.751/0.388  → 只加 P3
#   GPU2 r12_fem_p4  yolo26s_db_fem_p4.yaml   参照 0.751/0.388  → 只加 P4
#   GPU3 r12_ep_base yolo26s_fem_ep.yaml      参照 0.711/0.371  → baseline + EP（单模块行，不含 DB）
#
# 判定门（单 seed）：r12_ep / r12_fem_p3 / r12_fem_p4 ≥ 0.403（+0.015）→ 第三创新点落地；
#                    r12_ep_base ≥ 0.386 → EP 单模块行成立（论文可写两条独立贡献行）；
#                    全部不过 → 论文主线：DB+FEM + NMS-free 头代价分析 + 轻量化（MRB/SPD）。
#
# 用法：
#   cd /path/to/HED-YOLO
#   chmod +x run_round12_ep_4gpu.sh && bash run_round12_ep_4gpu.sh
#   监控：tail -f logs/r12_round_*.log
# ============================================================================
set -u
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # 自动定位仓库根目录；也可用 ROOT=/your/path bash xxx.sh 覆盖
cd "$ROOT" || { echo "找不到项目目录 $ROOT"; exit 1; }
mkdir -p logs runs

DATA=ultralytics/cfg/datasets/neu_det.yaml
CFG=ultralytics/cfg/models/26
TAIL="imgsz=640 batch=24 workers=4 seed=42 scale=0.5 cache=False"
STAMP=$(date +%m%d_%H%M)
ROUND_LOG="logs/r12_round_${STAMP}.log"

for f in yolo26s_db_fem_ep.yaml yolo26s_db_fem_p3.yaml yolo26s_db_fem_p4.yaml yolo26s_fem_ep.yaml; do
  [ -f "$CFG/$f" ] || { echo "❌ 缺少 $CFG/$f（需上传）"; exit 1; }
done

# name|gpu|yaml|额外参数|参照|说明
JOBS=(
  "r12_ep|0|${CFG}/yolo26s_db_fem_ep.yaml|epochs=250|0.751/0.388|EP 增强金字塔（FEM@P3+P4+P5）"
  "r12_fem_p3|1|${CFG}/yolo26s_db_fem_p3.yaml|epochs=250|0.751/0.388|只加 FEM@P3"
  "r12_fem_p4|2|${CFG}/yolo26s_db_fem_p4.yaml|epochs=250|0.751/0.388|只加 FEM@P4"
  "r12_ep_base|3|${CFG}/yolo26s_fem_ep.yaml|epochs=250|0.711/0.371|baseline + EP（单模块行，不含 DB）"
)

{
  echo "================ [$(date '+%F %T')] 第12轮（EP 增强金字塔）开始 ================"
  echo "协议：imgsz=640 / batch=24 / workers=4 / seed=42 / scale=0.5 / cache=False / 从零训练"
  for j in "${JOBS[@]}"; do
    IFS='|' read -r NAME GPU YAML EXTRA REF NOTE <<< "$j"
    printf "[%s] GPU%s ← %-11s 参照 %-12s %s\n" "$(date '+%H:%M:%S')" "$GPU" "$NAME" "$REF" "$NOTE"
  done
  for j in "${JOBS[@]}"; do
    IFS='|' read -r NAME GPU YAML EXTRA REF NOTE <<< "$j"
    nohup yolo detect train data="$DATA" model="$YAML" device="$GPU" $TAIL $EXTRA \
      project="$ROOT/runs/$NAME" name="$NAME" > "logs/${NAME}.log" 2>&1 &
    echo "  started $NAME (pid=$!) → logs/${NAME}.log"
  done

  echo "---- 90 秒启动自检 ----"
  sleep 90
  FAIL=0
  for j in "${JOBS[@]}"; do
    IFS='|' read -r NAME GPU YAML EXTRA REF NOTE <<< "$j"
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
    IFS='|' read -r NAME GPU YAML EXTRA REF NOTE <<< "$j"
    OUT="logs/${NAME}_val.txt"
    yolo detect val model="$ROOT/runs/$NAME/$NAME/weights/best.pt" data="$DATA" device=0 split=val > "$OUT" 2>&1
    echo
    echo "=== $NAME  （参照 $REF ；$NOTE）"
    grep -E "^ +(all|crazing|inclusion|patches|pitted_surface|rolled-in_scale|scratches) +[0-9]" "$OUT" | tail -7
  done

  echo
  echo "================ 判定门（单 seed 42）================"
  echo "  r12_ep      vs 0.751/0.388 ：mAP50-95 ≥0.403 → 第三创新点（EP）落地"
  echo "  r12_fem_p3  vs 0.751/0.388 ：mAP50-95 ≥0.403 → 只加 P3 也成立（更省）"
  echo "  r12_fem_p4  vs 0.751/0.388 ：mAP50-95 ≥0.403 → 只加 P4 也成立"
  echo "  r12_ep_base vs 0.711/0.371 ：mAP50-95 ≥0.386 → EP 单模块行成立（两条独立贡献行）"
  echo "  全不过 → 论文主线定稿：DB+FEM(+1.7) + NMS-free 头代价分析(3.2, 4σ) + 轻量化(MRB −5.4% / SPD −10.6%)"
} 2>&1 | tee -a "$ROUND_LOG"

echo "全部完成。总日志：$ROUND_LOG"
