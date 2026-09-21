#!/usr/bin/env bash
# ============================================================================
# 第 5 轮（MRB 重参数化多分支 + ASFF 头输入自适应融合）——4 卡并行，单 seed
# 协议与上一轮完全一致：epochs=250 imgsz=640 batch=24 seed=42 scale=0.5 cache=False
# 每个配置独占一张卡（单卡单配置，与历史所有结果可比）
#
# 用法：
#   cd /path/to/HED-YOLO
#   chmod +x run_round5_mrb_asff_4gpu.sh
#   bash run_round5_mrb_asff_4gpu.sh            # 或 nohup bash run_round5_mrb_asff_4gpu.sh &
#   监控：tail -f logs/r5_round_*.log
# ============================================================================
set -u
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # 自动定位仓库根目录；也可用 ROOT=/your/path bash xxx.sh 覆盖
cd "$ROOT" || { echo "找不到项目目录 $ROOT"; exit 1; }
mkdir -p logs runs

DATA=ultralytics/cfg/datasets/neu_det.yaml
CFG=ultralytics/cfg/models/26
TAIL="epochs=250 imgsz=640 batch=24 workers=8 seed=42 scale=0.5 cache=False exist_ok=True"
STAMP=$(date +%m%d_%H%M)
ROUND_LOG="logs/r5_round_${STAMP}.log"

# name|gpu|yaml|说明
JOBS=(
  "r5_E|0|${CFG}/yolo26s_db_fem_11_13.yaml|E 基座复核（当前最好：0.743/0.388）"
  "r5_mrb|1|${CFG}/yolo26s_mrb.yaml|基线+MRB（第三创新点单模块行）"
  "r5_E_mrb|2|${CFG}/yolo26s_db_fem_11_13_mrb.yaml|E+MRB（叠加行）"
  "r5_E_asff|3|${CFG}/yolo26s_db_fem_11_13_asff.yaml|E+ASFF（备选叠加行）"
)

{
  echo "================ [$(date '+%F %T')] 第5轮开始（4 卡 × 单 seed 42）================"
  for j in "${JOBS[@]}"; do
    IFS='|' read -r NAME GPU YAML NOTE <<< "$j"
    printf "[%s] GPU%s ← %-24s %s\n" "$(date '+%H:%M:%S')" "$GPU" "$NAME" "$NOTE"
  done

  for j in "${JOBS[@]}"; do
    IFS='|' read -r NAME GPU YAML NOTE <<< "$j"
    nohup yolo detect train data="$DATA" model="$YAML" device="$GPU" \
      project="$ROOT/runs/$NAME" name="$NAME" $TAIL \
      > "logs/${NAME}.log" 2>&1 &
    echo "  started $NAME (pid=$!) → logs/${NAME}.log"
  done

  echo "---- 90 秒启动自检 ----"
  sleep 90
  FAIL=0
  for j in "${JOBS[@]}"; do
    IFS='|' read -r NAME GPU YAML NOTE <<< "$j"
    LOG="logs/${NAME}.log"
    if grep -qE "Traceback|ModuleNotFoundError|KeyError|TypeError|AssertionError" "$LOG"; then
      echo "  ❌ $NAME 启动报错（见 $LOG 末尾）"; tail -5 "$LOG"; FAIL=1
    elif grep -qE "Starting training for 250 epochs" "$LOG"; then
      echo "  ✅ $NAME 运行中"
    else
      echo "  ⏳ $NAME 启动中（未见报错）"
    fi
  done
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
  [ "$FAIL" = "1" ] && echo "⚠️ 有任务启动失败：先杀掉全部进程再排查（见文末命令）"
} 2>&1 | tee -a "$ROUND_LOG"

# 等待全部结束
while pgrep -f "yolo detect train" > /dev/null; do sleep 60; done

{
  echo
  echo "================ [$(date '+%F %T')] 训练结束，开始统一验证（best.pt）================"
  for j in "${JOBS[@]}"; do
    IFS='|' read -r NAME GPU YAML NOTE <<< "$j"
    BEST="$ROOT/runs/$NAME/$NAME/weights/best.pt"
    OUT="logs/${NAME}_val.txt"
    yolo detect val model="$BEST" data="$DATA" device=0 split=val > "$OUT" 2>&1
    echo
    echo "=== $NAME  （$NOTE）"
    grep -E "^ +(all|crazing|inclusion|patches|pitted_surface|rolled-in_scale|scratches) +[0-9]" "$OUT" | tail -7
    echo "  seed: $(grep -E '^seed' "$ROOT/runs/$NAME/$NAME/args.yaml" 2>/dev/null | awk '{print $2}')"
  done

  echo
  echo "================ 判定门（单 seed，参照同协议基线 0.713/0.370 与 E 基座 0.743/0.388）================"
  echo "  ✅ 明显涨点：候选 mAP50-95 ≥ 参照 +0.015，或 crazing+scratches 的 mAP50-95 各 +2.0 且聚合不跌"
  echo "  ⚠️ 边缘    ：+0.006 ~ +0.015 → 该候选单独复跑一次确认"
  echo "  ❌ 无变化  ：≤ +0.006 → 换方向（ASD 频带解耦 / 头输入 GSConv 轻量化 / 更长训练）"
  echo "  ⚙️ 推理态  ：MRB 模型可先跑 python verify_mrb_asff.py 看折叠后参数量（零推理开销证据）"
} 2>&1 | tee -a "$ROUND_LOG"

echo "全部完成。总日志：$ROUND_LOG"
