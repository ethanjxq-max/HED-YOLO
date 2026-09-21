#!/usr/bin/env bash
# ============================================================================
# 第 11 轮：FEM 单模块行（论文必做）+ 第三创新点三个候选（全部单 seed 42）
#   ⚠ 协议纪律（用户 2026-09-14 决定）：**不做多种子**，所有行一律单 seed 42，论文不报 mean±std。
#
# 参照值（同协议、同机、单 seed 42、workers=4、从零训练）：
#   baseline            0.711 / 0.371   （与历史值逐位复现）
#   +C3k2-DB            0.727 / 0.373
#   +DB+FEM@11,13       0.751 / 0.388   ← 当前最好（论文主模型）
#
# 本轮 4 卡：
#   GPU0 r11_fem     yolo26s_fem_11_13.yaml                 参照 0.711/0.371 → FEM 单模块行（论文行）
#   GPU1 r11_p2      yolo26s_p2.yaml                        参照 0.711/0.371 → 第四尺度 P2/4
#   GPU2 r11_wide34  yolo26s_db_fem_11_13_wide34.yaml       参照 0.751/0.388 → P3/P4 分支加深
#   GPU3 r11_o2mw    yolo26s_db_fem_11_13_wide_o2m.yaml     参照 0.751/0.388 → 只加宽训练用辅助头（推理零成本）
#
# 第二轮（可选）：`yolo26s_db_fem_11_13.yaml` + `epochs=500`（收敛性，长跑约 1 小时）
#
# 用法：
#   cd /path/to/HED-YOLO
#   chmod +x run_round11_4gpu.sh && bash run_round11_4gpu.sh
#   监控：tail -f logs/r11_round_*.log
# ============================================================================
set -u
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # 自动定位仓库根目录；也可用 ROOT=/your/path bash xxx.sh 覆盖
cd "$ROOT" || { echo "找不到项目目录 $ROOT"; exit 1; }
mkdir -p logs runs

DATA=ultralytics/cfg/datasets/neu_det.yaml
CFG=ultralytics/cfg/models/26
TAIL="epochs=250 imgsz=640 batch=24 workers=4 seed=42 scale=0.5 cache=False"
STAMP=$(date +%m%d_%H%M)
ROUND_LOG="logs/r11_round_${STAMP}.log"

if ! grep -q "reshape(len(u), -1)" ultralytics/optim/muon.py 2>/dev/null; then
  echo "⚠️ 未检测到 muon.py 补丁（view → reshape），有概率训练中途崩；10 秒后继续。"; sleep 10
fi
for f in yolo26s_fem_11_13.yaml yolo26s_p2.yaml yolo26s_db_fem_11_13_wide34.yaml yolo26s_db_fem_11_13_wide_o2m.yaml; do
  [ -f "$CFG/$f" ] || { echo "❌ 缺少 $CFG/$f（需上传）"; exit 1; }
done

# name|gpu|yaml|参照|说明
JOBS=(
  "r11_fem|0|${CFG}/yolo26s_fem_11_13.yaml|0.711/0.371|FEM 单模块行（不含 DB）——论文必做"
  "r11_p2|1|${CFG}/yolo26s_p2.yaml|0.711/0.371|四尺度检测（加 P2/4）"
  "r11_wide34|2|${CFG}/yolo26s_db_fem_11_13_wide34.yaml|0.751/0.388|P3/P4 检测分支加深（n=2→4）"
  "r11_o2mw|3|${CFG}/yolo26s_db_fem_11_13_wide_o2m.yaml|0.751/0.388|只加宽训练用辅助头（推理零成本）"
)

{
  echo "================ [$(date '+%F %T')] 第11轮开始（单 seed 42，不做多种子）================"
  echo "协议：epochs=250 / imgsz=640 / batch=24 / workers=4 / seed=42 / scale=0.5 / cache=False / 从零训练"
  for j in "${JOBS[@]}"; do
    IFS='|' read -r NAME GPU YAML REF NOTE <<< "$j"
    printf "[%s] GPU%s ← %-12s 参照 %-12s %s\n" "$(date '+%H:%M:%S')" "$GPU" "$NAME" "$REF" "$NOTE"
  done
  for j in "${JOBS[@]}"; do
    IFS='|' read -r NAME GPU YAML REF NOTE <<< "$j"
    nohup yolo detect train data="$DATA" model="$YAML" device="$GPU" $TAIL \
      project="$ROOT/runs/$NAME" name="$NAME" > "logs/${NAME}.log" 2>&1 &
    echo "  started $NAME (pid=$!) → logs/${NAME}.log"
  done

  echo "---- 90 秒启动自检 ----"
  sleep 90
  FAIL=0
  for j in "${JOBS[@]}"; do
    IFS='|' read -r NAME GPU YAML REF NOTE <<< "$j"
    LOG="logs/${NAME}.log"
    if grep -qE "Traceback|ModuleNotFoundError|KeyError|TypeError|OutOfMemoryError|AssertionError" "$LOG"; then
      echo "  ❌ $NAME 启动报错："; tail -6 "$LOG"; FAIL=1
    elif grep -qE "Starting training for 250 epochs" "$LOG"; then
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
    IFS='|' read -r NAME GPU YAML REF NOTE <<< "$j"
    OUT="logs/${NAME}_val.txt"
    yolo detect val model="$ROOT/runs/$NAME/$NAME/weights/best.pt" data="$DATA" device=0 split=val > "$OUT" 2>&1
    echo
    echo "=== $NAME  （参照 $REF ；$NOTE）"
    grep -E "^ +(all|crazing|inclusion|patches|pitted_surface|rolled-in_scale|scratches) +[0-9]" "$OUT" | tail -7
  done

  echo
  echo "================ 判定门（单 seed 42）================"
  echo "  r11_fem    vs 0.711/0.371 ：mAP50-95 ≥0.381 → FEM 可写独立贡献行；否则按链式消融（baseline→+DB→+DB+FEM）"
  echo "  r11_p2     vs 0.711/0.371 ：mAP50-95 ≥0.386 → P2 尺度成立"
  echo "  r11_wide34 vs 0.751/0.388 ：mAP50-95 ≥0.403 → P3/P4 容量重分配成立"
  echo "  r11_o2mw   vs 0.751/0.388 ：mAP50-95 ≥0.403 → 辅助头扩容成立（推理零成本，卖点强）"
  echo "  全不过 → 论文主线：DB+FEM（+1.7，可复现）+ NMS-free 头代价分析（3.2，4σ）+ 轻量化（MRB/SPD）"
} 2>&1 | tee -a "$ROUND_LOG"

echo "全部完成。总日志：$ROUND_LOG"
