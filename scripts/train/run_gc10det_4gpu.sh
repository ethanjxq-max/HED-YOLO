#!/usr/bin/env bash
# ============================================================================
# GC10-DET 泛化实验（4 卡，单 seed 42，协议与 NEU-DET 主消融链逐字一致）
#   数据集：GC10-DET（钢带表面，10 类；本地转换后 train 1834 / val 458）
#   协议：epochs=250 / imgsz=640 / batch=24 / workers=4 / seed=42 / scale=0.5 / cache=False / 从零训练
#
#   行的意义（和 NEU-DET 的表一一对应，用于"泛化性"章节）：
#     GPU0 baseline            yolo26s_gc10.yaml                  ← 与 NEU-DET 的 0.711/0.371 对应
#     GPU1 +DB                 yolo26s_db_gc10.yaml
#     GPU2 +FEM                yolo26s_fem_11_13_gc10.yaml        ← 单模块行
#     GPU3 +DB+FEM（最终模型）  yolo26s_db_fem_11_13_gc10.yaml     ← 与 NEU-DET 的 0.751/0.388 对应
#
# 前置（必须先做）：
#   1) 上传 archive/ 到项目根目录，然后执行：
#        python convert_gc10det.py --src archive --dst GC10-DET
#      期望输出：train 1834 / val 458，10 类实例数见脚本打印
#   2) 确认 GC10-DET/gc10det.yaml 里的 path 是服务器上的绝对路径
#
# 用法：
#   cd /path/to/HED-YOLO
#   chmod +x run_gc10det_4gpu.sh && bash run_gc10det_4gpu.sh
#   监控：tail -f logs/gc10_round_*.log
# ============================================================================
set -u
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # 自动定位仓库根目录；也可用 ROOT=/your/path bash xxx.sh 覆盖
cd "$ROOT" || { echo "找不到项目目录 $ROOT"; exit 1; }
mkdir -p logs runs

DATA=ultralytics/cfg/datasets/gc10det.yaml   # 用户放在 cfg/datasets（与 neu_det.yaml 同目录，惯例位置）
CFG=ultralytics/cfg/models/26
TAIL="epochs=250 imgsz=640 batch=24 workers=4 seed=42 scale=0.5 cache=False"
STAMP=$(date +%m%d_%H%M)
ROUND_LOG="logs/gc10_round_${STAMP}.log"

[ -f "$DATA" ] || { echo "❌ 缺少 $DATA —— 确认 gc10det.yaml 在 ultralytics/cfg/datasets/ 下，且里面的 path 指向 $ROOT/GC10-DET"; exit 1; }
for f in yolo26s_gc10.yaml yolo26s_db_gc10.yaml yolo26s_fem_11_13_gc10.yaml yolo26s_db_fem_11_13_gc10.yaml; do
  [ -f "$CFG/$f" ] || { echo "❌ 缺少 $CFG/$f（需上传）"; exit 1; }
done

# name|gpu|yaml|说明
JOBS=(
  "gc10_base|0|${CFG}/yolo26s_gc10.yaml|① baseline"
  "gc10_db|1|${CFG}/yolo26s_db_gc10.yaml|② + C3k2-DB"
  "gc10_fem|2|${CFG}/yolo26s_fem_11_13_gc10.yaml|③ + FEM@11,13"
  "gc10_final|3|${CFG}/yolo26s_db_fem_11_13_gc10.yaml|④ + DB + FEM（最终模型）"
)

{
  echo "================ [$(date '+%F %T')] GC10-DET 泛化实验开始（4 卡 × 单 seed 42）================"
  echo "协议：epochs=250 / imgsz=640 / batch=24 / workers=4 / seed=42 / scale=0.5 / cache=False / 从零训练"
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
    IFS='|' read -r NAME GPU YAML NOTE <<< "$j"
    OUT="logs/${NAME}_val.txt"
    yolo detect val model="$ROOT/runs/$NAME/$NAME/weights/best.pt" data="$DATA" device=0 split=val > "$OUT" 2>&1
    echo
    echo "=== $NAME  （$NOTE）"
    grep -E "^ +[A-Za-z0-9_]+ +[0-9]+ +[0-9]+ +[0-9]" "$OUT" | tail -11
  done

  echo
  echo "================ 泛化性对表（与 NEU-DET 同协议）================"
  echo "  NEU-DET : baseline 0.711/0.371 → +DB 0.727/0.373 → +FEM 0.727/0.384 → 最终 0.751/0.388"
  echo "  读法：若 GC10-DET 上 ④ > ①（且 ②③ 同向），说明三个创新点不是 NEU-DET 特调，泛化性成立"
  echo "  注意：8_yahen(压痕)/9_zhehen(折痕) 在 val 里实例很少（7/12 个框），这两类的 AP 噪声大，"
  echo "        论文里可只报 aggregate + 主要类别，或注明样本量。"
} 2>&1 | tee -a "$ROUND_LOG"

echo "全部完成。总日志：$ROUND_LOG"
