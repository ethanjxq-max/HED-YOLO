#!/usr/bin/env bash
# ============================================================================
# 三条主消融链重跑（3 卡并行，单 seed 42）—— 论文里的核心消融表
#   GPU0  yolo26s.yaml                 历史参照 0.711 / 0.371
#   GPU1  yolo26s_db.yaml              历史参照 0.733 / 0.377
#   GPU2  yolo26s_db_fem_11_13.yaml    历史参照 0.743 / 0.388
#
# 协议与历史命令逐字一致（只把 project 换成当前服务器的绝对路径）：
#   epochs=250 imgsz=640 batch=24 workers=4 seed=42 scale=0.5  （不加 pretrained，从零训练）
#   cache=False 显式写出（历史默认就是 False，这里写明是为了可复现）
#   ⚠ workers 保持 4：worker 数会改变数据加载/增强的随机流，换 8 可能让结果轻微漂移
#
# 关于 loss.py / muon.py 的改动（**不需要回退**）：
#   · loss.py 只是加了 `getattr(head, "o2o_topk", 1)` 读取，官方 Detect 没有该属性 ⇒ 恒为 1，
#     与原始代码完全一致（本地已验证：三条配置的 E2ELoss 都是 one2one.topk2=1 / one2many.topk2=10）；
#   · muon.py 的 view→reshape 在连续张量下与 view 逐位等价（实测最大差 0.0），不影响任何数值；
#   · 其余改动（DetectPlus / ASFF / C3k2_MRB 等）只在这些模块被 yaml 引用时才生效，三条配置用不到。
#
# 用法：
#   cd /path/to/HED-YOLO
#   chmod +x run_ablation3_3gpu.sh && bash run_ablation3_3gpu.sh
#   监控：tail -f logs/ab_round_*.log
# ============================================================================
set -u
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # 自动定位仓库根目录；也可用 ROOT=/your/path bash xxx.sh 覆盖
cd "$ROOT" || { echo "找不到项目目录 $ROOT"; exit 1; }
mkdir -p logs runs

DATA=ultralytics/cfg/datasets/neu_det.yaml
CFG=ultralytics/cfg/models/26
TAIL="epochs=250 imgsz=640 batch=24 workers=4 seed=42 scale=0.5 cache=False"
STAMP=$(date +%m%d_%H%M)
ROUND_LOG="logs/ab_round_${STAMP}.log"

for f in yolo26s.yaml yolo26s_db.yaml yolo26s_db_fem_11_13.yaml; do
  [ -f "$CFG/$f" ] || { echo "❌ 缺少 $CFG/$f"; exit 1; }
done

# name|gpu|yaml|历史参照|说明
JOBS=(
  "ab_base|0|${CFG}/yolo26s.yaml|0.711/0.371|① baseline（纯 YOLO26s）"
  "ab_db|1|${CFG}/yolo26s_db.yaml|0.733/0.377|② + C3k2-DB"
  "ab_db_fem|2|${CFG}/yolo26s_db_fem_11_13.yaml|0.743/0.388|③ + DB + FEM@11,13（最佳）"
)

{
  echo "================ [$(date '+%F %T')] 三条主消融链重跑（3 卡 × seed 42）================"
  echo "协议：epochs=250 / imgsz=640 / batch=24 / workers=4 / seed=42 / scale=0.5 / cache=False / 从零训练"
  for j in "${JOBS[@]}"; do
    IFS='|' read -r NAME GPU YAML REF NOTE <<< "$j"
    printf "[%s] GPU%s ← %-10s 历史 %-12s %s\n" "$(date '+%H:%M:%S')" "$GPU" "$NAME" "$REF" "$NOTE"
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
      OPT=$(grep -m1 "with parameter groups" "$LOG" | cut -c1-60)
      echo "  ✅ $NAME 运行中   ${OPT:+[$OPT]}"
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
    echo "=== $NAME  （$NOTE ；历史 $REF）"
    grep -E "^ +(all|crazing|inclusion|patches|pitted_surface|rolled-in_scale|scratches) +[0-9]" "$OUT" | tail -7
  done

  echo
  echo "================ 消融链对表（同轮同机，单 seed 42）================"
  echo "  行① baseline       历史 0.711/0.371  → 本轮见 ab_base"
  echo "  行② +C3k2-DB       历史 0.733/0.377  → 贡献 = 行② − 行①（历史 +0.6 mAP50-95）"
  echo "  行③ +FEM@11,13     历史 0.743/0.388  → 贡献 = 行③ − 行②（历史 +1.1 mAP50-95）"
  echo "  判定：两条贡献都为正且合计 ≥ +1.0 mAP50-95 ⇒ 论文消融表成立；"
  echo "        随后再对最终模型与 baseline 各补 2 个种子（seed 123/2024）报均值±std。"
} 2>&1 | tee -a "$ROUND_LOG"

echo "全部完成。总日志：$ROUND_LOG"
