#!/usr/bin/env bash
# ============================================================================
# 第 10 轮：在 YOLO26 **原生协议**（NMS-free 端到端 + DFL-free）下继续找结构性涨点
#
# 第 9 轮留下的结论（决定了本轮的假设）：
#   r9_E     0.733/0.385   | r9_dra  0.731/0.387（+0.002，无效）
#   r9_wide  0.727/0.375（−0.010）| r9_o2o3 0.460/0.255（**崩塌 −0.130**）
#   → o2o3 崩塌证明："每 GT 单正样本"不是缺陷，而是 NMS-free 推理的**结构性前提**（多正样本 = 重复框无法去重）；
#     所以"推理头稀疏监督"假设作废，3.2 分差距是 NMS-free 的**固有代价**，不能靠修监督追回。
#   → 剩下的结构方向：① 尺度（第四检测层 P2，与官方 STAL 小目标分配互补）
#                     ② 容量重分配（P3/P4 检测分支加深）
#                     ③ 训练期辅助头容量（推理零成本：只有一对多的梯度能塑造共享特征）
#                     ④ 收敛性（500 epoch，之前被砍掉、从未在原生协议下测过）
#
# 本轮 4 卡（协议：end2end=True / reg_max=1 / 250ep / batch24 / imgsz640 / seed42 / cache=False）：
#   GPU0 r10_p2      yolo26s_p2.yaml                       四尺度（参照 vanilla 0.711/0.371）
#   GPU1 r10_wide34  yolo26s_db_fem_11_13_wide34.yaml      P3/P4 分支加深（参照 r9_E 0.733/0.385）
#   GPU2 r10_o2mw    yolo26s_db_fem_11_13_wide_o2m.yaml    只加宽训练用辅助头（参照 r9_E 0.385）
#   GPU3 r10_e500    yolo26s_db_fem_11_13.yaml + 500 epoch 收敛性（参照 r9_E 0.385，长跑约 1 小时）
#
# 用法：
#   cd /path/to/HED-YOLO
#   chmod +x run_round10_4gpu.sh && bash run_round10_4gpu.sh
#   监控：tail -f logs/r10_round_*.log
# ============================================================================
set -u
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # 自动定位仓库根目录；也可用 ROOT=/your/path bash xxx.sh 覆盖
cd "$ROOT" || { echo "找不到项目目录 $ROOT"; exit 1; }
mkdir -p logs runs

DATA=ultralytics/cfg/datasets/neu_det.yaml
CFG=ultralytics/cfg/models/26
TAIL="imgsz=640 batch=24 workers=8 seed=42 scale=0.5 cache=False"
STAMP=$(date +%m%d_%H%M)
ROUND_LOG="logs/r10_round_${STAMP}.log"

if ! grep -q "reshape(len(u), -1)" ultralytics/optim/muon.py 2>/dev/null; then
  echo "⚠️ 未检测到 muon.py 补丁（view → reshape），有概率训练中途崩；10 秒后继续。"
  sleep 10
fi
for f in yolo26s_p2.yaml yolo26s_db_fem_11_13_wide34.yaml yolo26s_db_fem_11_13_wide_o2m.yaml; do
  [ -f "$CFG/$f" ] || { echo "❌ 缺少 $CFG/$f（需上传）"; exit 1; }
done

# name|gpu|yaml|额外参数|参照值|说明
JOBS=(
  "r10_p2|0|${CFG}/yolo26s_p2.yaml|epochs=250|0.711/0.371|四尺度检测（加 P2/4）"
  "r10_wide34|1|${CFG}/yolo26s_db_fem_11_13_wide34.yaml|epochs=250|0.733/0.385|P3/P4 检测分支加深（n=2→4）"
  "r10_o2mw|2|${CFG}/yolo26s_db_fem_11_13_wide_o2m.yaml|epochs=250|0.733/0.385|只加宽训练用辅助头（推理零成本）"
  "r10_e500|3|${CFG}/yolo26s_db_fem_11_13.yaml|epochs=500|0.733/0.385|收敛性（500 epoch，长跑）"
)

{
  echo "================ [$(date '+%F %T')] 第10轮开始 ================"
  echo "协议：原生 NMS-free（end2end=True）+ DFL-free（reg_max=1），其余与历史逐字一致"
  for j in "${JOBS[@]}"; do
    IFS='|' read -r NAME GPU YAML EXTRA REF NOTE <<< "$j"
    printf "[%s] GPU%s ← %-10s 参照 %-12s %s\n" "$(date '+%H:%M:%S')" "$GPU" "$NAME" "$REF" "$NOTE"
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
  echo "================ 判定门（单 seed）================"
  echo "  r10_p2     vs 0.371  ：≥0.386 → 加入 P2 尺度"
  echo "  r10_wide34 vs 0.385  ：≥0.400 → P3/P4 容量重分配成立"
  echo "  r10_o2mw   vs 0.385  ：≥0.400 → 辅助头扩容成立（推理零成本，论文卖点强）"
  echo "  r10_e500   vs 0.385  ：≥0.400 → 250 epoch 欠收敛，延长训练"
  echo "  若四项全不过 → 论文主线定为：头部分析（NMS-free 代价 3.2 分，已验证 4σ）+ DB/FEM(+1.5) +"
  echo "                轻量化（MRB 折叠 −5.4%、SPD −10.6%）+ 噪声鲁棒性实验（2025-26 文献空缺）"
} 2>&1 | tee -a "$ROUND_LOG"

echo "全部完成。总日志：$ROUND_LOG"
