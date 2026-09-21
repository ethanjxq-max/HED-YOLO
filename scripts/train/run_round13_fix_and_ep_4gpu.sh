#!/usr/bin/env bash
# ============================================================================
# 第 13 轮：**SAFR 第三创新点的消融对照**（P4 节点：取精炼 14 vs 取原始 13）
#            + 同轮并行跑 EP 阶梯的完整版与前两级（第三创新点候选）
#            全部单 seed 42，协议与主消融链逐字一致
#
# 背景（2026-09-14 层号核查结论）：
#   E（yolo26s_db_fem_11_13）第 19 层写的是 Concat[-1,13] → 引用的是"第 13 层 Concat 的原始输出"，
#   而官方拓扑（含 vanilla 与 +DB 两行）引用的是"C3k2 精炼后的输出"。E 在 backbone 第 9 层插了 FEM，
#   层号整体 +1，所以官方的 13 在 E 里应写成 14（P5 路径的 11 已经在当年改对了）。
#   ⇒ 现在四行表里 ①② 用官方接线、③④ 用错位接线，E 的 +1.7 里混了两个变量，必须掰开：
#     · 若"修正版"更好 → 直接采用修正版（论文表格变干净，且可能白捡一个涨点）
#     · 若"错位版"更好 → 把"P4 节点跳跃式重连"作为一个显式的结构改动写进论文（第三创新点候选）
#
# 本轮 4 卡：
#   GPU0 r13_E_fix    yolo26s_db_fem_11_13_fix.yaml  参照 E(13) 0.751/0.388 → **对照组：P4 取精炼(14)**
#   GPU1 r13_fem_fix  yolo26s_fem_11_13_fix.yaml     参照 0.727/0.384（r11_fem, 13 版）→ 对照组：P4 取精炼(14)
#   GPU2 r13_ep       yolo26s_db_fem_ep.yaml         参照 0.751/0.388  → EP 完整版（FEM@P3+P4+P5，错位接线）
#   GPU3 r13_fem_p3   yolo26s_db_fem_p3.yaml         参照 0.751/0.388  → EP 剂量第一步（只加 P3）
#
# 用法：
#   cd /path/to/HED-YOLO
#   chmod +x run_round13_fix_and_ep_4gpu.sh && bash run_round13_fix_and_ep_4gpu.sh
#   监控：tail -f logs/r13_round_*.log
# ============================================================================
set -u
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # 自动定位仓库根目录；也可用 ROOT=/your/path bash xxx.sh 覆盖
cd "$ROOT" || { echo "找不到项目目录 $ROOT"; exit 1; }
mkdir -p logs runs

DATA=ultralytics/cfg/datasets/neu_det.yaml
CFG=ultralytics/cfg/models/26
TAIL="epochs=250 imgsz=640 batch=24 workers=4 seed=42 scale=0.5 cache=False"
STAMP=$(date +%m%d_%H%M)
ROUND_LOG="logs/r13_round_${STAMP}.log"

for f in yolo26s_db_fem_11_13_fix.yaml yolo26s_fem_11_13_fix.yaml yolo26s_db_fem_ep.yaml yolo26s_db_fem_p3.yaml; do
  [ -f "$CFG/$f" ] || { echo "❌ 缺少 $CFG/$f（需上传）"; exit 1; }
done

# name|gpu|yaml|参照|说明
JOBS=(
  "r13_E_fix|0|${CFG}/yolo26s_db_fem_11_13_fix.yaml|0.751/0.388|对照：P4 取精炼(14) vs E 的 13"
  "r13_fem_fix|1|${CFG}/yolo26s_fem_11_13_fix.yaml|0.727/0.384|对照：P4 取精炼(14) vs r11_fem 的 13"
  "r13_ep|2|${CFG}/yolo26s_db_fem_ep.yaml|0.751/0.388|EP 完整版：FEM@P3+P4+P5"
  "r13_fem_p3|3|${CFG}/yolo26s_db_fem_p3.yaml|0.751/0.388|EP 剂量第一步：只加 FEM@P3"
)

{
  echo "================ [$(date '+%F %T')] 第13轮（接线修正 + EP 阶梯）开始 ================"
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
  echo "  ① SAFR 消融（第三创新点）：r13_E_fix(14) vs E(13)=0.751/0.388"
  echo "     若 r13_E_fix 更低（预期）→ SAFR 成立，论文写「两个融合节点按尺度分工取原始/精炼」"
  echo "     同轮第二个对照：r13_fem_fix(14) vs r11_fem(13)=0.727/0.384"
  echo "  ② 历史同协议证据：P5 取 C2PSA(0.388) vs 取 SPPF(0.377) = +1.1 —— 两节点最优取法不同"
  echo "  ③ EP 阶梯：r13_ep / r13_fem_p3 vs 0.751/0.388，≥0.403 → EP 成立（第三创新点候选）"
} 2>&1 | tee -a "$ROUND_LOG"

echo "全部完成。总日志：$ROUND_LOG"
