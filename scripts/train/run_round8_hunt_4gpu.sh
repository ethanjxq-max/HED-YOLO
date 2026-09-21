#!/usr/bin/env bash
# ============================================================================
# 第 8 轮：在"高天花板头"（一对多 + NMS）下继续找真实涨点
#   参照：r7_base = yolo26s_noe2e = 0.757 / 0.402（同机同协议 seed42）
#   第 7 轮结论：DB+FEM(0.394)、MRB(0.382)、ASFF(0.390) 全部没过门 → 模块在新头下无效
#   本轮换四个"机制不同、都有大效应潜力"的杠杆，全部对着实测短板（定位弱 + crazing/rolled 低对比）：
#     GPU0 DFL 分布回归头（reg_max 1→16）——mAP50 与 mAP50-95 差 0.355，定位是最大缺口
#     GPU1 P2/4 第四检测尺度            ——pitted/crazing/细划痕的小目标尺度
#     GPU2 COCO 预训练初始化（99% 迁移）——e2e 头下无增益，新头下重测（文献数字全是预训练）
#     GPU3 P3/P4 检测分支加深（n=2→4）  ——把容量加到缺陷所在的尺度（早期"宽P4 +2.6"的重测）
#
# 用法：
#   cd /path/to/HED-YOLO
#   chmod +x run_round8_hunt_4gpu.sh && bash run_round8_hunt_4gpu.sh
#   监控：tail -f logs/r8_round_*.log
# ============================================================================
set -u
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # 自动定位仓库根目录；也可用 ROOT=/your/path bash xxx.sh 覆盖
cd "$ROOT" || { echo "找不到项目目录 $ROOT"; exit 1; }
mkdir -p logs runs

DATA=ultralytics/cfg/datasets/neu_det.yaml
CFG=ultralytics/cfg/models/26
TAIL="epochs=250 imgsz=640 batch=24 workers=8 seed=42 scale=0.5 cache=False"
STAMP=$(date +%m%d_%H%M)
ROUND_LOG="logs/r8_round_${STAMP}.log"

if ! grep -q "reshape(len(u), -1)" ultralytics/optim/muon.py 2>/dev/null; then
  echo "⚠️ 未检测到 muon.py 补丁（view → reshape）。建议先打补丁（命令见方案文档第九节），10 秒后继续。"
  sleep 10
fi
for f in yolo26s_noe2e_dfl.yaml yolo26s_noe2e_p2.yaml yolo26s_noe2e_wide34.yaml; do
  [ -f "$CFG/$f" ] || { echo "❌ 缺少 $CFG/$f（需上传）"; exit 1; }
done
[ -f "$ROOT/yolo26s.pt" ] || { echo "❌ 缺少 $ROOT/yolo26s.pt（GPU2 需要）"; exit 1; }

# name|gpu|yaml|额外参数|说明
JOBS=(
  "r8_dfl|0|${CFG}/yolo26s_noe2e_dfl.yaml||DFL 分布回归头（reg_max=16）"
  "r8_p2|1|${CFG}/yolo26s_noe2e_p2.yaml||四尺度检测（加 P2/4）"
  "r8_pt|2|${CFG}/yolo26s_noe2e.yaml|pretrained=$ROOT/yolo26s.pt|COCO 预训练（新头下）"
  "r8_wide34|3|${CFG}/yolo26s_noe2e_wide34.yaml||P3/P4 检测分支加深（n=2→4）"
)

{
  echo "================ [$(date '+%F %T')] 第8轮（新头下找涨点）开始 ================"
  echo "参照：r7_base = 0.757/0.402（yolo26s_noe2e，同机同协议）  过门线 mAP50-95 ≥ 0.417"
  for j in "${JOBS[@]}"; do
    IFS='|' read -r NAME GPU YAML EXTRA NOTE <<< "$j"
    printf "[%s] GPU%s ← %-10s %s\n" "$(date '+%H:%M:%S')" "$GPU" "$NAME" "$NOTE"
  done
  for j in "${JOBS[@]}"; do
    IFS='|' read -r NAME GPU YAML EXTRA NOTE <<< "$j"
    nohup yolo detect train data="$DATA" model="$YAML" device="$GPU" $TAIL $EXTRA \
      project="$ROOT/runs/$NAME" name="$NAME" > "logs/${NAME}.log" 2>&1 &
    echo "  started $NAME (pid=$!) → logs/${NAME}.log"
  done

  echo "---- 90 秒启动自检 ----"
  sleep 90
  FAIL=0
  for j in "${JOBS[@]}"; do
    IFS='|' read -r NAME GPU YAML EXTRA NOTE <<< "$j"
    LOG="logs/${NAME}.log"
    if grep -qE "Traceback|ModuleNotFoundError|KeyError|TypeError|OutOfMemoryError|AssertionError" "$LOG"; then
      echo "  ❌ $NAME 启动报错："; tail -6 "$LOG"; FAIL=1
    elif grep -qE "Starting training for" "$LOG"; then
      TR=$(grep -oE "Transferred [0-9]+/[0-9]+ items" "$LOG" | tail -1)
      echo "  ✅ $NAME 运行中 ${TR:+(预训练迁移 $TR)}"
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
    IFS='|' read -r NAME GPU YAML EXTRA NOTE <<< "$j"
    OUT="logs/${NAME}_val.txt"
    yolo detect val model="$ROOT/runs/$NAME/$NAME/weights/best.pt" data="$DATA" device=0 split=val > "$OUT" 2>&1
    echo
    echo "=== $NAME  （$NOTE）"
    grep -E "^ +(all|crazing|inclusion|patches|pitted_surface|rolled-in_scale|scratches) +[0-9]" "$OUT" | tail -7
  done

  echo
  echo "================ 判定门（单 seed；参照 r7_base 0.757/0.402）================"
  echo "  ✅ ≥ 0.417（+0.015）→ 该杠杆成立，作为主模型的候选改动，进入配对 3 seed 确认"
  echo "  ⚠️ 0.408~0.417 → 边缘，单独复跑一次"
  echo "  ❌ ≤ 0.408 → 放弃该杠杆"
  echo "  若四项全不过：本数据集在 yolo26s 规模下已饱和（架构级差异仅 ±2 分），"
  echo "  论文改走「头部分析 + 轻量化/鲁棒性」主线（MRB 折叠后 -5.4% 参数/FLOPs、SPD -10.6% 参数是现成证据）。"
} 2>&1 | tee -a "$ROUND_LOG"

echo "全部完成。总日志：$ROUND_LOG"
