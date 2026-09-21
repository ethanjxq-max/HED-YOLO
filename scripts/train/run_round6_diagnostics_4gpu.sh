#!/usr/bin/env bash
# ============================================================================
# 第 6 轮（协议/架构诊断）——4 卡并行，单 seed 42，全部 cache=False
#
# 为什么先做诊断（不再换模块）：
#   第 5 轮 MRB（训练期 5 分支、推理期折叠为单卷积的结构级改动）与 ASFF 全部落在
#   参照基座 ±0.012 以内；加上此前 20 次模块尝试，结论已经很清楚：
#   **在"从零训练 + 250 epoch + 单 seed"这个协议里，模块级效应（<1.0）小于测量分辨率（≈±0.8）。**
#   与其继续赌第 22 个模块，先用 1 小时把"天花板在哪"量出来：
#     A 预训练初始化（COCO 权重）——领域标准协议，最大的一根杠杆
#     B 训练 500 epoch         —— 250 epoch 从零训练是否欠收敛（种子方差 0.8 就是证据）
#     C 关闭端到端头（NMS）     —— 同协议下 v9s 0.397 / yolo11s 0.386 都高于 yolo26s 0.371，
#                                 差额可能全在这个"一对一 NMS-free 头"上
#     D imgsz 800              —— 分辨率/尺度杠杆
#   四个结果分别决定论文的协议与第三创新点的落点，然后模块筛选在新协议里重跑。
#
# 参照系（同一台机器、同协议、cache=False、seed=42）：
#   baseline yolo26s = 0.713/0.370（09-13 17:15 那轮）
#   E = yolo26s_db_fem_11_13 = 0.732/0.384（第 5 轮同机复核值，历史记录 0.743/0.388）
#
# 用法：
#   cd /path/to/HED-YOLO
#   chmod +x run_round6_diagnostics_4gpu.sh && bash run_round6_diagnostics_4gpu.sh
#   监控：tail -f logs/r6_round_*.log
# ============================================================================
set -u
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # 自动定位仓库根目录；也可用 ROOT=/your/path bash xxx.sh 覆盖
cd "$ROOT" || { echo "找不到项目目录 $ROOT"; exit 1; }
mkdir -p logs runs

DATA=ultralytics/cfg/datasets/neu_det.yaml
CFG=ultralytics/cfg/models/26
TAIL="imgsz=640 batch=24 workers=8 seed=42 scale=0.5 cache=False exist_ok=True"
STAMP=$(date +%m%d_%H%M)
ROUND_LOG="logs/r6_round_${STAMP}.log"

# ---- 预检：预训练权重在不在 ----
if [ ! -f "$ROOT/yolo26s.pt" ]; then
  echo "❌ 缺少 $ROOT/yolo26s.pt（COCO 预训练权重，约 20MB）"
  echo "   上传：把本地 yolo26_project/yolo26s.pt 传到 $ROOT/"
  echo "   或下载：yolo download model=yolo26s.pt  然后 mv yolo26s.pt $ROOT/"
  exit 1
fi

# name|gpu|yaml|额外参数|说明
JOBS=(
  "r6_pt250|0|${CFG}/yolo26s.yaml|epochs=250 pretrained=$ROOT/yolo26s.pt|A 预训练初始化（250ep）"
  "r6_e500|1|${CFG}/yolo26s.yaml|epochs=500|B 从零训练 500 epoch"
  "r6_noe2e|2|${CFG}/yolo26s_noe2e.yaml|epochs=250|C 关闭端到端头（一对多 + NMS）"
  "r6_img800|3|${CFG}/yolo26s.yaml|epochs=250 imgsz=800|D 输入分辨率 800"
)

{
  echo "================ [$(date '+%F %T')] 第6轮（协议/架构诊断）开始 ================"
  for j in "${JOBS[@]}"; do
    IFS='|' read -r NAME GPU YAML EXTRA NOTE <<< "$j"
    printf "[%s] GPU%s ← %-10s %-42s %s\n" "$(date '+%H:%M:%S')" "$GPU" "$NAME" "$EXTRA" "$NOTE"
  done

  for j in "${JOBS[@]}"; do
    IFS='|' read -r NAME GPU YAML EXTRA NOTE <<< "$j"
    nohup yolo detect train data="$DATA" model="$YAML" device="$GPU" $TAIL $EXTRA \
      project="$ROOT/runs/$NAME" name="$NAME" \
      > "logs/${NAME}.log" 2>&1 &
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
      echo "  ✅ $NAME 运行中 ${TR:+(预训练迁移: $TR)}"
    else
      echo "  ⏳ $NAME 启动中（未见报错）"
    fi
  done
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
  [ "$FAIL" = "1" ] && echo "⚠️ 有任务启动失败：pkill -f 'yolo detect train'，然后看 logs/ 下对应 *.log 末尾"
} 2>&1 | tee -a "$ROUND_LOG"

while pgrep -f "yolo detect train" > /dev/null; do sleep 60; done

{
  echo
  echo "================ [$(date '+%F %T')] 训练结束，统一验证（best.pt）================"
  for j in "${JOBS[@]}"; do
    IFS='|' read -r NAME GPU YAML EXTRA NOTE <<< "$j"
    BEST="$ROOT/runs/$NAME/$NAME/weights/best.pt"
    OUT="logs/${NAME}_val.txt"
    yolo detect val model="$BEST" data="$DATA" device=0 split=val > "$OUT" 2>&1
    echo
    echo "=== $NAME  （$NOTE）"
    grep -E "^ +(all|crazing|inclusion|patches|pitted_surface|rolled-in_scale|scratches) +[0-9]" "$OUT" | tail -7
  done

  echo
  echo "================ 对表（参照：baseline 0.713/0.370；E 0.732/0.384）================"
  echo "  A 预训练：若 mAP50-95 ≥ 0.385（+0.015）→ 论文主表换回预训练协议（领域标准），"
  echo "            随后用 map_pretrained.py 把 COCO 权重迁到自定义结构（E / E+MRB / E+ASFF），"
  echo "            模块筛选在新协议里重跑。"
  echo "  B 500ep ：若 ≥ 0.385 → 250 epoch 确实欠收敛，延长训练并把模块筛选放到 500ep 下重跑。"
  echo "  C noe2e ：若 ≥ 0.385 → 精度差在 NMS-free 一对一头上（v9s/yolo11s 高出的 1.5~2.6 分在这里），"
  echo "            第三创新点应改为「检测头结构 / 训练期监督」，而不是骨干或颈部模块。"
  echo "  D img800: 若 ≥ 0.385 → 尺度/分辨率是杠杆，第三点做「尺度重分配」（含 P2 或更高分辨率训练）。"
  echo "  若四项全部 ≤ 0.385：本数据集在该模型族上已饱和（架构级差异也只有 ±0.02），"
  echo "            论文改走「精度持平 + 效率/鲁棒性」叙事（MRB 折叠后 -5.4% 参数/FLOPs 现成可用）。"
} 2>&1 | tee -a "$ROUND_LOG"

echo "全部完成。总日志：$ROUND_LOG"
