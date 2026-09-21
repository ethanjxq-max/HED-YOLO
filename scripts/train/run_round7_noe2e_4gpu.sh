#!/usr/bin/env bash
# ============================================================================
# 第 7 轮：在"高天花板头"（关闭 NMS-free 一对一，改一对多 + NMS）下重筛三个创新点
#
# 为什么要换头重筛（第 6 轮诊断结论，同协议同机器 seed=42）：
#   vanilla 从零 0.713/0.370  |  vanilla+COCO预训练 0.713/0.369（=0，预训练无效）
#   vanilla @imgsz800 0.708/0.365（−0.5，无效）
#   vanilla 关闭 e2e（一对多+NMS）0.757/0.402 → +4.4 mAP50 / +3.2 mAP50-95（4× 种子噪声）
#   六个类全部改善，且 crazing/inclusion/patches/pitted/rolled/scratches 的 mAP50-95
#   都是 yolo26 系列的历史最佳 → **瓶颈在 YOLO26 的 NMS-free 一对一检测头，不在骨干/颈部**
#   过去 22 次模块尝试"零涨点"，是因为都在这个被压住的天花板下比较。
#
# 本轮 4 卡（全部 250ep / batch24 / seed42 / cache=False / 单卡单配置）：
#   GPU0 r7_base     yolo26s_noe2e                          本轮参照（复核 0.402）
#   GPU1 r7_E        yolo26s_db_fem_11_13_noe2e             +DB+FEM（对 GPU0）
#   GPU2 r7_mrb      yolo26s_noe2e_mrb                      +MRB（对 GPU0，第三点候选①）
#   GPU3 r7_E_asff   yolo26s_db_fem_11_13_noe2e_asff        +ASFF（对 GPU1，第三点候选②）
#
# 用法：
#   cd /path/to/HED-YOLO
#   chmod +x run_round7_noe2e_4gpu.sh && bash run_round7_noe2e_4gpu.sh
#   监控：tail -f logs/r7_round_*.log
# ============================================================================
set -u
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # 自动定位仓库根目录；也可用 ROOT=/your/path bash xxx.sh 覆盖
cd "$ROOT" || { echo "找不到项目目录 $ROOT"; exit 1; }
mkdir -p logs runs

DATA=ultralytics/cfg/datasets/neu_det.yaml
CFG=ultralytics/cfg/models/26
TAIL="epochs=250 imgsz=640 batch=24 workers=8 seed=42 scale=0.5 cache=False"
STAMP=$(date +%m%d_%H%M)
ROUND_LOG="logs/r7_round_${STAMP}.log"

# ---- 预检 1：MuSGD 补丁（不打补丁有概率训练中途崩） ----
if ! grep -q "reshape(len(u), -1)" ultralytics/optim/muon.py 2>/dev/null; then
  echo "⚠️ 未检测到 muon.py 补丁（view → reshape）。建议先打补丁，否则 MuSGD 有概率中途崩溃："
  echo "   cp ultralytics/optim/muon.py ultralytics/optim/muon.py.bak"
  echo "   python -c \"import pathlib;p=pathlib.Path('ultralytics/optim/muon.py');s=p.read_text();"
  echo "     old='m = u.view(len(u), -1) if u.ndim > 2 else u';new='m = u.reshape(len(u), -1) if u.ndim > 2 else u';"
  echo "     assert s.count(old)==1; p.write_text(s.replace(old,new)); print('patched')\""
  echo "   （10 秒后继续执行本轮）"; sleep 10
fi

# ---- 预检 2：三个新 yaml 在不在 ----
for f in yolo26s_noe2e.yaml yolo26s_db_fem_11_13_noe2e.yaml yolo26s_noe2e_mrb.yaml yolo26s_db_fem_11_13_noe2e_asff.yaml; do
  [ -f "$CFG/$f" ] || { echo "❌ 缺少 $CFG/$f（需上传）"; exit 1; }
done

JOBS=(
  "r7_base|0|${CFG}/yolo26s_noe2e.yaml|本轮参照（一对多 + NMS 基线）"
  "r7_E|1|${CFG}/yolo26s_db_fem_11_13_noe2e.yaml|+ DB+FEM@11,13（C1/C2 消融行）"
  "r7_mrb|2|${CFG}/yolo26s_noe2e_mrb.yaml|+ MRB（第三点候选①）"
  "r7_E_asff|3|${CFG}/yolo26s_db_fem_11_13_noe2e_asff.yaml|+ ASFF（第三点候选②，对 r7_E）"
)

{
  echo "================ [$(date '+%F %T')] 第7轮（高天花板头下重筛）开始 ================"
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
  echo "================ 判定门（单 seed，本轮内部互比）================"
  echo "  r7_E      vs r7_base ：mAP50-95 ≥ +0.015 → DB+FEM 在新头下仍有效（C1/C2 成立）"
  echo "  r7_mrb    vs r7_base ：mAP50-95 ≥ +0.015 → MRB 成为第三创新点"
  echo "  r7_E_asff vs r7_E    ：mAP50-95 ≥ +0.015 → ASFF 成为第三创新点"
  echo "  三个都不涨 → 换 D 阶梯（ASD 频带解耦 / ODConv 动态卷积 / 更长训练）"
  echo "  另有加分项：noe2e 模型可 fuse（无 NMS-free 头），参数量/FLOPs 低于 e2e 版本，"
  echo "            且需在论文里补 NMS 的耗时对比（FPS 表）。"
} 2>&1 | tee -a "$ROUND_LOG"

echo "全部完成。总日志：$ROUND_LOG"
