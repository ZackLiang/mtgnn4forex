#!/bin/bash

# 定义公共参数
DATA_PATH="./data/G31_RawPrice.txt"
ADJ_GRANGER="./data/sensor_graph/adj_mx.pkl"       # 格兰杰因果图
ADJ_RANDOM="./data/sensor_graph/adj_random.pkl"    # 随机静态图（对照）
NODES=31
EPOCHS=25
RUNS=3
DEVICE="mps"

echo "=================================================="
echo "🚀 开始执行 Regime-MoE-GNN 消融实验（新消融设计）"
echo "=================================================="
echo ""
echo "消融逻辑："
echo "  M0: Baseline MTGNN"
echo "  M1: +RevIN（解决分布漂移）"
echo "  M2: +RegimeMoE + 随机图（验证双专家架构本身有效）"
echo "  M3: +RegimeMoE + 格兰杰图（验证格兰杰先验是关键增量）"
echo "  M4: +DirLoss（方向性约束）"
echo "  M2→M3 的差异 = 格兰杰因果先验的贡献"
echo "=================================================="

# ⚠️ 前置条件：确认以下两个图文件存在
#   ./data/sensor_graph/adj_mx.pkl       （格兰杰因果图）
#   ./data/sensor_graph/adj_random.pkl   （随机图）
# 如不存在，先运行：
#   python gen_corr_matrix.py
#   python gen_random_graph.py

# ── M0: 原版 MTGNN（无 RevIN，无图，无路由）──────────────────────
echo "[1/5] 运行 M0: Baseline MTGNN..."
python train_single_step.py \
  --data $DATA_PATH --num_nodes $NODES --epochs $EPOCHS --runs $RUNS \
  --device $DEVICE --normalize 2 \
  --revin 0 --dual_graph 0 --use_router 0 --use_dirloss 0 \
  --save ./model/model_M0.pt > log_M0_baseline.txt 2>&1
echo "✅ M0 完毕 → log_M0_baseline.txt"

# ── M1: +RevIN ───────────────────────────────────────────────────
echo "[2/5] 运行 M1: +RevIN..."
python train_single_step.py \
  --data $DATA_PATH --num_nodes $NODES --epochs $EPOCHS --runs $RUNS \
  --device $DEVICE --normalize 0 \
  --revin 1 --dual_graph 0 --use_router 0 --use_dirloss 0 \
  --save ./model/model_M1.pt > log_M1_revin.txt 2>&1
echo "✅ M1 完毕 → log_M1_revin.txt"

# ── M2: +RegimeMoE + 随机图（对照：架构有效，先验无效）───────────
echo "[3/5] 运行 M2: +RegimeMoE + 随机图..."
python train_single_step.py \
  --data $DATA_PATH --num_nodes $NODES --epochs $EPOCHS --runs $RUNS \
  --device $DEVICE --normalize 0 \
  --revin 1 --dual_graph 1 --use_router 1 --use_dirloss 0 \
  --adj_data $ADJ_RANDOM \
  --save ./model/model_M2.pt > log_M2_random.txt 2>&1
echo "✅ M2 完毕 → log_M2_random.txt"

# ── M3: +RegimeMoE + 格兰杰图（关键：先验质量提升效果）─────────
echo "[4/5] 运行 M3: +RegimeMoE + 格兰杰因果图..."
python train_single_step.py \
  --data $DATA_PATH --num_nodes $NODES --epochs $EPOCHS --runs $RUNS \
  --device $DEVICE --normalize 0 \
  --revin 1 --dual_graph 1 --use_router 1 --use_dirloss 0 \
  --adj_data $ADJ_GRANGER \
  --save ./model/model_M3.pt > log_M3_granger.txt 2>&1
echo "✅ M3 完毕 → log_M3_granger.txt"

# ── M4: 完全体（+DirLoss）─────────────────────────────────────
echo "[5/5] 运行 M4: Ours 完全体 (+DirLoss)..."
python train_single_step.py \
  --data $DATA_PATH --num_nodes $NODES --epochs $EPOCHS --runs $RUNS \
  --device $DEVICE --normalize 0 \
  --revin 1 --dual_graph 1 --use_router 1 --use_dirloss 1 --dir_weight 0.05 \
  --adj_data $ADJ_GRANGER \
  --save ./model/model_M4.pt > log_M4_ours.txt 2>&1
echo "✅ M4 完毕 → log_M4_ours.txt"

echo "=================================================="
echo "🎉 消融实验完成！结果文件："
echo "   log_M0_baseline.txt  log_M1_revin.txt"
echo "   log_M2_random.txt    log_M3_granger.txt"
echo "   log_M4_ours.txt"
echo "=================================================="

# ── 自动运行金融回测（读取各模型保存的预测数据） ─────────────────────
echo ""
echo "📈 开始金融回测分析..."
conda run -n mtgnn python backtest.py --runs $RUNS --cost_bps 1.0
echo "✅ 回测完成，净值曲线已保存至 ppt_figures/equity_ablation.png"
