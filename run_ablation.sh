#!/bin/bash

# 定义公共参数，方便统一修改
DATA_PATH="./data/G31_RawPrice.txt"
NODES=31
EPOCHS=25
RUNS=3
DEVICE="mps"

echo "=================================================="
echo "🚀 开始执行 Regime-MoE-GNN 自动消融实验"
echo "=================================================="

# 1. 跑 M0 (Baseline)
echo "[1/5] 正在运行 M0: 原版 MTGNN (Baseline)..."
python train_single_step.py --data $DATA_PATH --num_nodes $NODES --epochs $EPOCHS --runs $RUNS --device $DEVICE --save ./model/model_M0.pt --revin 0 --dual_graph 0 --use_router 0 --use_dirloss 0 > log_M0_baseline.txt 2>&1
echo "✅ M0 运行完毕，日志已保存至 log_M0_baseline.txt"

# 2. 跑 M1 (+ RevIN)
echo "[2/5] 正在运行 M1: MTGNN + RevIN..."
python train_single_step.py --data $DATA_PATH --num_nodes $NODES --epochs $EPOCHS --runs $RUNS --device $DEVICE --save ./model/model_M1.pt --revin 1 --dual_graph 0 --use_router 0 --use_dirloss 0 > log_M1_revin.txt 2>&1
echo "✅ M1 运行完毕，日志已保存至 log_M1_revin.txt"

# 3. 跑 M2 (+ RevIN + 静态双图)
echo "[3/5] 正在运行 M2: MTGNN + RevIN + Static Dual Graph..."
python train_single_step.py --data $DATA_PATH --num_nodes $NODES --epochs $EPOCHS --runs $RUNS --device $DEVICE --save ./model/model_M2.pt --revin 1 --dual_graph 1 --use_router 0 --use_dirloss 0 --adj_data ./data/adj_mx.pkl > log_M2_dualgraph.txt 2>&1
echo "✅ M2 运行完毕，日志已保存至 log_M2_dualgraph.txt"

# 4. 跑 M3 (+ RevIN + 静态双图 + 动态路由)
echo "[4/5] 正在运行 M3: MTGNN + RevIN + Dual Graph + MoE Router..."
python train_single_step.py --data $DATA_PATH --num_nodes $NODES --epochs $EPOCHS --runs $RUNS --device $DEVICE --save ./model/model_M3.pt --revin 1 --dual_graph 1 --use_router 1 --use_dirloss 0 --adj_data ./data/adj_mx.pkl > log_M3_router.txt 2>&1
echo "✅ M3 运行完毕，日志已保存至 log_M3_router.txt"

# 5. 跑 M4 (+ RevIN + 静态双图 + 动态路由 + 方向性Loss)
echo "[5/5] 正在运行 M4: Ours 完全体 (+ DirLoss)..."
python train_single_step.py --data $DATA_PATH --num_nodes $NODES --epochs $EPOCHS --runs $RUNS --device $DEVICE --save ./model/model_M4.pt --revin 1 --dual_graph 1 --use_router 1 --use_dirloss 1 --dir_weight 0.1 --adj_data ./data/adj_mx.pkl > log_M4_ours.txt 2>&1
echo "✅ M4 运行完毕，日志已保存至 log_M4_ours.txt"

echo "=================================================="
echo "🎉 所有消融实验已全部完成！您可以查看对应的 log 文件收取表格数据了。"
echo "=================================================="