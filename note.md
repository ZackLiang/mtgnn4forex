# 初次训练：
## 训练指令
python train_single_step.py \
  --data ./data/G31_RawPrice.txt \
  --num_nodes 31 \
  --subgraph_size 20 \
  --seq_in_len 168 \
  --epochs 5 \
  --normalize 2 \
  --device mps \
  --save ./model/model_g31_raw.pt
## 结果：
  valid   rse     rae     corr
mean    0.0070  0.0038  0.9768
std     0.0013  0.0008  0.0079

test    rse     rae     corr
mean    0.0100  0.0053  0.9736
std     0.0024  0.0014  0.0086


# baseline：
## 训练指令：
python train_single_step.py \
  --data ./data/G31_RawPrice.txt \
  --num_nodes 31 \
  --subgraph_size 20 \
  --seq_in_len 168 \
  --epochs 30 \
  --normalize 2 \
  --revin 0 \
  --runs 5 \
  --save ./model/model_baseline.pt




# 创新点1:RevIN —— 解决“分布漂移”问题
## 论文来源：Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift. ICLR 2022.
## 背景：金融数据是非平稳的，训练集的统计特性和测试集完全不同。原来模型直接输入原始数据。当外汇市场发生剧烈波动时，模型见过的训练数据无法涵盖当前的分布，导致预测失效。
## 原理
### 归一化 (Normalize): 在输入模型前，减去当前窗口的均值 $\mu$，除以标准差 $\sigma$。让模型只学“形态”，不学“绝对数值”。
### 模型计算: 中间经过 MTGNN 的各种卷积。
### 反归一化 (Denormalize): 在输出预测结果后，乘回 $\sigma$，加上 $\mu$。恢复真实的金融数值
