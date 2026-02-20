import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import pandas as pd
from net import gtnet
# 设置风格，让图表看起来更学术、更像论文图
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# === 配置路径 (请根据实际情况修改) ===
DATA_PATH = './data/G31_RawPrice.txt'           # 原始数据路径
ADJ_PATH = './data/sensor_graph/adj_mx.pkl'     # 静态图路径
MODEL_PATH = './model/model_ours.pt'            # 您的“完全体”模型路径
OUTPUT_DIR = './ppt_figures'                    # 图片保存目录

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def plot_distribution_shift():
    """图1：证明分布漂移 (Pain Point 1)"""
    print("绘制 Fig 1: Distribution Shift...")
    try:
        data = np.loadtxt(DATA_PATH, delimiter=',')
        # 以前 60% 做训练，后 20% 做测试 (模拟 Data Split)
        train_len = int(data.shape[0] * 0.6)
        train_data = data[:train_len, 0] # 取第0列(比如AUD)来看
        test_data = data[int(data.shape[0]*0.8):, 0]

        plt.figure(figsize=(10, 6))
        sns.kdeplot(train_data, fill=True, color="blue", label="Train Data Distribution", alpha=0.3)
        sns.kdeplot(test_data, fill=True, color="red", label="Test Data Distribution", alpha=0.3)
        plt.title("Pain Point I: Distribution Shift (Non-stationarity)", fontsize=16)
        plt.xlabel("Exchange Rate Value", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.legend(fontsize=12)
        plt.savefig(f"{OUTPUT_DIR}/fig1_dist_shift.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"无法绘制 Fig 1 (可能缺少数据): {e}")

def plot_noise_zoom():
    """图2：证明高频噪声 (Pain Point 2)"""
    print("绘制 Fig 2: High Frequency Noise...")
    try:
        data = np.loadtxt(DATA_PATH, delimiter=',')
        # 取中间一段 200 个时间步的数据，展示其锯齿
        sample = data[1000:1200, 0] # Node 0
        
        plt.figure(figsize=(12, 5))
        plt.plot(sample, color='black', linewidth=1.5, label="Raw Minute-level Price")
        # 画一个平滑的趋势线对比一下
        df_sample = pd.Series(sample)
        trend = df_sample.rolling(window=20).mean()
        plt.plot(trend, color='orange', linewidth=2, linestyle='--', label="Underlying Trend")
        
        plt.title("Pain Point II: High-Frequency Noise & Microstructure", fontsize=16)
        plt.xlabel("Time Step (Minutes)", fontsize=14)
        plt.ylabel("Price", fontsize=14)
        plt.legend(fontsize=12)
        plt.savefig(f"{OUTPUT_DIR}/fig2_noise.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"无法绘制 Fig 2: {e}")

def plot_freq_weights(model):
    """图3：证明频域去噪 (Visualization I)"""
    print("绘制 Fig 3: Frequency Weights...")
    if hasattr(model, 'freq_att_layer'):
        # 提取权重: (1, Channels, 1, FreqLen) -> (Channels, FreqLen)
        weights = model.freq_att_layer.freq_weight.detach().cpu().numpy().squeeze()
        mean_weight = np.mean(weights, axis=0)
        
        plt.figure(figsize=(10, 6))
        # 画背景的所有通道线 (淡色)
        for i in range(min(weights.shape[0], 50)): # 最多画50条，别太乱
            plt.plot(weights[i], color='blue', alpha=0.05)
        
        # 画平均线 (深色)
        plt.plot(mean_weight, color='red', linewidth=3, label="Mean Freq Weight")
        
        plt.axhline(y=1.0, color='gray', linestyle='--', label="Identity (No Filter)")
        plt.title("Evidence: Learned Low-Pass Filter (Left High, Right Low)", fontsize=16)
        plt.xlabel("Frequency Component (Low -> High)", fontsize=14)
        plt.ylabel("Learned Weight", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{OUTPUT_DIR}/fig3_freq_weights.png", dpi=300)
        plt.close()
    else:
        print("模型中未找到 freq_att_layer，跳过 Fig 3")

def plot_dual_graph(model):
    """图4：证明双图结构 (Visualization II)"""
    print("绘制 Fig 4: Dual Graph Comparison...")
    try:
        # 1. 加载静态图
        with open(ADJ_PATH, 'rb') as f:
            static_adj = pickle.load(f)
        
        # 2. 提取动态图
        idx = torch.arange(model.num_nodes).to(DEVICE)
        dynamic_adj = model.gc(idx).detach().cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        sns.heatmap(static_adj, ax=axes[0], cmap="Blues", cbar=False, xticklabels=False, yticklabels=False)
        axes[0].set_title("Static Graph (Financial Prior)", fontsize=16)
        
        sns.heatmap(dynamic_adj, ax=axes[1], cmap="Greens", cbar=False, xticklabels=False, yticklabels=False)
        axes[1].set_title("Dynamic Graph (Data-Driven)", fontsize=16)
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/fig4_dual_graph.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"无法绘制 Fig 4: {e}")

def plot_fusion_weight_change():
    """图5：融合权重对比 (Insight)"""
    print("绘制 Fig 5: Fusion Weight Change...")
    # 这里用您实验跑出来的真实数据
    labels = ['Without Freq Attn', 'With Freq Attn (Ours)']
    values = [0.29, 0.31] # 根据之前实验结果填入
    
    plt.figure(figsize=(8, 6))
    colors = ['#bdc3c7', '#e74c3c'] # 灰色 vs 红色
    bars = plt.bar(labels, values, color=colors, width=0.5)
    
    plt.title("Impact of De-noising on Structural Prior Reliance", fontsize=16)
    plt.ylabel("Fusion Weight (Lambda)", fontsize=14)
    plt.ylim(0, 0.4)
    
    # 在柱子上标数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval}", ha='center', fontsize=14, fontweight='bold')
        
    plt.savefig(f"{OUTPUT_DIR}/fig5_fusion_weight.png", dpi=300)
    plt.close()

def plot_ablation_study():
    """图6：消融实验 (Ablation Study)"""
    print("绘制 Fig 6: Ablation Study...")
    models = ['Baseline', '+ RevIN', '+ Dual Graph', '+ Freq (Ours)']
    mae_scores = [0.4916, 0.1646, 0.1643, 0.1636] # 您的最佳实验数据
    
    plt.figure(figsize=(10, 6))
    # 使用阶梯图或者折线图来展示下降趋势
    plt.plot(models, mae_scores, marker='o', linewidth=3, markersize=10, color='navy')
    
    # 标出数值
    for i, score in enumerate(mae_scores):
        plt.text(i, score + 0.01, f"{score}", ha='center', fontsize=12, fontweight='bold')
        
    plt.title("Ablation Study: Step-by-Step Performance Gain", fontsize=16)
    plt.ylabel("MAE Loss (Lower is Better)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"{OUTPUT_DIR}/fig6_ablation.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    # 1. 加载模型
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        try:
            # 需要 import net 和 layer 才能 load
            with open(MODEL_PATH, 'rb') as f:
                model = torch.load(f, map_location=DEVICE)
            model.eval()
            
            # 画需要模型的图
            plot_freq_weights(model)
            plot_dual_graph(model)
        except Exception as e:
            print(f"模型加载失败 (可能是代码版本不匹配): {e}")
            print("提示: 请确保 net.py 和 layer.py 与训练时一致")
    else:
        print(f"未找到模型文件: {MODEL_PATH}，跳过模型相关绘图")

    # 2. 画不需要模型的图 (数据分析 & 结果展示)
    plot_distribution_shift()
    plot_noise_zoom()
    plot_fusion_weight_change()
    plot_ablation_study()
    
    print(f"\n✅ 所有图片已生成在 {OUTPUT_DIR} 文件夹中！")