import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from net import gtnet
from util import DataLoaderS

# ================= 配置区域 =================
# 指向您现在已有的模型和数据
DATA_PATH = './data/G31_RawPrice.txt'     # 您的数据路径
MODEL_PATH = './model/model_g31_raw.pt'   # 您跑出来的模型路径
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # Mac 用 mps，否则用 cpu

# 节点名称 (按您的数据顺序)
NODE_NAMES = [f"FX_{i}" for i in range(28)] + ['Gold', 'Oil', 'DXY'] 
NUM_NODES = 31
SEQ_IN_LEN = 168
HORIZON = 3
# ==========================================

def load_model_and_data():
    print("Loading data and model...")
    
    # 1. 加载数据
    Data = DataLoaderS(DATA_PATH, 0.6, 0.2, DEVICE, HORIZON, SEQ_IN_LEN, normalize=2)
    
    # 2. 计算均值标准差用于反归一化（从原始数据计算）
    raw_data = np.loadtxt(DATA_PATH, delimiter=',')
    train_len = int(raw_data.shape[0] * 0.6)
    train_data = raw_data[:train_len, :]
    # 注意：DataLoaderS 使用的是 max normalization，所以 scale 就是最大值
    # 但为了反归一化，我们需要用 scale
    mean = torch.zeros(NUM_NODES).to(DEVICE)  # max normalization 的 mean 是 0
    std = Data.scale.data  # scale 就是最大值，用于反归一化
    
    # 3. 加载模型（整个模型，不是 state_dict）
    print(f"Loading model from {MODEL_PATH}...")
    with open(MODEL_PATH, 'rb') as f:
        model = torch.load(f, map_location=DEVICE)
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")
    
    return model, Data, mean, std

def plot_heatmap_baseline(model):
    print("Generating baseline heatmap...")
    # 原版模型生成图不需要 external argument，只需要 idx
    idx = torch.arange(NUM_NODES).to(DEVICE)
    with torch.no_grad():
        # 调用 net.py 里的 graph_constructor
        # 原版 gc 的 forward 只需要 idx
        adj = model.gc(idx) 
        adj_numpy = adj.cpu().numpy()

    plt.figure(figsize=(16, 14))
    sns.heatmap(adj_numpy, cmap="Blues", vmin=0, vmax=adj_numpy.max(), 
                xticklabels=False, yticklabels=False, cbar_kws={'label': 'Edge Weight'})
    plt.title("Learned Graph Adjacency Matrix", fontsize=16)
    plt.xlabel("Source Node", fontsize=12)
    plt.ylabel("Target Node", fontsize=12)
    # 如果想看具体名字，把下面这行解开
    # plt.xticks(np.arange(NUM_NODES)+0.5, NODE_NAMES, rotation=90, fontsize=8)
    # plt.yticks(np.arange(NUM_NODES)+0.5, NODE_NAMES, rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig("baseline_heatmap.png", dpi=300, bbox_inches='tight')
    print("Saved baseline_heatmap.png")

# ... (预测走势图 plot_predictions 函数跟之前一样，不需要改) ...
# ... (Loss 图 plot_loss_curve 函数跟之前一样，不需要改) ...

def plot_predictions(model, Data, mean, std, target_node_idx=28, plot_len=500):
    print(f"Generating prediction plot for node {target_node_idx}...")
    test_loader = Data.get_batches(Data.test[0], Data.test[1], batch_size=64, shuffle=False)
    preds, trues = [], []
    
    with torch.no_grad():
        for bx, by in test_loader:
            # 根据 train_single_step.py 的格式，需要先 unsqueeze 和 transpose
            # bx shape: (batch, seq_len, num_nodes) -> (batch, 1, num_nodes, seq_len)
            bx_input = torch.unsqueeze(bx, dim=1)  # (batch, 1, seq_len, num_nodes)
            bx_input = bx_input.transpose(2, 3)    # (batch, 1, num_nodes, seq_len)
            
            output = model(bx_input)
            output = torch.squeeze(output)  # (batch, num_nodes) or (batch, 1, num_nodes)
            if len(output.shape) == 1:
                output = output.unsqueeze(dim=0)
            
            # 反归一化：output 是归一化后的，需要乘以 scale
            scale = Data.scale.expand(output.size(0), Data.m)
            preds.append(output * scale)
            trues.append(by * scale)
    
    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)
    
    node_pred = preds[:plot_len, target_node_idx].cpu().numpy()
    node_true = trues[:plot_len, target_node_idx].cpu().numpy()
    
    plt.figure(figsize=(15, 6))
    plt.plot(node_true, label='Actual', color='blue', alpha=0.7, linewidth=1.5)
    plt.plot(node_pred, label='Prediction', color='orange', alpha=0.8, linewidth=1.5)
    plt.title(f"Forecast: Node {target_node_idx} ({NODE_NAMES[target_node_idx] if target_node_idx < len(NODE_NAMES) else f'Node_{target_node_idx}'})", fontsize=14)
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"baseline_pred_{target_node_idx}.png", dpi=300, bbox_inches='tight')
    print(f"Saved baseline_pred_{target_node_idx}.png")

if __name__ == "__main__":
    model, Data, mean, std = load_model_and_data()
    
    # 画原版模型的热力图
    plot_heatmap_baseline(model)
    
    # 画原版模型的预测图 (比如看看黄金预测得咋样)
    plot_predictions(model, Data, mean, std, target_node_idx=28)