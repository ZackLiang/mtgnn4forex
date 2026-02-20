import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 引用您的项目模块
from util import DataLoaderS
from net import gtnet

# ================= 1. 配置区域 (Configuration) =================
# 数据文件路径
DATA_PATH = './data/G31_RawPrice.txt'      

# 设备配置
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# 模型文件路径字典
MODEL_PATHS = {
    'MTGNN (Baseline)': './model/model_baseline.pt',
    'MTGNN + ReVIN': './model/model_revin.pt',
    'MTGNN + ReVIN + Dual': './model/model_dual_adaptive.pt'  # 修正：去掉了末尾空格，保持一致
}

# 核心参数 (必须与训练时保持一致)
SEQ_IN_LEN = 168       
HORIZON = 3            
NUM_NODES = 31         
PLOT_LEN = 500         

# ================= 2. 数据与模型加载 =================

def load_data():
    """加载测试数据和归一化参数"""
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        sys.exit(1)
        
    print("Loading data...")
    Data = DataLoaderS(DATA_PATH, 0.6, 0.2, DEVICE, HORIZON, SEQ_IN_LEN, normalize=2)
    return Data

def get_predictions(model_path, Data):
    """加载指定模型并计算在测试集上的预测值 (含全套兼容性补丁)"""
    print(f"Processing model: {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found. Skipping.")
        return None, None

    try:
        with open(model_path, 'rb') as f:
            # weights_only=False 兼容旧版 pytorch 序列化
            model = torch.load(f, map_location=DEVICE)
            
        # =======================================================
        # 🚑 全能兼容性补丁 (Universal Compatibility Patch)
        # =======================================================
        
        # 1. 补丁：频域注意力
        if not hasattr(model, 'freq_att_enabled'):
            model.freq_att_enabled = False  
            
        # 2. 补丁：双图开关
        if not hasattr(model, 'dual_graph'):
            model.dual_graph = False        
        
        # 3. 补丁：双图融合权重 (Fusion Weight) - 【关键修复】
        # 如果模型开启了双图，但缺少 fusion_weight 参数，手动注入默认值 0.3
        if getattr(model, 'dual_graph', False) and not hasattr(model, 'fusion_weight'):
            model.fusion_weight = torch.nn.Parameter(torch.tensor(0.3), requires_grad=False)
            model.fusion_weight.to(DEVICE)
            
        # 4. 补丁：RevIN 开关
        if not hasattr(model, 'revin_enabled'):
            model.revin_enabled = hasattr(model, 'revin') 
            
        # =======================================================
            
        model.to(DEVICE)
        model.eval()
        
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None, None

    # 获取测试集 Batch
    test_loader = Data.get_batches(Data.test[0], Data.test[1], batch_size=64, shuffle=False)
    
    preds = []
    trues = []
    
    with torch.no_grad():
        for bx, by in test_loader:
            bx_input = torch.unsqueeze(bx, dim=1).transpose(2, 3)
            
            output = model(bx_input)
            output = torch.squeeze(output) 
            
            if len(output.shape) == 1:
                output = output.unsqueeze(dim=0)
                
            scale = Data.scale.expand(output.size(0), Data.m)
            preds.append(output * scale)
            trues.append(by * scale)
            
    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)
    
    return preds.cpu().numpy(), trues.cpu().numpy()

# ================= 3. 绘图逻辑 (Visualization) =================

def plot_mainstream_pairs(results, ground_truth, node_indices, labels_map):
    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(24, 14))
    axes = axes.flatten()
    
    # === 样式配置 (Style Configuration) ===
    # 全部改为实线 ('-')
    styles = {
        'MTGNN (Baseline)':   {'color': 'green',   'style': '-', 'alpha': 0.8, 'width': 1.5},
        'MTGNN + ReVIN':      {'color': 'orange',  'style': '-', 'alpha': 0.8, 'width': 1.5},
        'MTGNN + ReVIN + Dual': {'color': '#D32F2F', 'style': '-',  'alpha': 0.8, 'width': 1.5} 
    }

    for i, ax in enumerate(axes):
        if i >= len(node_indices): break
        
        node_idx = node_indices[i]
        node_name = labels_map.get(node_idx, f"Node {node_idx}")
        
        # 1. 绘制真实值 (Ground Truth) -> 蓝色实线
        ax.plot(ground_truth[:PLOT_LEN, node_idx], label='Ground Truth', 
                color='blue', linestyle='-', linewidth=2.0, alpha=0.6)
        
        # 2. 绘制各个模型的预测值
        for model_name, pred_data in results.items():
            if pred_data is not None:
                st = styles.get(model_name, {'color': 'black', 'style': '-'})
                
                ax.plot(pred_data[:PLOT_LEN, node_idx], label=model_name, 
                        color=st['color'], linestyle=st['style'], 
                        alpha=st['alpha'], linewidth=st.get('width', 1.5))
        
        # 3. 图表修饰
        ax.set_title(f"{node_name} Price Forecast", fontsize=18, fontweight='bold', pad=10)
        ax.set_xlabel("Time Step (Hour)", fontsize=14)
        ax.set_ylabel("Exchange Rate", fontsize=14)
        
        # 优化图例
        ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True, fancybox=True)
        ax.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    save_path = "pic/forecast_comparison_mainstream_solid.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[Success] Plot saved to: {save_path}")
    plt.show()

# ================= 4. 主程序入口 =================

if __name__ == "__main__":
    print("=== Starting Forecast Visualization Script ===\n")
    
    Data = load_data()
    
    all_preds = {}
    ground_truth = None
    
    print("\n--- Running Inference ---")
    for name, path in MODEL_PATHS.items():
        pred, true = get_predictions(path, Data)
        if pred is not None:
            all_preds[name] = pred
            if ground_truth is None:
                ground_truth = true
    
    if ground_truth is not None and len(all_preds) > 0:
        print("\n--- Plotting Results ---")
        
        # 四大主流货币对
        target_indices = [15, 27, 20, 4]
        
        target_labels = {
            15: "EUR/USD ",
            27: "USD/JPY ",
            20: "GBP/USD ",
            4: "AUD/USD "
        }
        
        plot_mainstream_pairs(all_preds, ground_truth, target_indices, target_labels)
        
    else:
        print("\n[Error] No predictions generated. Please check model paths and data.")