import pandas as pd
import numpy as np
import pickle
import os

# === 配置参数 ===
DATA_PATH = './data/G31_RawPrice.txt' 
OUTPUT_PATH = './data/sensor_graph/adj_mx.pkl'
TOP_K = 15  # 【关键修改】只保留最强的 10 个邻居，剔除弱相关噪声

def gen_corr_matrix():
    print(f"Reading data from {DATA_PATH}...")
    try:
        # 1. 读取数据
        df = pd.read_csv(DATA_PATH, header=None)
        print(f"Raw Price Data shape: {df.shape}")
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    # === 核心修改 1: 使用收益率 (Returns) 算相关性 ===
    # 原始价格是非平稳的，直接算相关性会有“伪相关” (Spurious Correlation)。
    # 用 (P_t - P_{t-1})/P_{t-1} 来算，这才是真正的市场联动。
    df_returns = df.pct_change().dropna()
    print(f"Returns Data shape: {df_returns.shape}")

    # 2. 计算皮尔逊相关系数
    # 这时候算出来的才是“真·金融相关性”
    corr_matrix = df_returns.corr().values
    
    # 取绝对值（我们只关心“是否强相关”，不关心正负，GCN会自己学符号）
    corr_matrix = np.abs(corr_matrix)
    
    # === 核心修改 2: Top-K 稀疏化 (去噪) ===
    # 原始矩阵是全连接的(Dense)，包含大量微弱的随机噪声。
    # 我们只保留每个货币最强的 TOP_K 个“盟友”。
    num_nodes = corr_matrix.shape[0]
    sparse_matrix = np.zeros_like(corr_matrix)
    
    for i in range(num_nodes):
        row = corr_matrix[i, :]
        # 找到前 Top_K 大的索引 (argsort 默认从小到大，所以取最后 K 个)
        top_k_indices = np.argsort(row)[-TOP_K:]
        
        # 只把这些位置的值填回去，其他位置保留为 0
        sparse_matrix[i, top_k_indices] = row[top_k_indices]
        
    # 转成 Tensor 友好的 float32
    adj_mx = sparse_matrix.astype(np.float32)
    
    print(f"\nGenerated Sparse Graph (Top-{TOP_K}):")
    print(adj_mx[:5, :5]) # 打印前5行看看，应该有很多 0

    # 3. 保存
    if not os.path.exists(os.path.dirname(OUTPUT_PATH)):
        os.makedirs(os.path.dirname(OUTPUT_PATH))
        
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(adj_mx, f)
    
    print(f"\nSuccessfully saved to {OUTPUT_PATH}")

if __name__ == '__main__':
    gen_corr_matrix()