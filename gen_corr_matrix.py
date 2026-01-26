import pandas as pd
import numpy as np
import pickle
import os

# 配置你的数据路径
DATA_PATH = './data/G31_RawPrice.txt' # 或者是 ./data/exchange_rate.txt
OUTPUT_PATH = './data/sensor_graph/adj_mx.pkl'

def gen_corr_matrix():
    # 1. 读取数据
    # 假设数据是 (Samples, Nodes) 格式，逗号分隔
    try:
        df = pd.read_csv(DATA_PATH, header=None)
        print(f"Data shape: {df.shape}")
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    data = df.values
    
    # 2. 计算皮尔逊相关系数 (Pearson Correlation)
    # df.corr() 默认是按列计算 (Columns as Variables)
    corr_matrix = df.corr().values
    
    # 3. 处理一下：取绝对值（关注相关强度），并设个阈值去噪（可选）
    # 这里我们保留原始相关性，让模型自己去学权重
    # 但为了作为邻接矩阵，通常需要归一化或二值化，这里我们只做简单的绝对值处理
    # corr_matrix = np.abs(corr_matrix) 
    
    print("Correlation Matrix generated.")
    print(corr_matrix[:5, :5]) # 打印前5行看看

    # 4. 保存为 pkl
    if not os.path.exists(os.path.dirname(OUTPUT_PATH)):
        os.makedirs(os.path.dirname(OUTPUT_PATH))
        
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(corr_matrix, f)
    
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == '__main__':
    gen_corr_matrix()