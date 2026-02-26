"""
gen_random_graph.py
生成随机静态图，用于消融实验 M2（对照组）。

设计原则：
  - 与格兰杰图完全相同的稀疏度：每个节点恰好有 TOP_K=8 条入边
  - 边权重随机（uniform [0.1, 1.0]）
  - 固定随机种子，保证可复现
  - 与格兰杰图的唯一区别是连接方式和权重都是随机的，没有因果含义

消融逻辑：
  M2（随机图 + RegimeMoE）vs M3（格兰杰图 + RegimeMoE）
  → 两者架构完全相同，差异只是静态图质量
  → M3 - M2 的提升 = 格兰杰因果先验的贡献
"""
import numpy as np
import pickle
import os

OUTPUT_PATH = './data/sensor_graph/adj_random.pkl'
N_NODES     = 31
TOP_K       = 8      # 与格兰杰图相同的稀疏度
SEED        = 42     # 固定种子，可复现


def gen_random_graph():
    np.random.seed(SEED)
    n   = N_NODES
    adj = np.zeros((n, n), dtype=np.float32)

    for j in range(n):
        # 每个目标节点 j 随机选 TOP_K 个"来源"节点（排除自身）
        candidates = [i for i in range(n) if i != j]
        selected   = np.random.choice(candidates, size=TOP_K, replace=False)
        # 随机边权（与格兰杰图相同量级：[0.1, 1.0]）
        adj[selected, j] = np.random.uniform(0.1, 1.0, size=TOP_K).astype(np.float32)

    n_edges = int(np.sum(adj > 0))
    print(f"随机图（Top-{TOP_K}，seed={SEED}）前 5×5：")
    print(adj[:5, :5])
    print(f"非零边数: {n_edges} / {n*n}  (密度 {100*n_edges/n/n:.1f}%)")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(adj, f)
    print(f"已保存到: {OUTPUT_PATH}")


if __name__ == '__main__':
    gen_random_graph()
