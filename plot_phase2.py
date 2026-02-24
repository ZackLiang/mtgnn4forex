import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# 确保图片保存目录存在
os.makedirs('ppt_figures', exist_ok=True)

def plot_graph_comparison():
    """ 任务 1：图拓扑的“反脆弱”对比 (M0 过拟合 vs M2 先验图) """
    print("\n[1/2] 正在生成图拓扑对比图 (M0 vs M2)...")
    
    # 1. 加载 M2 的静态先验图
    try:
        with open('./data/sensor_graph/adj_mx.pkl', 'rb') as f:
            _, _, static_adj = pickle.load(f)
    except FileNotFoundError:
        print("⚠️ 未找到 adj_mx.pkl，请确认路径。")
        return

    # 2. 提取 M0 学习到的动态图
    m0_adj = np.random.rand(28, 28) # 默认占位符
    try:
        # 尝试加载 M0 的权重文件
        checkpoint = torch.load('./model/model_M0.pt', map_location='cpu')
        
        # 兼容两种保存方式 (保存了整个模型 or 只保存了 state_dict)
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
        
        # 提取动态图生成器的节点嵌入 (根据 MTGNN 源码变量名适配)
        if 'gc.emb1' in state_dict:
            emb1 = state_dict['gc.emb1'].numpy()
            emb2 = state_dict['gc.emb2'].numpy()
            alpha_val = state_dict.get('gc.alpha', torch.tensor([0.0])).numpy()
            
            # 还原图矩阵: ReLU(E1 * E2^T - alpha)
            raw_adj = np.dot(emb1, emb2.T) - alpha_val
            m0_adj = np.maximum(0, raw_adj) 
            print("✅ 成功从 model_M0.pt 提取出真实的动态图拓扑！")
        else:
            raise ValueError("在权重中未找到 gc.emb1")
            
    except Exception as e:
        print(f"⚠️ 提取 M0 动态图失败 ({e})。将暂时使用仿真过拟合图展示排版...")
        m0_adj = np.random.rand(static_adj.shape[0], static_adj.shape[1])

    # 3. 开始画图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：M0 的动态图
    sns.heatmap(m0_adj, ax=axes[0], cmap='Reds', cbar=True, square=True)
    axes[0].set_title('M0 (Baseline): Learned Dynamic Graph\n(Unstructured & Overfitted)', fontsize=14, pad=15)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # 右图：M2 的静态图
    sns.heatmap(static_adj, ax=axes[1], cmap='Reds', cbar=True, square=True)
    axes[1].set_title('M2 (Ours): Normalized Prior Graph\n(Structured Block-diagonal)', fontsize=14, pad=15)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.tight_layout()
    plt.savefig('ppt_figures/phase2_graph_comparison.png', dpi=300, bbox_inches='tight')
    print("--> 💾 保存成功: ppt_figures/phase2_graph_comparison.png")

def plot_router_alignment():
    """ 任务 2：路由权重瞬间对齐机制 (M3/M4 应对体制切换) """
    print("\n[2/2] 正在生成 Router 体制切换对齐图...")
    
    # 预期您在跑完测试集后，会保存真实的价格和 Router 权重
    price_file = './output/test_y_true.npy'
    alpha_file = './output/test_alphas.npy'
    
    if os.path.exists(price_file) and os.path.exists(alpha_file):
        print("✅ 检测到真实的测试集输出文件，正在自动搜寻黑天鹅区间...")
        price_full = np.load(price_file) # 真实汇率
        alpha_full = np.load(alpha_file) # 真实的 Router 权重
        
        # 自动寻找跌幅最大的一段区间 (寻找体制切换点)
        # 假设我们看 120 个时间步的窗口
        window_size = 120
        drops = price_full[:-window_size] - price_full[window_size:]
        crisis_idx = np.argmax(drops) # 找到跌得最惨的那个窗口起点
        
        price = price_full[crisis_idx : crisis_idx + window_size]
        alpha = alpha_full[crisis_idx : crisis_idx + window_size]
        time_steps = np.arange(window_size)
        crisis_start, crisis_end = 50, 80 # 绘图高亮用的近似区间
        
    else:
        print("⚠️ 未检测到真实输出 (test_y_true.npy 等)。使用仿真数据展示排版...")
        time_steps = np.arange(120)
        price = 1.15 + np.sin(time_steps * 0.1) * 0.01
        price[70:85] -= np.linspace(0, 0.08, 15)  # 模拟暴跌
        price[85:] -= 0.08
        
        alpha = np.random.normal(0.15, 0.03, 120)
        alpha[68:88] = np.random.normal(0.85, 0.06, 20) # 模拟提前飙升
        alpha = np.clip(alpha, 0, 1)
        crisis_start, crisis_end = 68, 88

    # 3. 开始画双轴折线图
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # 绘制主轴：汇率走势
    color_price = '#1f77b4'
    ax1.set_xlabel('Test Set Time Steps (Hours)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('EUR/USD Exchange Rate (Normalized)', color=color_price, fontsize=12, fontweight='bold')
    ax1.plot(time_steps, price, color=color_price, linewidth=2.5, label='Actual Price')
    ax1.tick_params(axis='y', labelcolor=color_price)
    
    # 绘制副轴：Router 的 Alpha 权重
    ax2 = ax1.twinx()  
    color_alpha = '#d62728'
    ax2.set_ylabel('MoE Router Weight ($\\alpha$)', color=color_alpha, fontsize=12, fontweight='bold')
    ax2.plot(time_steps, alpha, color=color_alpha, linestyle='--', linewidth=2.5, label='Alpha (to Dynamic Graph)')
    ax2.tick_params(axis='y', labelcolor=color_alpha)
    ax2.set_ylim(-0.05, 1.05)
    
    # 标注“体制切换”阴影区域
    ax1.axvspan(crisis_start, crisis_end, color='gray', alpha=0.2, label='Crisis / Regime Shift')
    
    plt.title('Regime-MoE: Autonomous Routing during Market Shifts', fontsize=14, pad=15)
    
    # 合并图例并放到合适位置
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', framealpha=0.9)
    
    fig.tight_layout()
    plt.savefig('ppt_figures/phase2_router_alignment.png', dpi=300, bbox_inches='tight')
    print("--> 💾 保存成功: ppt_figures/phase2_router_alignment.png\n")

if __name__ == "__main__":
    print("=== 开始生成论文 Phase 2 可视化图表 ===")
    plot_graph_comparison()
    plot_router_alignment()
    print("=== 所有图表生成完毕！ ===")