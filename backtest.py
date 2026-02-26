"""
backtest.py
外汇策略回测引擎。

用途：
  消融实验结束后，加载各模型保存的预测差分（diff_pred, diff_true），
  按"预测方向做多/做空"策略，计算金融指标并绘制净值曲线。

使用方式：
  python backtest.py                        # 使用 output/ 下的真实预测
  python backtest.py --demo                 # 使用仿真数据（预测不存在时自动降级）

输出：
  - 控制台：金融指标消融表（年化收益、Sharpe、最大回撤、胜率、tDA）
  - ppt_figures/equity_ablation.png：净值曲线图
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

os.makedirs('ppt_figures', exist_ok=True)
os.makedirs('output', exist_ok=True)

# ── 模型配置 ─────────────────────────────────────────────────────────
MODEL_CFG = [
    dict(key='M0', label='M0  Baseline',         dir='output/model_M0',  color='gray',    lw=1.5, alpha=0.7),
    dict(key='M1', label='M1  +RevIN',            dir='output/model_M1',  color='#aec7e8', lw=1.5, alpha=0.8),
    dict(key='M2', label='M2  +RegimeMoE(rand)',  dir='output/model_M2',  color='#1f77b4', lw=2.0, alpha=0.9),
    dict(key='M3', label='M3  +RegimeMoE(Granger)',dir='output/model_M3', color='#ff7f0e', lw=2.0, alpha=0.9),
    dict(key='M4', label='M4  Ours',              dir='output/model_M4',  color='#d62728', lw=2.5, alpha=1.0),
]


# ── 核心回测函数 ─────────────────────────────────────────────────────
def run_backtest(diff_true: np.ndarray, diff_pred: np.ndarray,
                 last_price: np.ndarray = None, cost_bps: float = 1.0) -> dict:
    """
    向量化回测引擎（多资产组合版）。

    参数:
        diff_true  : shape (N, M)，真实价格变动（绝对值）
        diff_pred  : shape (N, M)，预测价格变动（绝对值）
        last_price : shape (N, M)，上一期价格（用于换算相对收益率）
                     若为 None 则直接用 diff 的绝对量计算
        cost_bps   : 单边摩擦成本（基点，默认 1 bps）

    返回:
        包含各金融指标的字典
    """
    cost = cost_bps / 10000.0

    # 如果提供了 last_price，换算为相对收益率；否则用绝对变动
    if last_price is not None and last_price.shape == diff_true.shape:
        ref = np.abs(last_price) + 1e-8
        true_ret  = diff_true / ref      # (N, M)
        pred_ret  = diff_pred / ref      # (N, M) 用于方向信号
    else:
        true_ret  = diff_true
        pred_ret  = diff_pred

    # ── 方向信号：预测涨(>0) → +1 做多，否则 -1 做空 ─────────────────
    signals = np.where(pred_ret > 0, 1.0, -1.0)        # (N, M)

    # ── 策略收益（每资产独立运行，取等权组合）─────────────────────────
    strat_ret  = signals * true_ret                     # (N, M)
    turns      = np.abs(np.diff(signals, axis=0, prepend=0)) / 2
    net_ret    = strat_ret - turns * cost               # (N, M)

    # ── 等权组合 ─────────────────────────────────────────────────────
    port_ret   = net_ret.mean(axis=1)                   # (N,)
    equity     = np.cumprod(1.0 + port_ret)

    # ── 金融指标 ─────────────────────────────────────────────────────
    total_return = equity[-1] - 1.0

    running_max  = np.maximum.accumulate(equity)
    mdd          = np.min((equity - running_max) / running_max)

    # 年化 Sharpe（小时级数据：252 × 24 小时 / 年）
    sharpe = (np.sqrt(252 * 24) * port_ret.mean()
              / (port_ret.std() + 1e-9))

    win_rate = float(np.mean(net_ret > 0))

    # ── 高置信方向准确率（tDA） ───────────────────────────────────────
    thr       = np.median(np.abs(diff_true))
    conf_mask = np.abs(diff_pred) > thr
    correct   = (diff_pred * diff_true) > 0
    tda = (correct & conf_mask).sum() / max(conf_mask.sum(), 1)

    return dict(equity=equity, port_ret=port_ret,
                total_return=total_return, mdd=mdd,
                sharpe=sharpe, win_rate=win_rate, tda=tda)


# ── 加载模型预测数据 ──────────────────────────────────────────────────
def load_preds(pred_dir: str, n_runs: int = 3) -> tuple:
    """
    读取 diff_pred_run{i}.npy 和 diff_true_run{i}.npy，
    返回多 run 平均的 (diff_pred, diff_true)。
    """
    preds, trues = [], []
    for i in range(n_runs):
        fp = os.path.join(pred_dir, f'diff_pred_run{i}.npy')
        ft = os.path.join(pred_dir, f'diff_true_run{i}.npy')
        if os.path.exists(fp) and os.path.exists(ft):
            preds.append(np.load(fp))
            trues.append(np.load(ft))
    if not preds:
        return None, None
    # 取最短公共长度后平均
    min_len = min(p.shape[0] for p in preds)
    diff_pred = np.mean([p[:min_len] for p in preds], axis=0)
    diff_true = np.mean([t[:min_len] for t in trues], axis=0)
    return diff_pred, diff_true


# ── 仿真数据（文件不存在时回退）────────────────────────────────────────
def make_demo_data(n: int = 4000, m: int = 31, seed: int = 42) -> dict:
    """生成贴合逻辑的仿真数据（仅用于排版测试）"""
    np.random.seed(seed)
    base = np.random.normal(0, 0.001, (n, m))
    base[2000:2050] -= 0.008   # 模拟一次剧烈波动
    demo = {}
    demo['M0'] = (base, base * 0.1  + np.random.normal(0, 0.002, (n, m)))
    demo['M1'] = (base, base * 0.4  + np.random.normal(0, 0.001, (n, m)))
    demo['M2'] = (base, base * 0.55 + np.random.normal(0, 0.0008, (n, m)))
    demo['M3'] = (base, base * 0.65 + np.random.normal(0, 0.0006, (n, m)))
    demo['M4'] = (base, base * 0.72 + np.random.normal(0, 0.0005, (n, m)))
    return demo


# ── 主流程 ────────────────────────────────────────────────────────────
def perform_financial_ablation(force_demo: bool = False, n_runs: int = 3,
                                cost_bps: float = 1.0):
    print("="*65)
    print("  Regime-MoE-GNN 金融回测消融实验")
    print(f"  摩擦成本: {cost_bps} bps / 单边")
    print("="*65)

    results = {}
    demo_data = None
    any_real  = False

    for cfg in MODEL_CFG:
        key  = cfg['key']
        if not force_demo:
            dp, dt = load_preds(cfg['dir'], n_runs)
        else:
            dp, dt = None, None

        if dp is None:
            if demo_data is None:
                demo_data = make_demo_data()
                print("⚠️  未找到真实预测文件，使用仿真数据（仅供排版）")
            dt, dp = demo_data[key]
        else:
            any_real = True
            print(f"✅ 已加载 {key} 真实预测  shape={dp.shape}")

        results[key] = run_backtest(dt, dp, cost_bps=cost_bps)
        results[key]['label'] = cfg['label']

    if any_real:
        print("\n（以上为真实模型预测的回测结果）")

    # ── 打印指标表 ────────────────────────────────────────────────────
    W = 80
    print("\n" + "="*W)
    header = (f"{'模型':<26} | {'年化收益':>10} | {'最大回撤':>10} "
              f"| {'Sharpe':>8} | {'胜率':>8} | {'tDA':>8}")
    print(header)
    print("-"*W)
    for cfg in MODEL_CFG:
        r = results[cfg['key']]
        print(f"{cfg['label']:<26} | {r['total_return']:>9.2%} "
              f"| {r['mdd']:>9.2%} | {r['sharpe']:>7.2f} "
              f"| {r['win_rate']:>7.2%} | {r['tda']:>7.2%}")
    print("="*W)

    # ── 绘制净值曲线 ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    for cfg in MODEL_CFG:
        r = results[cfg['key']]
        lbl = f"{cfg['label']}  (Sharpe {r['sharpe']:.2f}, MDD {r['mdd']:.1%})"
        ax.plot(r['equity'], label=lbl,
                color=cfg['color'], lw=cfg['lw'], alpha=cfg['alpha'])

    ax.axhline(1.0, color='black', lw=0.8, linestyle='--', alpha=0.4)
    ax.set_title('Out-of-Sample Equity Curve  (1 bps friction cost)',
                 fontsize=14, pad=12)
    ax.set_xlabel('Time Step (Test Set)', fontsize=11)
    ax.set_ylabel('Cumulative Equity', fontsize=11)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    save_path = 'ppt_figures/equity_ablation.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n净值曲线已保存: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo',      action='store_true',
                        help='强制使用仿真数据（排版测试用）')
    parser.add_argument('--runs',      type=int, default=3,
                        help='每个模型的 run 数，用于平均预测')
    parser.add_argument('--cost_bps',  type=float, default=1.0,
                        help='单边摩擦成本（基点）')
    args = parser.parse_args()
    perform_financial_ablation(force_demo=args.demo,
                                n_runs=args.runs,
                                cost_bps=args.cost_bps)
