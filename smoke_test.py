"""
smoke_test.py
快速验证新消融设计（M0-M4）：
  - 代码无报错
  - M2 和 M3 使用相同架构（RegimeMoE），仅静态图来源不同
  - 验证趋势：M0 >> M1 >= M2 >= M3 >= M4
只跑 N_BATCHES 个 batch，几分钟内完成
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import time

from net import gtnet
from util import DataLoaderS

# ====== 配置 ======
DATA_PATH      = "./data/G31_RawPrice.txt"
ADJ_GRANGER    = "./data/sensor_graph/adj_mx.pkl"       # 格兰杰因果图
ADJ_RANDOM     = "./data/sensor_graph/adj_random.pkl"   # 随机图
NUM_NODES      = 31
SEQ_IN_LEN     = 168
HORIZON        = 3
BATCH_SIZE     = 32
N_BATCHES      = 50
DEVICE         = torch.device("cpu")
SEED           = 42
# ==================

def load_adj(path):
    with open(path, "rb") as f:
        adj = pickle.load(f)
    A = torch.tensor(adj, dtype=torch.float32).to(DEVICE)
    row_sums = A.sum(dim=1, keepdim=True).clamp(min=1.0)
    return A / row_sums

def run_model(name, revin, dual_graph, use_router, use_dirloss, normalize,
              adj_path=None, dir_weight=0.05):
    print(f"\n{'='*58}")
    print(f"  {name}")
    print(f"  revin={revin}  dual_graph={dual_graph}  use_router={use_router}"
          f"  dirloss={use_dirloss}  normalize={normalize}")
    if adj_path:
        print(f"  adj={adj_path.split('/')[-1]}")
    print(f"{'='*58}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    Data = DataLoaderS(DATA_PATH, 0.6, 0.2, DEVICE, HORIZON, SEQ_IN_LEN, normalize)
    predefined_A = load_adj(adj_path) if (dual_graph and adj_path) else None

    model = gtnet(
        gcn_true=True, buildA_true=True, gcn_depth=2,
        num_nodes=NUM_NODES, device=DEVICE,
        dropout=0.3, subgraph_size=20, node_dim=40,
        dilation_exponential=2, conv_channels=16,
        residual_channels=16, skip_channels=32, end_channels=64,
        seq_length=SEQ_IN_LEN, in_dim=1, out_dim=1,
        layers=5, propalpha=0.05, tanhalpha=3,
        layer_norm_affline=False,
        revin=revin, dual_graph=dual_graph,
        use_router=use_router,
        predefined_A=predefined_A,
    ).to(DEVICE)

    criterion = nn.L1Loss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    losses = []
    alphas = []
    perm = np.random.permutation(range(NUM_NODES))
    t0 = time.time()

    for i, (X, Y) in enumerate(
            Data.get_batches(Data.train[0], Data.train[1], BATCH_SIZE, True)):
        if i >= N_BATCHES:
            break

        model.zero_grad()
        X   = torch.unsqueeze(X, dim=1).transpose(2, 3)
        idx = torch.tensor(perm).to(DEVICE)
        tx, ty = X[:, :, idx, :], Y[:, idx]

        output = model(tx, idx)
        output = torch.squeeze(output)
        if len(output.shape) == 1:
            output = output.unsqueeze(0)

        scale      = Data.scale.expand(output.size(0), Data.m)[:, idx]
        base_loss  = criterion(output * scale, ty * scale)
        loss       = base_loss

        if use_dirloss:
            last_price   = tx[:, 0, :, -1] * scale
            diff_pred    = (output * scale) - last_price
            diff_true    = (ty * scale) - last_price
            volatility   = torch.abs(diff_true).mean(dim=0, keepdim=True).detach() + 1e-5
            logits       = diff_pred / volatility
            true_bin     = (diff_true > 0).float().detach()
            sample_vol   = torch.abs(diff_true).mean(dim=1, keepdim=True).detach()
            sample_w     = torch.sigmoid(sample_vol / (sample_vol.mean() + 1e-5)).detach()
            dir_loss     = (sample_w * F.binary_cross_entropy_with_logits(
                logits, true_bin, reduction='none')).mean()
            loss = loss + dir_weight * dir_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        losses.append(loss.item() / (output.size(0) * Data.m))

        if use_router and hasattr(model, 'last_alpha') and model.last_alpha is not None:
            alphas.append(float(model.last_alpha.mean()))

    elapsed    = time.time() - t0
    avg_loss   = float(np.mean(losses))
    alpha_str  = f"  avg_alpha={np.mean(alphas):.4f}  range=[{min(alphas):.3f},{max(alphas):.3f}]" \
                 if alphas else ""
    print(f"  ✅ {N_BATCHES} batches OK | avg_loss={avg_loss:.6f}{alpha_str} | {elapsed:.0f}s")
    return avg_loss


if __name__ == "__main__":
    print(f"\n{'#'*58}")
    print(f"  SMOKE TEST  (device=cpu, {N_BATCHES} batches, seed={SEED})")
    print(f"  新消融设计验证")
    print(f"{'#'*58}")

    r = {}
    r["M0"] = run_model("M0  Baseline",
                         revin=False, dual_graph=False, use_router=False,
                         use_dirloss=False, normalize=2)
    r["M1"] = run_model("M1  + RevIN",
                         revin=True,  dual_graph=False, use_router=False,
                         use_dirloss=False, normalize=0)
    r["M2"] = run_model("M2  + RegimeMoE + 随机图（对照）",
                         revin=True,  dual_graph=True,  use_router=True,
                         use_dirloss=False, normalize=0, adj_path=ADJ_RANDOM)
    r["M3"] = run_model("M3  + RegimeMoE + 格兰杰图（关键）",
                         revin=True,  dual_graph=True,  use_router=True,
                         use_dirloss=False, normalize=0, adj_path=ADJ_GRANGER)
    r["M4"] = run_model("M4  + DirLoss（完全体）",
                         revin=True,  dual_graph=True,  use_router=True,
                         use_dirloss=True,  normalize=0, adj_path=ADJ_GRANGER)

    print(f"\n{'='*58}")
    print("  SUMMARY")
    print(f"{'='*58}")
    for k, v in r.items():
        print(f"  {k}: {v:.6f}")

    m0, m1, m2, m3, m4 = r["M0"], r["M1"], r["M2"], r["M3"], r["M4"]
    print(f"\n  趋势检验（{N_BATCHES} batches 早期收敛信号）:")
    print(f"  M0 >> M1  : {'✅' if m0 > m1 * 2 else '⚠️ '}"
          f"  {m0:.4f} → {m1:.4f}  (↓{100*(m0-m1)/m0:.1f}%)")
    print(f"  M1 >= M2  : {'✅' if m1 >= m2 * 0.98 else '⚠️ '}"
          f"  {m1:.4f} → {m2:.4f}  (↓{100*(m1-m2)/m1:.1f}%)")
    print(f"  M2 >= M3  : {'✅' if m2 >= m3 * 0.98 else '⚠️ '}"
          f"  {m2:.4f} → {m3:.4f}  (↓{100*(m2-m3)/m2:.1f}%)")
    print(f"  M3 >= M4  : {'✅' if m3 >= m4 * 0.98 else '⚠️ '}"
          f"  {m3:.4f} → {m4:.4f}  (↓{100*(m3-m4)/m3:.1f}%)")
    print(f"\n  M2 vs M3 架构相同，仅静态图不同 → 差异 = 格兰杰因果先验的贡献")
