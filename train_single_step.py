import argparse
import math
import time
import os
import torch
import torch.nn as nn
from net import gtnet
import numpy as np
import importlib
import pickle
from util import DataLoaderS
from trainer import Optim

# ==========================================
# 1. 升级版 Evaluate 函数 (全指标)
# ==========================================
def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size,
             save_dir=None, run_id=None):
    """
    评估函数。
    save_dir + run_id 均不为 None 时，将预测差分（diff_pred, diff_true）
    保存到 save_dir/diff_pred_run{run_id}.npy 供 backtest.py 使用。
    新增指标：
      DA   - 全样本方向准确率
      tDA  - 高置信样本方向准确率（|diff_pred| > median(|diff_true|)）
    """
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    total_mape = 0
    total_da_correct  = 0
    total_tda_correct = 0
    total_tda_count   = 0
    n_samples = 0
    predict = None
    test    = None

    diff_pred_batches = []
    diff_true_batches = []

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        if len(output.shape) == 1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict, test = output, Y
        else:
            predict = torch.cat((predict, output))
            test    = torch.cat((test, Y))

        scale     = data.scale.expand(output.size(0), data.m)
        pred_real = output * scale          # (B, M) 真实空间预测
        true_real = Y * scale               # (B, M) 真实空间标签

        total_loss    += evaluateL2(pred_real, true_real).item()
        total_loss_l1 += evaluateL1(pred_real, true_real).item()
        mape_batch     = torch.abs((pred_real - true_real) / (true_real.abs() + 1e-5))
        total_mape    += torch.sum(mape_batch).item()

        # ── 方向指标 ─────────────────────────────────────────────────
        last_price = X[:, 0, :, -1] * scale    # 输入窗口最后一个价格
        diff_pred  = pred_real - last_price     # 预测涨跌量
        diff_true  = true_real - last_price     # 真实涨跌量

        # 全样本 DA
        correct_mask = (diff_pred * diff_true) > 0
        total_da_correct += correct_mask.sum().item()

        # 高置信 tDA：只统计 |diff_pred| > 该 batch 的 median(|diff_true|)
        threshold = torch.abs(diff_true).median()
        conf_mask = torch.abs(diff_pred) > threshold
        total_tda_correct += (correct_mask & conf_mask).sum().item()
        total_tda_count   += conf_mask.sum().item()

        # 收集差分供回测保存
        diff_pred_batches.append(diff_pred.detach().cpu().numpy())
        diff_true_batches.append(diff_true.detach().cpu().numpy())

        n_samples += output.size(0) * data.m

    # ── 统计指标 ─────────────────────────────────────────────────────
    rse  = math.sqrt(total_loss / n_samples) / data.rse
    rae  = (total_loss_l1 / n_samples) / data.rae
    rmse = math.sqrt(total_loss / n_samples)
    mae  = total_loss_l1 / n_samples
    mape = total_mape / n_samples
    da   = total_da_correct / n_samples
    tda  = total_tda_correct / max(total_tda_count, 1)

    predict = predict.data.cpu().numpy()
    Ytest   = test.data.cpu().numpy()
    sigma_p = predict.std(axis=0)
    sigma_g = Ytest.std(axis=0)
    mean_p  = predict.mean(axis=0)
    mean_g  = Ytest.mean(axis=0)
    index   = sigma_g != 0
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = correlation[index].mean()

    r2_list = []
    for i in range(predict.shape[1]):
        y_t = Ytest[:, i];  y_p = predict[:, i]
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2_list.append(0.0 if ss_tot < 1e-5 else 1 - ss_res / ss_tot)
    r2 = np.mean(r2_list)

    # ── 保存预测差分（仅最终测试评估时） ──────────────────────────────
    if save_dir is not None and run_id is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/diff_pred_run{run_id}.npy",
                np.vstack(diff_pred_batches))
        np.save(f"{save_dir}/diff_true_run{run_id}.npy",
                np.vstack(diff_true_batches))

    return rse, rae, correlation, mae, rmse, mape, r2, da, tda

# ==========================================
# 2. Train 函数
# ==========================================
def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        if iter % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]
            id = torch.tensor(id).to(device)
            tx = X[:, :, id, :]
            ty = Y[:, id]
            output = model(tx,id)
            output = torch.squeeze(output)
            scale = data.scale.expand(output.size(0), data.m)
            scale = scale[:,id]
            # 1. 主损失 (MAE)
            base_loss = criterion(output * scale, ty * scale)
            loss = base_loss

            # 2. 波动率加权方向性损失（DirLoss）
            # 核心思想：高波动时段方向预测更有意义（收益更大），加大惩罚权重；
            #          低波动时段价格几乎不动，方向本就难预测，降低权重避免干扰
            if args.use_dirloss == 1:
                last_price = tx[:, 0, :, -1] * scale
                diff_pred  = (output * scale) - last_price
                diff_true  = (ty * scale) - last_price

                # 每个货币对的 batch 平均绝对波动率（做量纲归一化）
                volatility = torch.abs(diff_true).mean(dim=0, keepdim=True).detach() + 1e-5
                logits     = diff_pred / volatility

                true_bin = (diff_true > 0).float().detach()

                # 样本级动态权重：波动率大的时段权重高（sigmoid 将绝对波动率映射到 (0,1)）
                sample_vol   = torch.abs(diff_true).mean(dim=1, keepdim=True).detach()
                sample_weight = torch.sigmoid(sample_vol / (sample_vol.mean() + 1e-5)).detach()
                elem_loss    = nn.functional.binary_cross_entropy_with_logits(
                    logits, true_bin, reduction='none')
                dir_loss = (sample_weight * elem_loss).mean()
                loss = loss + args.dir_weight * dir_loss

            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * data.m)
            grad_norm = optim.step()

        if iter%100==0:
            print('iter:{:3d} | loss: {:.3f}'.format(iter,loss.item()/(output.size(0) * data.m)))
        iter += 1
    return total_loss / n_samples

# ==========================================
# 3. 参数配置
# ==========================================
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='./data/solar_AL.txt', help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default='model/model.pt', help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device',type=str,default=None,help='Device to use (cuda/cpu/mps). If not specified, will auto-detect: mps for Mac, cuda if available, else cpu')
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=137,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=20,help='k')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=2,help='dilation exponential')
parser.add_argument('--conv_channels',type=int,default=16,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=16,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=32,help='skip channels')
parser.add_argument('--end_channels',type=int,default=64,help='end channels')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=24*7,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=1,help='output sequence length')
parser.add_argument('--horizon', type=int, default=3)
parser.add_argument('--layers',type=int,default=5,help='number of layers')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--lr',type=float,default=0.0001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=5,help='clip')
parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='tanh alpha')
parser.add_argument('--epochs',type=int,default=1,help='')
parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--step_size',type=int,default=100,help='step_size')
parser.add_argument('--revin', type=int, default=1, help='1 to use RevIN, 0 to disable')
parser.add_argument('--dual_graph', type=int, default=1, help='1 to use Dual Graph, 0 to disable')
parser.add_argument('--adj_data', type=str, default='./data/sensor_graph/adj_mx.pkl', help='path to static graph')
parser.add_argument('--use_router', type=int, default=1, help='1 to use Router, 0 to disable')
parser.add_argument('--use_dirloss', type=int, default=1, help='1 to use DirLoss, 0 to disable')
parser.add_argument('--dir_weight', type=float, default=0.05, help='Weight lambda for Directional Loss')
# 新增 runs 参数
parser.add_argument('--runs', type=int, default=10, help='number of runs to average')

args = parser.parse_args()

# 自动检测设备
if args.device is None:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
else:
    device = torch.device(args.device)
    print(f"Using specified device: {device}")

torch.set_num_threads(3)

def main(run_id):
    # Data Loader 初始化
    Data = DataLoaderS(args.data, 0.6, 0.2, device, args.horizon, args.seq_in_len, args.normalize)

# === 创新点2：加载静态图代码开始 ===
    predefined_A = None
    if args.dual_graph == 1:
        print(f"Loading static graph from: {args.adj_data}")
        try:
            with open(args.adj_data, 'rb') as f:
                adj_mx = pickle.load(f)
            
            # 1. 转为 Tensor 并送到 GPU
            predefined_A = torch.tensor(adj_mx, dtype=torch.float32).to(device)
            
            # 2. 【终极工程修复】：强制行归一化 (Row Normalization)
            # 计算每行的和，并防止除以0
            row_sums = predefined_A.sum(dim=1, keepdim=True)
            row_sums[row_sums == 0] = 1.0 
            
            # 将图矩阵除以行和，保证每行权重相加为1，彻底杜绝特征爆炸！
            predefined_A = predefined_A / row_sums
            
            print("Static graph loaded and normalized successfully.")
        except Exception as e:
            print(f"Failed to load static graph: {e}")
            print("Fallback: Dual Graph will be disabled (predefined_A=None).")
            predefined_A = None


    # 模型初始化
    model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                  device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, 
                  layer_norm_affline=False, 
                  revin=(args.revin == 1),
                  dual_graph=(args.dual_graph == 1),
                  use_router=(args.use_router == 1),
                  predefined_A=predefined_A)
    model = model.to(device)

    # Loss 设置
    if args.L1Loss:
        criterion = nn.L1Loss(reduction='sum').to(device)
    else:
        criterion = nn.MSELoss(reduction='sum').to(device)
    evaluateL2 = nn.MSELoss(reduction='sum').to(device)
    evaluateL1 = nn.L1Loss(reduction='sum').to(device)

    best_val = 10000000
    optim = Optim(
        model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay
    )

    print(f'\n>>> Run {run_id+1}/{args.runs} Begin Training...')
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_loss = train(Data, Data.train[0], Data.train[1],
                               model, criterion, optim, args.batch_size)

            val_rse, val_rae, val_corr, val_mae, val_rmse, val_mape, val_r2, val_da, val_tda = \
                evaluate(Data, Data.valid[0], Data.valid[1],
                         model, evaluateL2, evaluateL1, args.batch_size)

            print(
                '| end of epoch {:3d} | time: {:5.2f}s | loss {:5.4f} | rse {:5.4f} '
                '| mae {:5.4f} | rmse {:5.4f} | r2 {:5.4f} | da {:5.4f} | tda {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss,
                    val_rse, val_mae, val_rmse, val_r2, val_da, val_tda), flush=True)

            if val_rse < best_val:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val = val_rse

            if epoch % 5 == 0:
                test_rse, test_rae, test_corr, test_mae, test_rmse, test_mape, test_r2, test_da, test_tda = \
                    evaluate(Data, Data.test[0], Data.test[1],
                             model, evaluateL2, evaluateL1, args.batch_size)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # ── 加载最佳模型，最终测试 + 保存预测数据 ────────────────────────
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    # 根据 --save 路径自动推导输出目录，例如 ./model/model_M3.pt → ./output/model_M3/
    model_name = os.path.splitext(os.path.basename(args.save))[0]
    pred_save_dir = os.path.join('./output', model_name)

    test_rse, test_rae, test_corr, test_mae, test_rmse, test_mape, test_r2, test_da, test_tda = \
        evaluate(Data, Data.test[0], Data.test[1],
                 model, evaluateL2, evaluateL1, args.batch_size,
                 save_dir=pred_save_dir, run_id=run_id)

    print(f"Run {run_id+1} Final Test: RSE {test_rse:.4f} | MAE {test_mae:.4f} | "
          f"RMSE {test_rmse:.4f} | R2 {test_r2:.4f} | DA {test_da:.4f} | tDA {test_tda:.4f}")

    if args.dual_graph == 1 and hasattr(model, 'fusion_weight'):
        print(f"Model Learned Fusion Weight: {model.fusion_weight.item():.4f}")

    return test_rse, test_rae, test_corr, test_mae, test_rmse, test_mape, test_r2, test_da, test_tda

if __name__ == "__main__":
    results = {
        'rse': [], 'rae': [], 'corr': [],
        'mae': [], 'rmse': [], 'mape': [], 'r2': [], 'da': [], 'tda': []
    }

    for i in range(args.runs):
        rse, rae, corr, mae, rmse, mape, r2, da, tda = main(i)
        results['rse'].append(rse);   results['rae'].append(rae)
        results['corr'].append(corr); results['mae'].append(mae)
        results['rmse'].append(rmse); results['mape'].append(mape)
        results['r2'].append(r2);     results['da'].append(da)
        results['tda'].append(tda)

    print('\n' + '='*50)
    print(f'Summary over {args.runs} runs')
    print('='*50)
    print(f"{'Metric':<10} | {'Mean':<10} | {'Std':<10}")
    print('-'*36)

    for key in ['mae', 'rmse', 'mape', 'r2', 'rse', 'corr', 'da', 'tda']:
        mean_val = np.mean(results[key])
        std_val  = np.std(results[key])
        print(f"{key.upper():<10} | {mean_val:<10.4f} | {std_val:<10.4f}")

    print('='*50)