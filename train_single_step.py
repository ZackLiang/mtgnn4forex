import argparse
import math
import time
import torch
import torch.nn as nn
from net import gtnet
import numpy as np
from util import DataLoaderS
from trainer import Optim

# ==========================================
# 1. 升级版 Evaluate 函数 (全指标)
# ==========================================
def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    total_mape = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        if len(output.shape)==1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        # 核心逻辑：无论是 Baseline 还是 RevIN，output * scale 都是真实值
        scale = data.scale.expand(output.size(0), data.m)
        pred_real = output * scale
        true_real = Y * scale
        
        # 累加误差
        total_loss += evaluateL2(pred_real, true_real).item() # MSE Sum
        total_loss_l1 += evaluateL1(pred_real, true_real).item() # MAE Sum
        
        # 计算 MAPE (加 1e-5 防止除零)
        mape_batch = torch.abs((pred_real - true_real) / (true_real + 1e-5))
        total_mape += torch.sum(mape_batch).item()
        
        n_samples += (output.size(0) * data.m)

    # 计算各项指标
    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae
    
    rmse = math.sqrt(total_loss / n_samples)
    mae = total_loss_l1 / n_samples
    mape = total_mape / n_samples

    # 计算 Correlation 和 R2
    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    
    # R2 Score
    ss_res = np.sum((Ytest - predict) ** 2)
    ss_tot = np.sum((Ytest - np.mean(Ytest)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return rse, rae, correlation, mae, rmse, mape, r2

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
            loss = criterion(output * scale, ty * scale)
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

    # 模型初始化
    model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                  device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, 
                  layer_norm_affline=False, 
                  revin=(args.revin == 1))
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
            train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
            
            val_rse, val_rae, val_corr, val_mae, val_rmse, val_mape, val_r2 = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args.batch_size)
            
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | loss {:5.4f} | rse {:5.4f} | mae {:5.4f} | rmse {:5.4f} | r2 {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, val_rse, val_mae, val_rmse, val_r2), flush=True)

            if val_rse < best_val:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val = val_rse
                
            if epoch % 5 == 0:
                test_rse, test_rae, test_corr, test_mae, test_rmse, test_mape, test_r2 = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # 加载最佳模型进行最终测试
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    test_rse, test_rae, test_corr, test_mae, test_rmse, test_mape, test_r2 = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size)
    print(f"Run {run_id+1} Final Test: RSE {test_rse:.4f} | MAE {test_mae:.4f} | RMSE {test_rmse:.4f} | R2 {test_r2:.4f}")
    
    return test_rse, test_rae, test_corr, test_mae, test_rmse, test_mape, test_r2

if __name__ == "__main__":
    # 存储每次运行的结果
    results = {
        'rse': [], 'rae': [], 'corr': [], 
        'mae': [], 'rmse': [], 'mape': [], 'r2': []
    }
    
    for i in range(args.runs):
        rse, rae, corr, mae, rmse, mape, r2 = main(i)
        results['rse'].append(rse)
        results['rae'].append(rae)
        results['corr'].append(corr)
        results['mae'].append(mae)
        results['rmse'].append(rmse)
        results['mape'].append(mape)
        results['r2'].append(r2)
        
    print('\n' + '='*50)
    print(f'Summary over {args.runs} runs')
    print('='*50)
    print(f"{'Metric':<10} | {'Mean':<10} | {'Std':<10}")
    print('-'*36)
    
    for key in ['mae', 'rmse', 'mape', 'r2', 'rse', 'corr']:
        mean_val = np.mean(results[key])
        std_val = np.std(results[key])
        print(f"{key.upper():<10} | {mean_val:<10.4f} | {std_val:<10.4f}")
    
    print('='*50)