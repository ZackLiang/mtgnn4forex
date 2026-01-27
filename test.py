import torch
import torch.nn as nn

# 模拟一个 (Batch=2, Channel=32, Node=10, Time=168) 的数据
x = torch.randn(2, 32, 10, 168)

# 变换到频域
x_fft = torch.fft.rfft(x, dim=-1)
print(f"频域形状: {x_fft.shape}") # 应该是 (2, 32, 10, 85)

# 逆变换回来
x_back = torch.fft.irfft(x_fft, n=168, dim=-1)
print(f"还原误差: {(x - x_back).abs().sum().item()}") # 应该接近 0