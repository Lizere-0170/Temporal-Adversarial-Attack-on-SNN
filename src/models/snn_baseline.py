# src/models/snn_baseline.py
"""
Baseline SNN using SpikingJelly's LIFNode
----------------------------------------
This version replaces the manual LIFLayer with SpikingJelly's built-in neuron.LIFNode.

Advantages:
- Much faster GPU performance (vectorized)
- Stable surrogate gradient training
- Easy extension to multi-layer or convolutional SNNs
- Clean integration with your SHD preprocessing (spike tensors [T, B, C])

Author: Lizere Yun, 2025
"""

import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron, functional

class BaselineSNN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_neurons: int = 512,
                 n_classes: int = 10,
                 tau_mem: float = 20.0,
                 dt: float = 1.0,
                 surrogate: str = 'SoftSign',
                 detach_reset: bool = True,
                 device: str = 'cpu'):
        """
        Parameters
        ----------
        in_channels : 输入通道数（SHD约700）
        hidden_neurons : 隐藏层神经元数量
        n_classes : 分类数（默认10）
        tau_mem : LIF膜电位时间常数
        dt : 时间步宽
        surrogate : surrogate 梯度函数 ('ATan', 'Sigmoid', 'Triangle', 'Exp')
        detach_reset : 是否在reset时截断梯度（推荐True）
        device : 设备 ('cpu' 或 'cuda')
        """
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.hidden_neurons = hidden_neurons
        self.n_classes = n_classes

        # 线性层
        self.fc1 = nn.Linear(in_channels, hidden_neurons)
        self.norm1 = nn.LayerNorm(hidden_neurons)
        self.dropout1 = nn.Dropout(p=0.1)

        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.norm2 = nn.LayerNorm(hidden_neurons)
        self.dropout2 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(hidden_neurons, n_classes)

        # 选择 surrogate 函数
        surrogate_fn_map = {
            'ATan': neuron.surrogate.ATan(),
            'Sigmoid': neuron.surrogate.Sigmoid(),
            'PiecewiseQuadratic': neuron.surrogate.PiecewiseQuadratic(),
            'SoftSign': neuron.surrogate.SoftSign(),
            'Erf': neuron.surrogate.Erf(),
        }
        surrogate_fn = surrogate_fn_map.get(surrogate, neuron.surrogate.SoftSign())

        # 使用 SpikingJelly 的 LIF 神经元
        self.lif1 = neuron.LIFNode(
            tau=tau_mem,
            surrogate_function=surrogate_fn,
            detach_reset=detach_reset
        )

        self.lif2 = neuron.LIFNode(
            tau=tau_mem,
            surrogate_function=surrogate_fn,
            detach_reset=detach_reset
        )

        self.dt = dt
        self.tau_mem = tau_mem

    def forward(self, x: torch.Tensor):
        """
        Forward pass for time-driven SNN.
        Input:
            x: Tensor [T, B, C] (spike trains over time)
        Returns:
            logits: Tensor [B, n_classes]
        """
        # 初始化膜电位
        functional.reset_net(self)

        T, B, C = x.shape
        outputs = []

        for t in range(T):
            # 输入 -> 线性 -> LIF -> spike
            z1 = self.fc1(x[t])        # [B, hidden]
            z1 = self.norm1(z1)
            s1 = self.lif1(z1)         # LIF1 放电
            s1 = self.dropout1(s1)

            # 第二层
            z2 = self.fc2(s1)          # [B, hidden]
            z2 = self.norm2(z2)
            s2 = self.lif2(z2)         # LIF2 放电
            s2 = self.dropout2(s2)

            outputs.append(s2)

        # 时间聚合（rate coding）
        spk_sum = torch.stack(outputs, dim=0).sum(dim=0)  # [B, hidden]
        logits = self.fc3(spk_sum)
        return logits
