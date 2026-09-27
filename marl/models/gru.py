from __future__ import annotations

import torch
from torch import Tensor, nn

from marl.models.base import BackboneOutput, BaseBackbone


class GRUBackbone(BaseBackbone):
    """输入 [...,T,input_dim]，输出 [...,T,hidden_dim] 的循环特征。

    PyTorch hidden 使用 [num_layers,prod(前置批维),hidden_dim]，不是 [...,T,H]。
    调用者负责 episode reset 时清空 hidden，并按连续序列采样；现有展平后随机
    打乱的 PPO mini-batch 不会自动变成 recurrent PPO。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 1) -> None:
        super().__init__(input_dim, hidden_dim)
        self.hidden_dim, self.num_layers = hidden_dim, num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(
        self, inputs: Tensor, hidden_state: Tensor | None = None, **kwargs: Tensor
    ) -> BackboneOutput:
        self._validate_input(inputs)
        if inputs.ndim < 2:
            raise ValueError("GRU 输入至少需要 [time, feature] 两个维度")
        leading_shape, time_steps = inputs.shape[:-2], inputs.shape[-2]
        flattened = inputs.reshape(-1, time_steps, self.input_dim)
        features, next_hidden = self.gru(flattened, hidden_state)
        features = features.reshape(*leading_shape, time_steps, self.output_dim)
        return BackboneOutput(features=features, hidden_state=next_hidden)

    def initial_state(self, *batch_shape: int, device=None) -> Tensor:
        flat_batch = 1
        for size in batch_shape:
            flat_batch *= size
        return torch.zeros(self.num_layers, flat_batch, self.hidden_dim, device=device)
