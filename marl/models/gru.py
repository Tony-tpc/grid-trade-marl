from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from marl.core.recurrent import RecurrentState, validate_state
from marl.models.base import BackboneOutput, BaseBackbone


class GRUBackbone(BaseBackbone):
    """输入 [...,T,input_dim]，输出 [...,T,hidden_dim] 的循环特征。

    PyTorch hidden 使用 [num_layers,prod(前置批维),hidden_dim]，不是 [...,T,H]。
    调用者负责 reset 时清空 hidden；整条序列交给原生 GRU，不按时间步展开。
    """

    is_recurrent = True

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 1) -> None:
        super().__init__(input_dim, hidden_dim)
        if type(num_layers) is not int or num_layers < 1:
            raise ValueError("num_layers 必须为正整数")
        self.hidden_dim, self.num_layers = hidden_dim, num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(
        self, inputs: Tensor, hidden_state: RecurrentState = None, **kwargs: Tensor
    ) -> BackboneOutput:
        self._validate_input(inputs)
        if inputs.ndim < 2 or inputs.shape[-2] < 1:
            raise ValueError("GRU 输入至少需要 [time, feature] 两个维度")
        leading_shape, time_steps = inputs.shape[:-2], inputs.shape[-2]
        flattened = inputs.reshape(-1, time_steps, self.input_dim)
        validate_state(hidden_state, (self.num_layers, flattened.shape[0], self.hidden_dim),
                       inputs, lstm=False)
        assert hidden_state is None or isinstance(hidden_state, Tensor)
        features, next_hidden = self.gru(flattened, hidden_state)
        features = features.reshape(*leading_shape, time_steps, self.output_dim)
        return BackboneOutput(features=features, hidden_state=next_hidden)

    def initial_state(self, *batch_shape: int, device=None, dtype=None) -> Tensor:
        parameter = next(self.parameters())
        return torch.zeros(self.num_layers, math.prod(batch_shape), self.hidden_dim,
                           device=device or parameter.device, dtype=dtype or parameter.dtype)


@dataclass(frozen=True, slots=True)
class GRUBackboneConfig:
    kind: Literal["gru"] = "gru"
    hidden_dim: int = 128
    num_layers: int = 1

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 1 for v in (self.hidden_dim, self.num_layers)):
            raise ValueError("hidden_dim/num_layers 必须为正整数")

    def build(self, input_dim: int) -> GRUBackbone:
        return GRUBackbone(input_dim, self.hidden_dim, self.num_layers)
