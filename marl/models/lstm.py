"""原生 LSTM 的形状适配；循环门与反向传播交给 PyTorch。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from marl.core.recurrent import RecurrentState, validate_state
from marl.models.base import BackboneOutput, BaseBackbone


class LSTMBackbone(BaseBackbone):
    """输入 [...,T,O] → [...,T,H]；h/c 均为 [K,prod(...),H]。

    batch_first 不改变 h/c 顺序。单向、无 projection/dropout，防止未来信息泄漏，
    并使给定观测序列和初态的 old log-prob 可以被 PPO 确定性重算。
    """

    is_recurrent = True

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 1) -> None:
        super().__init__(input_dim, hidden_dim)
        if type(num_layers) is not int or num_layers < 1:
            raise ValueError("num_layers 必须为正整数")
        self.hidden_dim, self.num_layers = hidden_dim, num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(
        self, inputs: Tensor, hidden_state: RecurrentState = None, **kwargs: Tensor,
    ) -> BackboneOutput:
        self._validate_input(inputs)
        if inputs.ndim < 2 or inputs.shape[-2] < 1:
            raise ValueError("LSTM 输入需要非空的 [time,feature] 维")
        leading, time = inputs.shape[:-2], inputs.shape[-2]
        flat = inputs.reshape(-1, time, self.input_dim)
        validate_state(hidden_state, (self.num_layers, flat.shape[0], self.hidden_dim),
                       inputs, lstm=True)
        assert hidden_state is None or isinstance(hidden_state, tuple)
        features, following = self.lstm(flat, hidden_state)
        return BackboneOutput(features.reshape(*leading, time, self.output_dim), following)

    def initial_state(self, *batch_shape: int, device=None, dtype=None) -> tuple[Tensor, Tensor]:
        parameter = next(self.parameters())
        hidden = torch.zeros(self.num_layers, math.prod(batch_shape), self.hidden_dim,
                             device=device or parameter.device, dtype=dtype or parameter.dtype)
        return hidden, torch.zeros_like(hidden)


@dataclass(frozen=True, slots=True)
class LSTMBackboneConfig:
    kind: Literal["lstm"] = "lstm"
    hidden_dim: int = 128
    num_layers: int = 1

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 1 for v in (self.hidden_dim, self.num_layers)):
            raise ValueError("hidden_dim/num_layers 必须为正整数")

    def build(self, input_dim: int) -> LSTMBackbone:
        return LSTMBackbone(input_dim, self.hidden_dim, self.num_layers)
