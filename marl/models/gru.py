from __future__ import annotations

import torch
from torch import Tensor, nn

from marl.models.base import BackboneOutput, BaseBackbone


class GRUBackbone(BaseBackbone):
    """部分可观测任务的循环骨干，输入形状为 ``[..., T, input_dim]``。"""

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
