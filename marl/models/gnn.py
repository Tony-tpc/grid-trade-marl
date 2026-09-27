from __future__ import annotations

import torch
from torch import Tensor, nn

from marl.core.recurrent import RecurrentState
from marl.models.base import BackboneOutput, BaseBackbone


class GNNBackbone(BaseBackbone):
    """零额外依赖的消息传递网络。

    输入 ``[..., N, input_dim]``；关键字参数 adjacency 为 ``[..., N, N]``。
    图连接属于输入数据，不在此处假定通信规则。加入自环后按行归一化，用 PyTorch
    matmul 聚合邻居，再拼接自身特征；这不是独立的多智能体训练算法。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2) -> None:
        super().__init__(input_dim, hidden_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU())
            for _ in range(num_layers)
        )

    def forward(
        self, inputs: Tensor, hidden_state: RecurrentState = None, **kwargs: Tensor
    ) -> BackboneOutput:
        self._validate_input(inputs)
        if hidden_state is not None:
            raise ValueError("无状态 backbone 不接受循环状态")
        adjacency = kwargs.get("adjacency")
        if adjacency is None:
            raise ValueError("GNNBackbone.forward 需要 adjacency=[..., N, N]")
        if adjacency.shape[-2:] != (inputs.shape[-2], inputs.shape[-2]):
            raise ValueError("adjacency 的节点维度必须与 agent 维度一致")
        identity = torch.eye(inputs.shape[-2], device=inputs.device, dtype=inputs.dtype)
        normalized = adjacency.to(inputs.dtype) + identity
        normalized = normalized / normalized.sum(dim=-1, keepdim=True).clamp_min(1.0)
        features = self.input_projection(inputs)
        for layer in self.layers:
            messages = torch.matmul(normalized, features)
            features = layer(torch.cat((features, messages), dim=-1))
        return BackboneOutput(features=features)
