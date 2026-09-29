"""静态智能体图上的消息聚合，不处理动作或动态通信图。"""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from marl.core.recurrent import RecurrentState
from marl.models.base import BackboneOutput, BaseBackbone


class GNNBackbone(BaseBackbone):
    """输入 [*B,N,F]，逐层拼接自身特征与行归一化邻居均值，输出 [*B,N,H]。"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2) -> None:
        super().__init__(input_dim, hidden_dim)
        if num_layers < 1:
            raise ValueError("num_layers 必须为正")
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU())
            for _ in range(num_layers)
        )

    def forward(
        self, inputs: Tensor, hidden_state: RecurrentState = None, **kwargs: Tensor
    ) -> BackboneOutput:
        """adjacency 为固定 [N,N]；添加单位自环 A+I 后按行归一化，不修改输入。"""
        self._validate_input(inputs)
        if inputs.ndim < 2 or inputs.shape[-2] < 1:
            raise ValueError("GNN 需要非空节点维")
        if hidden_state is not None:
            raise ValueError("GNN 不接受循环状态")
        n = inputs.shape[-2]
        adjacency = kwargs.get("adjacency")
        if adjacency is None:
            adjacency = torch.ones(n, n, device=inputs.device, dtype=inputs.dtype)
            adjacency = adjacency - torch.eye(n, device=inputs.device, dtype=inputs.dtype)
        if (
            adjacency.shape != (n, n)
            or not torch.isfinite(adjacency).all()
            or (adjacency < 0).any()
        ):
            raise ValueError("adjacency 必须是有限非负的固定 [N,N] 矩阵")
        adjacency = adjacency.to(inputs) + torch.eye(n, device=inputs.device, dtype=inputs.dtype)
        weights = adjacency / adjacency.sum(-1, keepdim=True)
        features = self.input_projection(inputs)
        for layer in self.layers:
            features = layer(torch.cat((features, weights @ features), dim=-1))
        return BackboneOutput(features)


@dataclass(frozen=True, slots=True)
class GNNBackboneConfig:
    kind: Literal["gnn"] = "gnn"
    hidden_dim: int = 128
    num_layers: int = 2

    def __post_init__(self) -> None:
        if any(type(x) is not int or x < 1 for x in (self.hidden_dim, self.num_layers)):
            raise ValueError("GNN 尺寸必须为正整数")

    def build(self, input_dim: int) -> GNNBackbone:
        return GNNBackbone(input_dim, self.hidden_dim, self.num_layers)
