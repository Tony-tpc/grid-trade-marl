from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from torch import Tensor, nn

from marl.core.recurrent import RecurrentState
from marl.models.base import BackboneOutput, BaseBackbone


class MLPBackbone(BaseBackbone):
    """输入 [...,input_dim]，输出 [...,output_dim]，保留所有样本/智能体维。

    hidden_dims 控制中间层，output_dim 控制最后的特征层；每层均接可选 LayerNorm
    和 ReLU。MLPBackboneConfig 分别控制中间层及输出层，不隐式改变默认层宽。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        output_dim: int = 128,
        layer_norm: bool = False,
    ) -> None:
        super().__init__(input_dim, output_dim)
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError("hidden_dims 必须包含正整数")
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in (*hidden_dims, output_dim):
            layers.append(nn.Linear(previous, hidden))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU())
            previous = hidden
        self.network = nn.Sequential(*layers)

    def forward(
        self, inputs: Tensor, hidden_state: RecurrentState = None, **kwargs: Tensor
    ) -> BackboneOutput:
        self._validate_input(inputs)
        if hidden_state is not None:
            raise ValueError("MLP 不接受循环状态")
        return BackboneOutput(features=self.network(inputs))


@dataclass(frozen=True, slots=True)
class MLPBackboneConfig:
    kind: Literal["mlp"] = "mlp"
    hidden_dims: tuple[int, ...] = (128, 128)
    output_dim: int = 128
    layer_norm: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.hidden_dims, tuple) or any(
            type(dim) is not int or dim < 1 for dim in self.hidden_dims
        ):
            raise ValueError("hidden_dims 必须是正整数 tuple；允许空 tuple")
        if type(self.output_dim) is not int or self.output_dim < 1:
            raise ValueError("output_dim 必须是正整数")

    def build(self, input_dim: int) -> MLPBackbone:
        return MLPBackbone(input_dim, self.hidden_dims, self.output_dim, self.layer_norm)
