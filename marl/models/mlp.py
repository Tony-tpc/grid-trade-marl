from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from marl.models.base import BackboneOutput, BaseBackbone


class MLPBackbone(BaseBackbone):
    """输入 [...,input_dim]，输出 [...,output_dim]，保留所有样本/智能体维。

    hidden_dims 控制中间层，output_dim 控制最后的特征层；每层均接可选 LayerNorm
    和 ReLU。内置 policy/critic Config.hidden_dim 传给 output_dim，并不会把默认
    中间层 (128,128) 一起改小；要修改完整层宽，可直接构造本类或外部组件。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        output_dim: int = 128,
        layer_norm: bool = False,
    ) -> None:
        super().__init__(input_dim, output_dim)
        if not hidden_dims or any(dim <= 0 for dim in hidden_dims):
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
        self, inputs: Tensor, hidden_state: Tensor | None = None, **kwargs: Tensor
    ) -> BackboneOutput:
        self._validate_input(inputs)
        return BackboneOutput(features=self.network(inputs))
