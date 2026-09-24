from __future__ import annotations

from typing import cast

from torch import Tensor, nn

from marl.models.base import BaseBackbone


class ValueCritic(nn.Module):
    """对每个输入位置输出标量 V/Q，适用于独立 critic。"""

    def __init__(self, backbone: BaseBackbone) -> None:
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Linear(backbone.output_dim, 1)

    def forward(self, inputs: Tensor, **backbone_kwargs: Tensor) -> Tensor:
        features = self.backbone.forward(inputs, **backbone_kwargs).features
        return cast(Tensor, self.value_head(features).squeeze(-1))


class CentralizedCritic(nn.Module):
    """CTDE 的集中式 critic。

    调用者负责把 global state、联合观测或联合动作拼成最后一维；这种显式约定避免
    critic 猜测各算法不同的数据布局。
    """

    def __init__(self, backbone: BaseBackbone, output_dim: int = 1) -> None:
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Linear(backbone.output_dim, output_dim)  # 输出Q值

    def forward(self, centralized_input: Tensor, **backbone_kwargs: Tensor) -> Tensor:
        """
        centralized_input :
            batch{
                [o1,o2,o3,...,a1,a2,a3,...],
                [o1,o2,o3,...,a1,a2,a3,...]
                ...
            }

        centralized_input -> backbone -> features -> value_head -> Q_value
        """
        features = self.backbone.forward(centralized_input, **backbone_kwargs).features
        return cast(Tensor, self.value_head(features))
