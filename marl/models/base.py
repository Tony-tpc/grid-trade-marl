from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from marl.core.recurrent import RecurrentState


@dataclass(slots=True)
class BackboneOutput:
    """骨干网络的统一输出；循环状态和调试信息均为可选。"""

    features: Tensor
    hidden_state: RecurrentState = None
    extras: dict[str, Tensor] | None = None


class BaseBackbone(nn.Module, ABC):
    """MLP/GRU/LSTM/GNN/Transformer 的共同接口。

    Backbone 只做表示学习，不输出动作、Q 值或 loss，因此算法层可以独立替换网络。
    这里统一的是特征输出，不保证不同 backbone 的输入布局可直接互换：GRU/LSTM 需要
    时间维，GNN 需要图结构，Transformer 的序列语义也必须由使用者显式指定。
    """

    is_recurrent = False

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim 和 output_dim 必须为正整数")
        self.input_dim, self.output_dim = input_dim, output_dim

    def _validate_input(self, inputs: Tensor) -> None:
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(f"期望最后一维为 {self.input_dim}，实际为 {inputs.shape[-1]}")

    @abstractmethod
    def forward(  # type: ignore[override]
        self, inputs: Tensor, hidden_state: RecurrentState = None, **kwargs: Tensor
    ) -> BackboneOutput:
        """把输入映射到 ``output_dim`` 维特征。"""

    def initial_state(
        self, *batch_shape: int, device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> RecurrentState:
        """无状态网络无需 hidden；循环网络默认继承参数的设备和 dtype。"""
        return None
