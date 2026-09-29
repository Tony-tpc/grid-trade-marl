"""有位置编码、因果屏蔽的完整窗口 Transformer，固定 dropout=0。"""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from marl.core.recurrent import RecurrentState
from marl.models.base import BackboneOutput, BaseBackbone


class TransformerBackbone(BaseBackbone):
    """[*B,W,C] -> [*B,W,D]，时间位置不与批次或智能体维混合。"""

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(input_dim, model_dim)
        if num_heads < 1 or num_layers < 1 or model_dim % num_heads:
            raise ValueError("正整数 num_heads 必须整除 model_dim，num_layers 必须为正")
        if dropout != 0:
            raise ValueError("历史 Transformer 固定 dropout=0")
        self.input_projection = nn.Linear(input_dim, model_dim)
        layer = nn.TransformerEncoderLayer(
            model_dim, num_heads, model_dim * 4, 0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)

    def forward(
        self, inputs: Tensor, hidden_state: RecurrentState = None, **kwargs: Tensor
    ) -> BackboneOutput:
        """可选 padding_mask [*B,W]，True 为屏蔽；不接受外部循环状态。"""
        self._validate_input(inputs)
        if inputs.ndim < 2 or inputs.shape[-2] < 1:
            raise ValueError("Transformer 输入必须包含非空时间维")
        if hidden_state is not None:
            raise ValueError("无状态 backbone 不接受循环状态")
        length = inputs.shape[-2]
        features = self.input_projection(inputs).reshape(-1, length, self.output_dim)
        position = torch.arange(length, device=inputs.device, dtype=features.dtype)[:, None]
        frequency = torch.exp(
            torch.arange(0, self.output_dim, 2, device=inputs.device, dtype=features.dtype)
            * (-9.210340371976184 / self.output_dim)
        )
        angles = position * frequency
        encoding = torch.zeros(length, self.output_dim, device=inputs.device, dtype=features.dtype)
        encoding[:, 0::2] = angles.sin()
        encoding[:, 1::2] = angles[:, : self.output_dim // 2].cos()
        causal = torch.ones(length, length, device=inputs.device, dtype=torch.bool).triu(1)
        padding = kwargs.get("padding_mask")
        if padding is not None and padding.shape != inputs.shape[:-1]:
            raise ValueError("padding_mask 必须为 [*B,W]")
        output = self.encoder(
            features + encoding,
            mask=causal,
            src_key_padding_mask=None if padding is None else padding.reshape(-1, length).bool(),
        )
        return BackboneOutput(output.reshape(*inputs.shape[:-1], self.output_dim))


@dataclass(frozen=True, slots=True)
class TransformerBackboneConfig:
    kind: Literal["transformer"] = "transformer"
    model_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2

    def __post_init__(self) -> None:
        if any(
            type(x) is not int or x < 1 for x in (self.model_dim, self.num_heads, self.num_layers)
        ):
            raise ValueError("Transformer 尺寸必须为正整数")
        if self.model_dim % self.num_heads:
            raise ValueError("num_heads 必须整除 model_dim")

    def build(self, input_dim: int) -> TransformerBackbone:
        return TransformerBackbone(input_dim, self.model_dim, self.num_heads, self.num_layers)
