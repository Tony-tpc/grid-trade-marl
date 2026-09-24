from __future__ import annotations

from torch import Tensor, nn

from marl.models.base import BackboneOutput, BaseBackbone


class TransformerBackbone(BaseBackbone):
    """对智能体集合或时间序列建模的 Transformer encoder。"""

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(input_dim, model_dim)
        if model_dim % num_heads:
            raise ValueError("model_dim 必须能被 num_heads 整除")
        self.input_projection = nn.Linear(input_dim, model_dim)
        layer = nn.TransformerEncoderLayer(
            model_dim,
            num_heads,
            model_dim * 4,
            dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(
        self, inputs: Tensor, hidden_state: Tensor | None = None, **kwargs: Tensor
    ) -> BackboneOutput:
        self._validate_input(inputs)
        padding_mask = kwargs.get("padding_mask")
        features = self.encoder(
            self.input_projection(inputs),
            src_key_padding_mask=padding_mask.bool() if padding_mask is not None else None,
        )
        return BackboneOutput(features=features)
