"""可替换的特征提取骨干网络。"""

from marl.models.base import BackboneOutput, BaseBackbone
from marl.models.encoder import (
    EncoderConfig,
    HistoryEncoder,
    HistoryEncoderConfig,
    IdentityEncoderConfig,
    InputEncoder,
)
from marl.models.gnn import GNNBackbone, GNNBackboneConfig
from marl.models.gru import GRUBackbone, GRUBackboneConfig
from marl.models.lstm import LSTMBackbone, LSTMBackboneConfig
from marl.models.mlp import MLPBackbone, MLPBackboneConfig
from marl.models.transformer import TransformerBackbone, TransformerBackboneConfig

BackboneConfig = MLPBackboneConfig | GRUBackboneConfig | LSTMBackboneConfig

__all__ = [
    "EncoderConfig",
    "HistoryEncoder",
    "HistoryEncoderConfig",
    "IdentityEncoderConfig",
    "InputEncoder",
    "GNNBackboneConfig",
    "TransformerBackboneConfig",
    "BackboneOutput",
    "BaseBackbone",
    "GNNBackbone",
    "GRUBackbone",
    "GRUBackboneConfig",
    "LSTMBackbone",
    "LSTMBackboneConfig",
    "MLPBackboneConfig",
    "BackboneConfig",
    "MLPBackbone",
    "TransformerBackbone",
]
