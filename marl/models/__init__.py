"""可替换的特征提取骨干网络。"""

from marl.models.base import BackboneOutput, BaseBackbone
from marl.models.gnn import GNNBackbone
from marl.models.gru import GRUBackbone, GRUBackboneConfig
from marl.models.lstm import LSTMBackbone, LSTMBackboneConfig
from marl.models.mlp import MLPBackbone, MLPBackboneConfig
from marl.models.transformer import TransformerBackbone

BackboneConfig = MLPBackboneConfig | GRUBackboneConfig | LSTMBackboneConfig

__all__ = [
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
