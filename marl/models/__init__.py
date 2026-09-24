"""可替换的特征提取骨干网络。"""

from marl.models.base import BackboneOutput, BaseBackbone
from marl.models.gnn import GNNBackbone
from marl.models.gru import GRUBackbone
from marl.models.mlp import MLPBackbone
from marl.models.transformer import TransformerBackbone

__all__ = [
    "BackboneOutput",
    "BaseBackbone",
    "GNNBackbone",
    "GRUBackbone",
    "MLPBackbone",
    "TransformerBackbone",
]
