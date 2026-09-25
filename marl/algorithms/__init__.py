"""算法层：训练目标、target network 与参数更新。"""

from marl.algorithms.base import AlgorithmConfig, BaseMARLAlgorithm
from marl.algorithms.maac import MAAC, MAACConfig
from marl.algorithms.maddpg import MADDPG, MADDPGConfig
from marl.algorithms.mappo import MAPPO, MAPPOConfig, default_mappo_recipe
from marl.algorithms.masac import MASAC, MASACConfig
from marl.algorithms.qmix import QMIX, QMIXConfig

__all__ = [
    "AlgorithmConfig",
    "BaseMARLAlgorithm",
    "MAAC",
    "MAACConfig",
    "MADDPG",
    "MADDPGConfig",
    "MAPPO",
    "MAPPOConfig",
    "default_mappo_recipe",
    "MASAC",
    "MASACConfig",
    "QMIX",
    "QMIXConfig",
]
