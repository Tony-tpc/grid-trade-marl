"""算法和配置并列导出；新增算法在这里暴露公共入口。"""
from marl.algorithms.base import BaseMARLAlgorithm
from marl.algorithms.maac import MAAC, MAACConfig
from marl.algorithms.maddpg import MADDPG, MADDPGConfig
from marl.algorithms.mappo import MAPPO, MAPPOConfig
from marl.algorithms.masac import MASAC, MASACConfig
from marl.algorithms.qmix import QMIX, QMIXConfig

__all__ = [
    "BaseMARLAlgorithm",
    "MAPPO", "MAPPOConfig",
    "MAAC", "MAACConfig",
    "MADDPG", "MADDPGConfig",
    "MASAC", "MASACConfig",
    "QMIX", "QMIXConfig",
]
