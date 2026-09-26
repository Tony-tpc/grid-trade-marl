"""具体算法薄装配类及其默认 recipe。"""

from marl.algorithms.base import BaseMARLAlgorithm
from marl.algorithms.maac import MAAC, default_maac_recipe
from marl.algorithms.maddpg import MADDPG, default_maddpg_recipe
from marl.algorithms.mappo import MAPPO, default_mappo_recipe
from marl.algorithms.masac import MASAC, default_masac_recipe
from marl.algorithms.qmix import QMIX, default_qmix_recipe

__all__ = [
    "BaseMARLAlgorithm",
    "MAAC",
    "MADDPG",
    "MAPPO",
    "MASAC",
    "QMIX",
    "default_maac_recipe",
    "default_maddpg_recipe",
    "default_mappo_recipe",
    "default_masac_recipe",
    "default_qmix_recipe",
]
