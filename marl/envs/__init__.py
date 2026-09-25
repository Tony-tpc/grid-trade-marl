"""环境与算法之间的稳定接口。新论文的环境实现放在本目录。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from marl.envs.base import (
    ActionKind,
    EnvironmentAdapter,
    EnvironmentSpec,
    EnvironmentStep,
    RewardStructure,
    Transition,
    transitions_to_batch,
)
from marl.envs.energy_trading import EnergyProfiles, EnergyTradingConfig, EnergyTradingEnv
from marl.envs.energy_trading_adapter import EnergyTradingAdapter

if TYPE_CHECKING:
    from marl.envs.config import algorithm_config_from_env
else:

    def algorithm_config_from_env(*args: Any, **kwargs: Any) -> Any:
        """惰性加载配置工厂，避免环境基础类型反向触发算法包导入。"""

        from marl.envs.config import algorithm_config_from_env as factory

        return factory(*args, **kwargs)

__all__ = [
    "ActionKind",
    "EnvironmentAdapter",
    "EnvironmentSpec",
    "EnvironmentStep",
    "RewardStructure",
    "Transition",
    "transitions_to_batch",
    "algorithm_config_from_env",
    "EnergyProfiles",
    "EnergyTradingConfig",
    "EnergyTradingEnv",
    "EnergyTradingAdapter",
]
