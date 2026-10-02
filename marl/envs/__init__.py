"""环境与算法之间的稳定接口。新论文的环境实现放在本目录。"""

from __future__ import annotations

from marl.core.layout import HistoryLayout, StateLayout
from marl.envs.base import (
    ActionKind,
    EnvironmentAdapter,
    EnvironmentSpec,
    EnvironmentStep,
    RewardStructure,
    Transition,
    transitions_to_batch,
)
from marl.envs.config import (
    EnvironmentConfig,
    build_energy_trading_adapter,
    build_mpe2_simple_adversary_adapter,
    energy_trading_config_from_environment,
    environment_config_from_dict,
    load_environment_config,
    mpe2_simple_adversary_config_from_environment,
)
from marl.envs.demand_response import (
    DemandResponseConfig,
    DemandResponseEnv,
    FixedLeaderDemandAdapter,
)
from marl.envs.demand_response_data import DemandProfiles
from marl.envs.energy_trading import EnergyProfiles, EnergyTradingConfig, EnergyTradingEnv
from marl.envs.energy_trading_adapter import EnergyTradingAdapter
from marl.envs.memory_cue import MemoryCueAdapter
from marl.envs.mpe2_simple_adversary_adapter import (
    MPE2SimpleAdversaryAdapter,
    MPE2SimpleAdversaryConfig,
    build_mpe2_simple_adversary,
)

__all__ = [
    "DemandProfiles",
    "DemandResponseConfig",
    "DemandResponseEnv",
    "FixedLeaderDemandAdapter",
    "HistoryLayout",
    "StateLayout",
    "MemoryCueAdapter",
    "ActionKind",
    "EnvironmentAdapter",
    "EnvironmentSpec",
    "EnvironmentStep",
    "RewardStructure",
    "Transition",
    "transitions_to_batch",
    "EnergyProfiles",
    "EnergyTradingConfig",
    "EnergyTradingEnv",
    "EnergyTradingAdapter",
    "EnvironmentConfig",
    "environment_config_from_dict",
    "load_environment_config",
    "energy_trading_config_from_environment",
    "build_energy_trading_adapter",
    "mpe2_simple_adversary_config_from_environment",
    "build_mpe2_simple_adversary_adapter",
    "MPE2SimpleAdversaryConfig",
    "MPE2SimpleAdversaryAdapter",
    "build_mpe2_simple_adversary",
]
