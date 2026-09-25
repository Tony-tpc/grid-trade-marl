"""只依据环境规格创建算法配置，不读取环境内部实现。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast, overload

from marl.algorithms import (
    MAACConfig,
    MADDPGConfig,
    MAPPOConfig,
    MASACConfig,
    QMIXConfig,
)
from marl.algorithms.base import AlgorithmConfig
from marl.envs.base import ActionKind, EnvironmentSpec, RewardStructure


@overload
def algorithm_config_from_env(
    algorithm: Literal["maac"], spec: EnvironmentSpec, **overrides: object
) -> MAACConfig: ...


@overload
def algorithm_config_from_env(
    algorithm: Literal["maddpg"], spec: EnvironmentSpec, **overrides: object
) -> MADDPGConfig: ...


@overload
def algorithm_config_from_env(
    algorithm: Literal["mappo"], spec: EnvironmentSpec, **overrides: object
) -> MAPPOConfig: ...


@overload
def algorithm_config_from_env(
    algorithm: Literal["masac"], spec: EnvironmentSpec, **overrides: object
) -> MASACConfig: ...


@overload
def algorithm_config_from_env(
    algorithm: Literal["qmix"], spec: EnvironmentSpec, **overrides: object
) -> QMIXConfig: ...


@overload
def algorithm_config_from_env(
    algorithm: str, spec: EnvironmentSpec, **overrides: object
) -> AlgorithmConfig: ...


def algorithm_config_from_env(
    algorithm: str, spec: EnvironmentSpec, **overrides: object
) -> AlgorithmConfig:
    """从适配器规格生成模型配置，并尽早检查动作类型是否兼容。

    ``overrides`` 可设置 gamma、hidden_dim 等超参数，但环境尺寸始终由 spec 决定。
    """

    name = algorithm.lower()
    registry: dict[str, tuple[type[AlgorithmConfig], ActionKind]] = {
        "maac": (MAACConfig, ActionKind.DISCRETE),
        "maddpg": (MADDPGConfig, ActionKind.CONTINUOUS),
        "mappo": (MAPPOConfig, ActionKind.DISCRETE),
        "masac": (MASACConfig, ActionKind.CONTINUOUS),
        "qmix": (QMIXConfig, ActionKind.DISCRETE),
    }
    if name not in registry:
        raise ValueError(f"未知算法 {algorithm!r}；可选：{', '.join(registry)}")
    config_type, expected_kind = registry[name]
    if spec.action_kind != expected_kind:
        raise ValueError(
            f"{name} 需要 {expected_kind.value} 动作，环境提供 {spec.action_kind.value}"
        )
    if name == "qmix" and spec.reward_structure != RewardStructure.SHARED:
        raise ValueError("QMIX 要求所有智能体共享同一个团队奖励；个体收益博弈请选 MAAC/MAPPO")

    derived: dict[str, object] = {
        "num_agents": spec.num_agents,
        "observation_dim": spec.observation_dim,
        "action_dim": spec.action_dim,
    }
    if name in ("mappo", "qmix"):
        derived["state_dim"] = spec.state_dim
    conflicting = set(overrides) & set(derived)
    if conflicting:
        raise ValueError(f"环境尺寸由 spec 决定，不能覆盖：{sorted(conflicting)}")
    constructor = cast(Callable[..., AlgorithmConfig], config_type)
    return constructor(**derived, **overrides)
