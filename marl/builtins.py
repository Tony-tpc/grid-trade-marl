"""内置通用组件及其显式注册。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from marl.envs.base import ActionKind, EnvironmentSpec
from marl.models.mlp import MLPBackbone
from marl.modules.critic import CentralizedCritic
from marl.objectives import EntropyObjective, PPOClipObjective, ValueMSEObjective
from marl.policies import IndependentDiscretePolicy
from marl.registry import ComponentKind, ComponentRegistry
from marl.returns import GAEEstimator
from marl.training.on_policy import PPOUpdateConfig


def _strict_options(
    options: Mapping[str, object], allowed: frozenset[str], component: str
) -> None:
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"{component} 含未知 options：{sorted(unknown)}")


def _number(options: Mapping[str, object], name: str, default: float) -> float:
    value = options.get(name, default)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{name} 必须是数值")
    return float(value)


def _integer(options: Mapping[str, object], name: str, default: int) -> int:
    value = options.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} 必须是整数")
    return value


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    horizon: int | None = None

    def __post_init__(self) -> None:
        if self.horizon is not None and self.horizon < 1:
            raise ValueError("rollout horizon 必须大于 0 或为 None")


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    learning_rate: float = 3e-4
    update: PPOUpdateConfig = PPOUpdateConfig()

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate 必须大于 0")


@dataclass(frozen=True, slots=True)
class NoTargetUpdate:
    def step(self, pairs: object = ()) -> None:
        del pairs


def independent_discrete_factory(
    options: Mapping[str, object], spec: EnvironmentSpec
) -> IndependentDiscretePolicy:
    _strict_options(options, frozenset({"hidden_dim"}), "independent_discrete")
    return IndependentDiscretePolicy(
        spec.num_agents,
        spec.observation_dim,
        spec.action_dim,
        _integer(options, "hidden_dim", 128),
    )


def centralized_value_factory(
    options: Mapping[str, object], spec: EnvironmentSpec
) -> CentralizedCritic:
    _strict_options(options, frozenset({"hidden_dim"}), "centralized_value")
    hidden_dim = _integer(options, "hidden_dim", 128)
    return CentralizedCritic(
        MLPBackbone(spec.state_dim, output_dim=hidden_dim),
        output_dim=spec.num_agents,
    )


def ppo_clip_factory(options: Mapping[str, object], spec: EnvironmentSpec) -> PPOClipObjective:
    del spec
    _strict_options(options, frozenset({"clip_ratio"}), "ppo_clip")
    return PPOClipObjective(_number(options, "clip_ratio", 0.2))


def value_mse_factory(options: Mapping[str, object], spec: EnvironmentSpec) -> ValueMSEObjective:
    del spec
    _strict_options(options, frozenset({"coefficient"}), "value_mse")
    return ValueMSEObjective(_number(options, "coefficient", 0.5))


def entropy_factory(options: Mapping[str, object], spec: EnvironmentSpec) -> EntropyObjective:
    del spec
    _strict_options(options, frozenset({"coefficient"}), "entropy")
    return EntropyObjective(_number(options, "coefficient", 0.01))


def gae_factory(options: Mapping[str, object], spec: EnvironmentSpec) -> GAEEstimator:
    del spec
    _strict_options(
        options,
        frozenset({"gamma", "lambda", "normalize_advantage"}),
        "gae",
    )
    normalize = options.get("normalize_advantage", True)
    if not isinstance(normalize, bool):
        raise TypeError("normalize_advantage 必须是 bool")
    return GAEEstimator(
        gamma=_number(options, "gamma", 0.99),
        gae_lambda=_number(options, "lambda", 0.95),
        normalize=normalize,
    )


def rollout_factory(options: Mapping[str, object], spec: EnvironmentSpec) -> RolloutConfig:
    del spec
    _strict_options(options, frozenset({"horizon"}), "rollout")
    horizon = options.get("horizon")
    if horizon is not None and (not isinstance(horizon, int) or isinstance(horizon, bool)):
        raise TypeError("rollout horizon 必须是整数或 None")
    return RolloutConfig(cast(int | None, horizon))


def ppo_update_factory(options: Mapping[str, object], spec: EnvironmentSpec) -> OptimizerConfig:
    del spec
    _strict_options(
        options,
        frozenset({"learning_rate", "epochs", "mini_batch_size", "max_grad_norm"}),
        "ppo_update",
    )
    max_grad_norm = options.get("max_grad_norm", 10.0)
    if max_grad_norm is not None and (
        not isinstance(max_grad_norm, int | float) or isinstance(max_grad_norm, bool)
    ):
        raise TypeError("max_grad_norm 必须是数值或 None")
    return OptimizerConfig(
        learning_rate=_number(options, "learning_rate", 3e-4),
        update=PPOUpdateConfig(
            epochs=_integer(options, "epochs", 4),
            mini_batch_size=_integer(options, "mini_batch_size", 256),
            max_grad_norm=(float(max_grad_norm) if max_grad_norm is not None else None),
        ),
    )


def no_target_factory(options: Mapping[str, object], spec: EnvironmentSpec) -> NoTargetUpdate:
    del spec
    _strict_options(options, frozenset(), "none target update")
    return NoTargetUpdate()


def register_builtin_components(registry: ComponentRegistry) -> None:
    """幂等注册内置组件，方便测试使用独立 registry。"""

    registrations = (
        (ComponentKind.POLICY, "independent_discrete", independent_discrete_factory),
        (ComponentKind.CRITIC, "centralized_value", centralized_value_factory),
        (ComponentKind.OBJECTIVE, "ppo_clip", ppo_clip_factory),
        (ComponentKind.OBJECTIVE, "value_mse", value_mse_factory),
        (ComponentKind.OBJECTIVE, "entropy", entropy_factory),
        (ComponentKind.RETURN_ESTIMATOR, "gae", gae_factory),
        (ComponentKind.EXPERIENCE_SOURCE, "rollout", rollout_factory),
        (ComponentKind.UPDATE_PLAN, "ppo_update", ppo_update_factory),
        (ComponentKind.TARGET_UPDATE, "none", no_target_factory),
    )
    for kind, name, factory in registrations:
        if name in registry.names(kind):
            continue
        registry.register(
            kind,
            name,
            factory,
            action_kinds=(
                frozenset({ActionKind.DISCRETE})
                if kind == ComponentKind.POLICY and name == "independent_discrete"
                else None
            ),
        )
