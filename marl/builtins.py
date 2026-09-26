"""内置通用组件及其显式注册。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch

from marl.envs.base import ActionKind, EnvironmentSpec, RewardStructure
from marl.models.mlp import MLPBackbone
from marl.modules.critic import (
    AttentionCritic,
    CentralizedCritic,
    IndependentCentralizedCritics,
    TwinIndependentCentralizedCritics,
)
from marl.modules.mixer import QMixer
from marl.objectives import (
    CounterfactualPolicyObjective,
    DeterministicPolicyObjective,
    EntropyObjective,
    PPOClipObjective,
    SACEntropyObjective,
    TDLossObjective,
    ValueMSEObjective,
)
from marl.policies import (
    IndependentDeterministicPolicy,
    IndependentDiscretePolicy,
    IndependentGaussianPolicy,
    SharedDiscreteQPolicy,
)
from marl.registry import ComponentKind, ComponentRegistry
from marl.returns import GAEEstimator, TD0Estimator
from marl.target_updates import HardTargetUpdate, NoTargetUpdate, SoftTargetUpdate
from marl.training.off_policy import (
    OffPolicyOptimizerConfig,
    OffPolicyUpdateConfig,
    ReplayConfig,
)
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


def _optional_number(
    options: Mapping[str, object], name: str, default: float | None
) -> float | None:
    value = options.get(name, default)
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{name} 必须是数值或 None")
    return float(value)


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    horizon: int | None = None

    def __post_init__(self) -> None:
        if self.horizon is not None and self.horizon < 1:
            raise ValueError("rollout horizon 必须大于 0 或为 None")


@dataclass(frozen=True, slots=True)
class PPOOptimizerConfig:
    learning_rate: float = 3e-4
    update: PPOUpdateConfig = PPOUpdateConfig()

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate 必须大于 0")


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


def independent_deterministic_factory(
    options: Mapping[str, object], spec: EnvironmentSpec
) -> IndependentDeterministicPolicy:
    _strict_options(options, frozenset({"hidden_dim"}), "independent_deterministic")
    return IndependentDeterministicPolicy(
        spec.num_agents,
        spec.observation_dim,
        spec.action_dim,
        _integer(options, "hidden_dim", 128),
    )


def independent_gaussian_factory(
    options: Mapping[str, object], spec: EnvironmentSpec
) -> IndependentGaussianPolicy:
    _strict_options(options, frozenset({"hidden_dim"}), "independent_gaussian")
    return IndependentGaussianPolicy(
        spec.num_agents,
        spec.observation_dim,
        spec.action_dim,
        _integer(options, "hidden_dim", 128),
    )


def shared_discrete_q_factory(
    options: Mapping[str, object], spec: EnvironmentSpec
) -> SharedDiscreteQPolicy:
    _strict_options(options, frozenset({"hidden_dim"}), "shared_discrete_q")
    return SharedDiscreteQPolicy(
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


def attention_q_factory(
    options: Mapping[str, object], spec: EnvironmentSpec
) -> AttentionCritic:
    _strict_options(
        options, frozenset({"hidden_dim", "attention_heads"}), "attention_q"
    )
    return AttentionCritic(
        spec.num_agents,
        spec.observation_dim,
        spec.action_dim,
        _integer(options, "hidden_dim", 128),
        _integer(options, "attention_heads", 4),
    )


def independent_centralized_q_factory(
    options: Mapping[str, object], spec: EnvironmentSpec
) -> IndependentCentralizedCritics:
    _strict_options(options, frozenset({"hidden_dim"}), "independent_centralized_q")
    input_dim = spec.num_agents * (spec.observation_dim + spec.action_dim)
    return IndependentCentralizedCritics(
        spec.num_agents, input_dim, _integer(options, "hidden_dim", 128)
    )


def twin_independent_centralized_q_factory(
    options: Mapping[str, object], spec: EnvironmentSpec
) -> TwinIndependentCentralizedCritics:
    _strict_options(
        options, frozenset({"hidden_dim"}), "twin_independent_centralized_q"
    )
    input_dim = spec.num_agents * (spec.observation_dim + spec.action_dim)
    return TwinIndependentCentralizedCritics(
        spec.num_agents, input_dim, _integer(options, "hidden_dim", 128)
    )


def qmix_mixer_factory(
    options: Mapping[str, object], spec: EnvironmentSpec
) -> QMixer:
    _strict_options(options, frozenset({"mixing_dim"}), "qmix_mixer")
    return QMixer(
        spec.num_agents,
        spec.state_dim,
        _integer(options, "mixing_dim", 32),
    )


def ppo_clip_factory(options: Mapping[str, object], spec: EnvironmentSpec) -> PPOClipObjective:
    del spec
    _strict_options(options, frozenset({"clip_ratio"}), "ppo_clip")
    return PPOClipObjective(_number(options, "clip_ratio", 0.2))


def value_mse_factory(options: Mapping[str, object], spec: EnvironmentSpec) -> ValueMSEObjective:
    del spec
    _strict_options(options, frozenset({"coefficient"}), "value_mse")
    return ValueMSEObjective(_number(options, "coefficient", 0.5))


def td_mse_factory(options: Mapping[str, object], spec: EnvironmentSpec) -> TDLossObjective:
    del spec
    _strict_options(options, frozenset({"coefficient"}), "td_mse")
    return TDLossObjective(_number(options, "coefficient", 1.0))


def entropy_factory(options: Mapping[str, object], spec: EnvironmentSpec) -> EntropyObjective:
    del spec
    _strict_options(options, frozenset({"coefficient"}), "entropy")
    return EntropyObjective(_number(options, "coefficient", 0.01))


def counterfactual_factory(
    options: Mapping[str, object], spec: EnvironmentSpec
) -> CounterfactualPolicyObjective:
    del spec
    _strict_options(options, frozenset(), "counterfactual")
    return CounterfactualPolicyObjective()


def deterministic_policy_factory(
    options: Mapping[str, object], spec: EnvironmentSpec
) -> DeterministicPolicyObjective:
    del spec
    _strict_options(options, frozenset(), "deterministic_policy")
    return DeterministicPolicyObjective()


def sac_entropy_factory(
    options: Mapping[str, object], spec: EnvironmentSpec
) -> SACEntropyObjective:
    _strict_options(
        options, frozenset({"initial_alpha", "target_entropy"}), "sac_entropy"
    )
    return SACEntropyObjective(
        spec.num_agents,
        spec.action_dim,
        initial_alpha=_number(options, "initial_alpha", 0.2),
        target_entropy=_optional_number(options, "target_entropy", None),
    )


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


def td0_factory(options: Mapping[str, object], spec: EnvironmentSpec) -> TD0Estimator:
    del spec
    _strict_options(options, frozenset({"gamma"}), "td0")
    return TD0Estimator(gamma=_number(options, "gamma", 0.99))


def rollout_factory(options: Mapping[str, object], spec: EnvironmentSpec) -> RolloutConfig:
    del spec
    _strict_options(options, frozenset({"horizon"}), "rollout")
    horizon = options.get("horizon")
    if horizon is not None and (not isinstance(horizon, int) or isinstance(horizon, bool)):
        raise TypeError("rollout horizon 必须是整数或 None")
    return RolloutConfig(cast(int | None, horizon))


def replay_factory(options: Mapping[str, object], spec: EnvironmentSpec) -> ReplayConfig:
    del spec
    _strict_options(options, frozenset({"capacity", "batch_size"}), "replay")
    return ReplayConfig(
        capacity=_integer(options, "capacity", 100_000),
        batch_size=_integer(options, "batch_size", 256),
    )


def ppo_update_factory(options: Mapping[str, object], spec: EnvironmentSpec) -> PPOOptimizerConfig:
    del spec
    _strict_options(
        options,
        frozenset({"learning_rate", "epochs", "mini_batch_size", "max_grad_norm"}),
        "ppo_update",
    )
    max_grad_norm = _optional_number(options, "max_grad_norm", 10.0)
    return PPOOptimizerConfig(
        learning_rate=_number(options, "learning_rate", 3e-4),
        update=PPOUpdateConfig(
            epochs=_integer(options, "epochs", 4),
            mini_batch_size=_integer(options, "mini_batch_size", 256),
            max_grad_norm=max_grad_norm,
        ),
    )


def off_policy_update_factory(
    options: Mapping[str, object], spec: EnvironmentSpec
) -> OffPolicyOptimizerConfig:
    del spec
    _strict_options(
        options,
        frozenset({"learning_rate", "max_grad_norm", "amp_dtype"}),
        "off_policy_update",
    )
    amp_name = options.get("amp_dtype")
    if amp_name not in (None, "bf16"):
        raise ValueError("amp_dtype 必须为 null 或 'bf16'")
    return OffPolicyOptimizerConfig(
        learning_rate=_number(options, "learning_rate", 3e-4),
        update=OffPolicyUpdateConfig(
            max_grad_norm=_optional_number(options, "max_grad_norm", 10.0),
            amp_dtype=torch.bfloat16 if amp_name == "bf16" else None,
        ),
    )


def no_target_factory(options: Mapping[str, object], spec: EnvironmentSpec) -> NoTargetUpdate:
    del spec
    _strict_options(options, frozenset(), "none target update")
    return NoTargetUpdate()


def soft_target_factory(
    options: Mapping[str, object], spec: EnvironmentSpec
) -> SoftTargetUpdate:
    del spec
    _strict_options(options, frozenset({"tau"}), "soft target update")
    return SoftTargetUpdate(_number(options, "tau", 0.005))


def hard_target_factory(
    options: Mapping[str, object], spec: EnvironmentSpec
) -> HardTargetUpdate:
    del spec
    _strict_options(options, frozenset({"interval"}), "hard target update")
    return HardTargetUpdate(_integer(options, "interval", 1))


def register_builtin_components(registry: ComponentRegistry) -> None:
    """幂等注册内置组件，方便测试使用独立 registry。"""

    discrete = frozenset({ActionKind.DISCRETE})
    continuous = frozenset({ActionKind.CONTINUOUS})
    shared = frozenset({RewardStructure.SHARED})
    registrations = (
        (
            ComponentKind.POLICY,
            "independent_discrete",
            independent_discrete_factory,
            discrete,
            None,
        ),
        (
            ComponentKind.POLICY,
            "independent_deterministic",
            independent_deterministic_factory,
            continuous,
            None,
        ),
        (
            ComponentKind.POLICY,
            "independent_gaussian",
            independent_gaussian_factory,
            continuous,
            None,
        ),
        (ComponentKind.POLICY, "shared_discrete_q", shared_discrete_q_factory, discrete, shared),
        (ComponentKind.CRITIC, "centralized_value", centralized_value_factory, discrete, None),
        (ComponentKind.CRITIC, "attention_q", attention_q_factory, discrete, None),
        (
            ComponentKind.CRITIC,
            "independent_centralized_q",
            independent_centralized_q_factory,
            continuous,
            None,
        ),
        (
            ComponentKind.CRITIC,
            "twin_independent_centralized_q",
            twin_independent_centralized_q_factory,
            continuous,
            None,
        ),
        (ComponentKind.CRITIC, "qmix_mixer", qmix_mixer_factory, discrete, shared),
        (ComponentKind.OBJECTIVE, "ppo_clip", ppo_clip_factory, None, None),
        (ComponentKind.OBJECTIVE, "value_mse", value_mse_factory, None, None),
        (ComponentKind.OBJECTIVE, "td_mse", td_mse_factory, None, None),
        (ComponentKind.OBJECTIVE, "entropy", entropy_factory, None, None),
        (ComponentKind.OBJECTIVE, "counterfactual", counterfactual_factory, discrete, None),
        (
            ComponentKind.OBJECTIVE,
            "deterministic_policy",
            deterministic_policy_factory,
            continuous,
            None,
        ),
        (ComponentKind.OBJECTIVE, "sac_entropy", sac_entropy_factory, continuous, None),
        (ComponentKind.RETURN_ESTIMATOR, "gae", gae_factory, None, None),
        (ComponentKind.RETURN_ESTIMATOR, "td0", td0_factory, None, None),
        (ComponentKind.EXPERIENCE_SOURCE, "rollout", rollout_factory, None, None),
        (ComponentKind.EXPERIENCE_SOURCE, "replay", replay_factory, None, None),
        (ComponentKind.UPDATE_PLAN, "ppo_update", ppo_update_factory, None, None),
        (
            ComponentKind.UPDATE_PLAN,
            "off_policy_update",
            off_policy_update_factory,
            None,
            None,
        ),
        (ComponentKind.TARGET_UPDATE, "none", no_target_factory, None, None),
        (ComponentKind.TARGET_UPDATE, "soft", soft_target_factory, None, None),
        (ComponentKind.TARGET_UPDATE, "hard", hard_target_factory, None, None),
    )
    for kind, name, factory, action_kinds, reward_structures in registrations:
        if name in registry.names(kind):
            continue
        registry.register(
            kind,
            name,
            factory,
            action_kinds=action_kinds,
            reward_structures=reward_structures,
        )
