"""由通用策略、价值函数、目标和训练组件组装的 MAPPO。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from torch import Tensor, nn

from marl.algorithms.assembly import assemble_algorithm_components, require_one
from marl.algorithms.base import BaseMARLAlgorithm
from marl.core import MARLBatch, MARLModelOutput
from marl.envs.base import ActionKind, EnvironmentSpec
from marl.extensions import Buildable
from marl.modules.critic import CentralizedValueConfig, ValueNetwork
from marl.modules.policy import DiscretePolicy, IndependentDiscreteConfig, IndependentDiscretePolicy
from marl.objectives import (
    EntropyObjective,
    LossBundle,
    PPOClipObjective,
    ValueMSEObjective,
)
from marl.recipes import AlgorithmRecipe, CompiledRecipe, ComponentRecipe
from marl.registry import DEFAULT_COMPONENT_REGISTRY, ComponentRegistry
from marl.returns import GAEConfig
from marl.training.on_policy import PPOUpdateConfig, RolloutConfig


@dataclass(frozen=True, slots=True)
class MAPPOLossConfig:
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01

    def __post_init__(self) -> None:
        PPOClipObjective(self.clip_ratio)
        ValueMSEObjective(self.value_coefficient)
        EntropyObjective(self.entropy_coefficient)


@dataclass(frozen=True, slots=True)
class MAPPOConfig:
    """MAPPO 配置；环境尺寸由 spec 注入。"""

    schema_version: Literal[1] = 1
    algorithm: Literal["mappo"] = "mappo"
    policy: Buildable[DiscretePolicy] = IndependentDiscreteConfig()
    critic: Buildable[ValueNetwork] = CentralizedValueConfig()
    loss: MAPPOLossConfig = MAPPOLossConfig()
    advantage: GAEConfig = GAEConfig()
    rollout: RolloutConfig = RolloutConfig()
    update: PPOUpdateConfig = PPOUpdateConfig()


    def build(self, spec: EnvironmentSpec) -> MAPPO:
        """在此处直接查看 MAPPO 的完整组件组合。"""
        if spec.action_kind != ActionKind.DISCRETE:
            raise ValueError("MAPPO 不支持当前动作类型")
        return MAPPO(
            spec, self.policy.build(spec), self.critic.build(spec),
            PPOClipObjective(self.loss.clip_ratio),
            ValueMSEObjective(self.loss.value_coefficient),
            EntropyObjective(self.loss.entropy_coefficient),
        )


def default_mappo_recipe() -> AlgorithmRecipe:
    return AlgorithmRecipe(
        schema_version=1,
        algorithm="mappo",
        policy=ComponentRecipe("independent_discrete", {"hidden_dim": 128}),
        critic=ComponentRecipe("centralized_value", {"hidden_dim": 128}),
        objectives=(
            ComponentRecipe("ppo_clip", {"clip_ratio": 0.2}),
            ComponentRecipe("value_mse", {"coefficient": 0.5}),
            ComponentRecipe("entropy", {"coefficient": 0.01}),
        ),
        returns=ComponentRecipe(
            "gae",
            {"gamma": 0.99, "lambda": 0.95, "normalize_advantage": True},
        ),
        experience=ComponentRecipe("rollout", {}),
        update=ComponentRecipe(
            "ppo_update",
            {
                "learning_rate": 3e-4,
                "epochs": 4,
                "mini_batch_size": 256,
                "max_grad_norm": 10.0,
            },
        ),
        target_update=ComponentRecipe("none", {}),
    )


class MAPPO(BaseMARLAlgorithm):
    """只负责 MAPPO 前向语义的薄装配类。"""

    def __init__(
        self,
        spec: EnvironmentSpec,
        policy: DiscretePolicy,
        critic: ValueNetwork,
        policy_objective: PPOClipObjective,
        value_objective: ValueMSEObjective,
        entropy_objective: EntropyObjective,
        compiled_recipe: CompiledRecipe | None = None,
    ) -> None:
        super().__init__(spec)
        self.policy = policy
        self.critic = critic
        self.policy_objective = policy_objective
        self.value_objective = value_objective
        self.entropy_objective = entropy_objective
        self.compiled_recipe = compiled_recipe

    @classmethod
    def from_recipe(
        cls,
        spec: EnvironmentSpec,
        recipe: AlgorithmRecipe,
        *,
        registry: ComponentRegistry = DEFAULT_COMPONENT_REGISTRY,
    ) -> MAPPO:
        parts = assemble_algorithm_components("mappo", spec, recipe, registry)
        if not isinstance(parts.policy, IndependentDiscretePolicy):
            raise TypeError("MAPPO policy 必须实现 IndependentDiscretePolicy")
        if not isinstance(parts.critic, nn.Module):
            raise TypeError("MAPPO critic 必须是 nn.Module")
        return cls(
            spec,
            parts.policy,
            parts.critic,
            cast(PPOClipObjective, require_one(parts.objectives, PPOClipObjective, "MAPPO")),
            cast(ValueMSEObjective, require_one(parts.objectives, ValueMSEObjective, "MAPPO")),
            cast(EntropyObjective, require_one(parts.objectives, EntropyObjective, "MAPPO")),
            parts.compiled,
        )

    def _critic_input(self, batch: MARLBatch) -> Tensor:
        return batch.state if batch.state is not None else batch.observations.flatten(-2)

    def values(self, observations: Tensor, state: Tensor | None = None) -> Tensor:
        critic_input = state if state is not None else observations.flatten(-2)
        return self.critic(critic_input)

    def sample(
        self,
        observations: Tensor,
        state: Tensor | None = None,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
    ) -> MARLModelOutput:
        output = self.policy.act(
            observations,
            deterministic=deterministic,
            action_mask=action_mask,
        )
        output.values = self.values(observations, state)
        return output

    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
        **kwargs: Tensor,
    ) -> Tensor:
        return self.policy.act(
            observations,
            deterministic=deterministic,
            action_mask=action_mask,
        ).actions

    def compute_loss_bundle(self, batch: MARLBatch) -> LossBundle:
        batch.validate(self.spec.num_agents, self.spec.observation_dim)
        if batch.actions is None:
            raise ValueError("MAPPO 训练需要 actions")
        required = ("old_log_prob", "advantages", "returns")
        if any(name not in batch.extras for name in required):
            raise ValueError(f"MAPPO batch.extras 必须包含 {required}")
        expected_shape = batch.observations.shape[:-1]
        for name in required:
            if batch.extras[name].shape != expected_shape:
                raise ValueError(f"MAPPO {name} 必须具有逐智能体形状 {expected_shape}")
        output = self.policy.evaluate(
            batch.observations,
            batch.actions,
            action_mask=batch.action_mask,
        )
        assert output.log_prob is not None and output.entropy is not None
        values = self.critic(self._critic_input(batch))
        return LossBundle.combine(
            (
                self.policy_objective(
                    output.log_prob,
                    batch.extras["old_log_prob"],
                    batch.extras["advantages"],
                ),
                self.value_objective(values, batch.extras["returns"]),
                self.entropy_objective(output.entropy),
            )
        )
