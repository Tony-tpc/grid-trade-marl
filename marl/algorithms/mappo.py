"""由通用策略、价值函数、目标和训练组件组装的 MAPPO。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import Tensor

from marl.algorithms.base import BaseMARLAlgorithm
from marl.core import MARLBatch, MARLModelOutput
from marl.envs.base import ActionKind, EnvironmentSpec
from marl.extensions import Buildable
from marl.modules.critic import CentralizedValueConfig, ValueNetwork
from marl.modules.policy import DiscretePolicy, IndependentDiscreteConfig
from marl.objectives import (
    EntropyObjective,
    LossBundle,
    PPOClipObjective,
    ValueMSEObjective,
)
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


    def validate(self, spec: EnvironmentSpec) -> None:
        if self.schema_version != 1 or self.algorithm != "mappo":
            raise ValueError("MAPPO config 的 algorithm/schema_version 不匹配")
        if spec.action_kind != ActionKind.DISCRETE:
            raise ValueError("MAPPO 不支持当前动作类型")

    def build(self, spec: EnvironmentSpec) -> MAPPO:
        return MAPPO(spec, self)


class MAPPO(BaseMARLAlgorithm):
    """只负责 MAPPO 前向语义的薄装配类。"""

    def __init__(self, spec: EnvironmentSpec, config: MAPPOConfig) -> None:
        super().__init__(spec)
        config.validate(spec)
        self.config = config
        self.policy = config.policy.build(spec)
        self.critic = config.critic.build(spec)
        self.policy_objective = PPOClipObjective(config.loss.clip_ratio)
        self.value_objective = ValueMSEObjective(config.loss.value_coefficient)
        self.entropy_objective = EntropyObjective(config.loss.entropy_coefficient)

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
