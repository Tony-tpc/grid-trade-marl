"""组合式离散动作 MAAC，保留逐智能体收益。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from marl.algorithms.base import BaseMARLAlgorithm
from marl.core import MARLBatch
from marl.envs.base import ActionKind, EnvironmentSpec
from marl.extensions import Buildable
from marl.modules.critic import AttentionQConfig, AttentionQNetwork
from marl.modules.policy import DiscretePolicy, IndependentDiscreteConfig
from marl.objectives import (
    CounterfactualPolicyObjective,
    EntropyObjective,
    LossBundle,
    TDLossObjective,
)
from marl.returns import TD0Config, ValueTargetEstimator
from marl.target_updates import HardTargetConfig, SoftTargetConfig
from marl.training.off_policy import OffPolicyUpdateConfig, ReplayConfig


@dataclass(frozen=True, slots=True)
class MAACLossConfig:
    td_coefficient: float = 1.0
    entropy_coefficient: float = 0.01

    def __post_init__(self) -> None:
        TDLossObjective(self.td_coefficient)
        EntropyObjective(self.entropy_coefficient)


@dataclass(frozen=True, slots=True)
class MAACConfig:
    """MAAC 配置；环境尺寸由 spec 注入。"""

    schema_version: Literal[1] = 1
    algorithm: Literal["maac"] = "maac"
    policy: Buildable[DiscretePolicy] = IndependentDiscreteConfig()
    critic: Buildable[AttentionQNetwork] = AttentionQConfig()
    loss: MAACLossConfig = MAACLossConfig()
    value_target: TD0Config = TD0Config()
    replay: ReplayConfig = ReplayConfig()
    update: OffPolicyUpdateConfig = OffPolicyUpdateConfig()
    target_update: SoftTargetConfig | HardTargetConfig = SoftTargetConfig()


    def validate(self, spec: EnvironmentSpec) -> None:
        if self.schema_version != 1 or self.algorithm != "maac":
            raise ValueError("MAAC config 的 algorithm/schema_version 不匹配")
        if spec.action_kind != ActionKind.DISCRETE:
            raise ValueError("MAAC 不支持当前动作类型")

    def build(self, spec: EnvironmentSpec) -> MAAC:
        return MAAC(spec, self)


class MAAC(BaseMARLAlgorithm):
    def __init__(self, spec: EnvironmentSpec, config: MAACConfig) -> None:
        super().__init__(spec)
        config.validate(spec)
        self.config = config
        self.policy = config.policy.build(spec)
        self.critic = config.critic.build(spec)
        self.target_policy = deepcopy(self.policy)
        self.get_submodule("target_policy").requires_grad_(False)
        self.target_critic = deepcopy(self.critic)
        self.get_submodule("target_critic").requires_grad_(False)
        self.td_loss = TDLossObjective(config.loss.td_coefficient)
        self.policy_objective = CounterfactualPolicyObjective()
        self.entropy_objective = EntropyObjective(config.loss.entropy_coefficient)
        self.return_estimator: ValueTargetEstimator = config.value_target.build()

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
        if batch.actions is None or batch.rewards is None or batch.next_observations is None:
            raise ValueError("MAAC 训练需要 actions/rewards/next_observations")
        terminated, _ = batch.terminal_flags()
        q_all = self.critic(batch.observations, batch.actions)
        chosen_q = q_all.gather(
            -1, batch.actions.long().unsqueeze(-1)
        ).squeeze(-1)
        with torch.no_grad():
            next_output = self.target_policy.act(
                batch.next_observations,
                action_mask=batch.next_action_mask,
            )
            assert next_output.log_prob is not None
            next_q_all = self.target_critic(
                batch.next_observations, next_output.actions
            )
            next_q = next_q_all.gather(
                -1, next_output.actions.unsqueeze(-1)
            ).squeeze(-1)
            soft_next_q = (
                next_q
                - self.entropy_objective.coefficient * next_output.log_prob
            )
            target_q = self.return_estimator.estimate(
                batch.rewards, soft_next_q, terminated
            )
        critic_result = self.td_loss(chosen_q, target_q)

        logits = self.policy.logits(batch.observations, batch.action_mask)
        actor_result = self.policy_objective(logits, q_all)
        entropy = Categorical(logits=logits, validate_args=False).entropy()
        return LossBundle.combine(
            (critic_result, actor_result, self.entropy_objective(entropy))
        )

    def target_pairs(self) -> tuple[tuple[nn.Module, nn.Module], ...]:
        return (
            (self.get_submodule("target_policy"), self.get_submodule("policy")),
            (self.get_submodule("target_critic"), self.get_submodule("critic")),
        )
