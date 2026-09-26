"""组合式一般和博弈多智能体 SAC。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from marl.algorithms.base import BaseMARLAlgorithm
from marl.core import MARLBatch
from marl.envs.base import ActionKind, EnvironmentSpec
from marl.extensions import Buildable
from marl.modules.critic import TwinQConfig, TwinQEnsemble
from marl.modules.policy import IndependentGaussianConfig, PolicyTopology
from marl.objectives import (
    LossBundle,
    ObjectiveResult,
    SACEntropyObjective,
    TDLossObjective,
)
from marl.returns import TD0Config, ValueTargetEstimator
from marl.target_updates import HardTargetConfig, SoftTargetConfig
from marl.training.gradients import frozen_parameters
from marl.training.off_policy import OffPolicyUpdateConfig, ReplayConfig


@dataclass(frozen=True, slots=True)
class MASACLossConfig:
    td_coefficient: float = 1.0
    initial_alpha: float = 0.2
    target_entropy: float | None = None

    def __post_init__(self) -> None:
        TDLossObjective(self.td_coefficient)
        if self.initial_alpha <= 0:
            raise ValueError("initial_alpha 必须大于 0")


@dataclass(frozen=True, slots=True)
class MASACConfig:
    """MASAC 配置；环境尺寸由 spec 注入。"""

    schema_version: Literal[1] = 1
    algorithm: Literal["masac"] = "masac"
    policy: Buildable[PolicyTopology] = IndependentGaussianConfig()
    critic: Buildable[TwinQEnsemble] = TwinQConfig()
    loss: MASACLossConfig = MASACLossConfig()
    value_target: TD0Config = TD0Config()
    replay: ReplayConfig = ReplayConfig()
    update: OffPolicyUpdateConfig = OffPolicyUpdateConfig()
    target_update: SoftTargetConfig | HardTargetConfig = SoftTargetConfig()


    def validate(self, spec: EnvironmentSpec) -> None:
        if self.schema_version != 1 or self.algorithm != "masac":
            raise ValueError("MASAC config 的 algorithm/schema_version 不匹配")
        if spec.action_kind != ActionKind.CONTINUOUS:
            raise ValueError("MASAC 不支持当前动作类型")

    def build(self, spec: EnvironmentSpec) -> MASAC:
        return MASAC(spec, self)


class MASAC(BaseMARLAlgorithm):
    def __init__(self, spec: EnvironmentSpec, config: MASACConfig) -> None:
        super().__init__(spec)
        config.validate(spec)
        self.config = config
        self.policy = config.policy.build(spec)
        self.critics = config.critic.build(spec)
        self.target_critics = deepcopy(self.critics)
        self.get_submodule("target_critics").requires_grad_(False)
        self.td_loss = TDLossObjective(config.loss.td_coefficient)
        self.entropy_objective = SACEntropyObjective(
            spec.num_agents, spec.action_dim,
            initial_alpha=config.loss.initial_alpha, target_entropy=config.loss.target_entropy,
        )
        self.return_estimator: ValueTargetEstimator = config.value_target.build()

    @staticmethod
    def _critic_input(observations: Tensor, actions: Tensor) -> Tensor:
        return torch.cat((observations.flatten(-2), actions.flatten(-2)), dim=-1)

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
            raise ValueError("MASAC 训练需要 actions/rewards/next_observations")
        terminated, _ = batch.terminal_flags()
        q1, q2 = self.critics(
            self._critic_input(batch.observations, batch.actions)
        )
        with torch.no_grad():
            next_output = self.policy.act(batch.next_observations)
            assert next_output.log_prob is not None
            next_q1, next_q2 = self.target_critics(
                self._critic_input(batch.next_observations, next_output.actions)
            )
            soft_next = (
                torch.minimum(next_q1, next_q2)
                - self.entropy_objective.alpha * next_output.log_prob
            )
            target_q = self.return_estimator.estimate(
                batch.rewards, soft_next, terminated
            )
        first_result = self.td_loss(q1, target_q)
        second_result = self.td_loss(q2, target_q)
        critic_loss = first_result.loss + second_result.loss
        critic_result = ObjectiveResult(
            critic_loss,
            {"critic_loss": critic_loss.detach()},
        )

        policy_output = self.policy.act(batch.observations)
        assert policy_output.log_prob is not None
        q_terms = []
        for index in range(self.spec.num_agents):
            joint_actions = torch.stack(
                [
                    policy_output.actions[..., other, :]
                    if other == index
                    else policy_output.actions[..., other, :].detach()
                    for other in range(self.spec.num_agents)
                ],
                dim=-2,
            )
            policy_input = self._critic_input(batch.observations, joint_actions)
            first = self.critics.first.critics[index]
            second = self.critics.second.critics[index]
            with frozen_parameters(first), frozen_parameters(second):
                q_terms.append(
                    torch.minimum(first(policy_input), second(policy_input)).squeeze(-1)
                )
        min_q = torch.stack(q_terms, dim=-1)
        actor_result = self.entropy_objective.actor(
            policy_output.log_prob, min_q
        )
        alpha_result = self.entropy_objective.temperature(
            policy_output.log_prob
        )
        return LossBundle.combine((critic_result, actor_result, alpha_result))

    def target_pairs(self) -> tuple[tuple[nn.Module, nn.Module], ...]:
        return ((self.get_submodule("target_critics"), self.get_submodule("critics")),)
