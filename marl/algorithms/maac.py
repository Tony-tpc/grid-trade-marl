"""组合式离散动作 MAAC，保留逐智能体收益。"""

from __future__ import annotations

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
    ObjectiveResult,
    TDLossObjective,
)
from marl.returns import TD0Config, ValueTargetEstimator
from marl.target_updates import HardTargetConfig, SoftTargetConfig, frozen_target
from marl.training.off_policy import (
    ActorCriticUpdateConfig,
    ReplayConfig,
)
from marl.training.optimization import OptimizerRuntime, metrics_to_float


@dataclass(frozen=True, slots=True)
class MAACUpdateConfig(ActorCriticUpdateConfig):
    """MAAC 的 critic→逐智能体 policy 更新参数。"""


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
    update: MAACUpdateConfig = MAACUpdateConfig()
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
        self.target_policy = frozen_target(self.policy)
        self.target_critic = frozen_target(self.critic)
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
        critic = self.compute_critic_loss_bundle(batch)
        actors = [
            self.compute_actor_loss_bundle(batch, index)
            for index in range(self.spec.num_agents)
        ]
        actor_loss = torch.stack([bundle.total for bundle in actors]).mean()
        actor_metric = torch.stack(
            [bundle.terms["actor_loss"] for bundle in actors]
        ).mean()
        entropy = torch.stack([bundle.terms["entropy"] for bundle in actors]).mean()
        return LossBundle.combine(
            (
                ObjectiveResult(critic.total, {
                    name: value for name, value in critic.terms.items() if name != "loss"
                }),
                ObjectiveResult(actor_loss, {
                    "actor_loss": actor_metric,
                    "entropy": entropy,
                }),
            )
        )

    def _validate_training_batch(self, batch: MARLBatch) -> None:
        batch.validate(self.spec.num_agents, self.spec.observation_dim)
        if batch.actions is None or batch.rewards is None or batch.next_observations is None:
            raise ValueError("MAAC 训练需要 actions/rewards/next_observations")

    def compute_critic_loss_bundle(self, batch: MARLBatch) -> LossBundle:
        """使用 replay 联合动作回归逐智能体 TD target。"""

        self._validate_training_batch(batch)
        assert batch.actions is not None and batch.rewards is not None
        assert batch.next_observations is not None
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
        return LossBundle.combine((critic_result,))

    def compute_actor_loss_bundle(
        self, batch: MARLBatch, agent_index: int
    ) -> LossBundle:
        """只构建一个智能体的 policy loss；其他 actor 不接收梯度。"""

        self._validate_training_batch(batch)
        if not 0 <= agent_index < self.spec.num_agents:
            raise IndexError("agent_index 超出智能体范围")
        assert batch.actions is not None
        with torch.no_grad():
            q_all = self.critic(batch.observations, batch.actions)
        logits = self.policy.logits(batch.observations, batch.action_mask)
        selected_logits = logits[..., agent_index, :]
        selected_q = q_all[..., agent_index, :]
        actor_result = self.policy_objective(selected_logits, selected_q)
        entropy = Categorical(
            logits=selected_logits, validate_args=False
        ).entropy()
        return LossBundle.combine(
            (actor_result, self.entropy_objective(entropy))
        )

    def update(
        self, batch: MARLBatch, runtime: OptimizerRuntime
    ) -> dict[str, float]:
        """按 critic → 每个智能体 policy → target 的 MAAC 顺序更新。"""

        device = batch.observations.device
        with runtime.autocast(device):
            critic = self.compute_critic_loss_bundle(batch)
        critic_norm = self.optimize(
            runtime.optimizer("critic"),
            critic,
            runtime.max_grad_norm("critic"),
        )
        runtime.record_optimizer_step()
        actor_totals: dict[str, Tensor] = {}
        actor_norms: list[Tensor] = []
        for agent_index in range(self.spec.num_agents):
            with runtime.autocast(device):
                actor = self.compute_actor_loss_bundle(batch, agent_index)
            actor_norms.append(
                self.optimize(
                    runtime.optimizer("actor"),
                    actor,
                    runtime.max_grad_norm("actor"),
                    drop_zero_gradients=True,
                )
            )
            runtime.record_optimizer_step()
            for name, value in actor.terms.items():
                actor_totals[name] = (
                    actor_totals.get(name, torch.zeros_like(value)) + value.detach()
                )
        runtime.finish(self.target_pairs())
        metrics = dict(critic.terms)
        metrics.update(
            {name: value / self.spec.num_agents for name, value in actor_totals.items()}
        )
        metrics["loss"] = critic.total.detach() + actor_totals["loss"] / self.spec.num_agents
        metrics["critic_gradient_norm"] = critic_norm
        metrics["actor_gradient_norm"] = torch.stack(actor_norms).mean()
        metrics["gradient_norm"] = torch.maximum(
            metrics["critic_gradient_norm"], metrics["actor_gradient_norm"]
        )
        return metrics_to_float(metrics)

    def target_pairs(self) -> tuple[tuple[nn.Module, nn.Module], ...]:
        return (
            (self.get_submodule("target_policy"), self.get_submodule("policy")),
            (self.get_submodule("target_critic"), self.get_submodule("critic")),
        )
