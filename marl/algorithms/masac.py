"""组合式一般和博弈多智能体 SAC。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from marl.algorithms.base import BaseMARLAlgorithm
from marl.core import MARLBatch
from marl.envs.base import ActionKind, EnvironmentSpec
from marl.extensions import Buildable
from marl.modules.critic import TwinQConfig, TwinQEnsemble, centralized_critic_input
from marl.modules.policy import IndependentGaussianConfig, PolicyTopology
from marl.objectives import (
    LossBundle,
    ObjectiveResult,
    SACEntropyObjective,
    TDLossObjective,
)
from marl.returns import TD0Config, ValueTargetEstimator
from marl.target_updates import HardTargetConfig, SoftTargetConfig, frozen_target
from marl.training.gradients import frozen_parameters, isolate_agent_action
from marl.training.off_policy import (
    ReplayConfig,
    TemperatureActorCriticUpdateConfig,
)
from marl.training.optimization import OptimizerRuntime
from marl.value_scaling import TargetScale, agent_statistics, value_diagnostics


@dataclass(frozen=True, slots=True)
class MASACLossConfig:
    td_coefficient: float = 1.0
    normalize_targets: bool = False
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
    update: TemperatureActorCriticUpdateConfig = TemperatureActorCriticUpdateConfig()
    target_update: SoftTargetConfig | HardTargetConfig = SoftTargetConfig()


    def validate(self, spec: EnvironmentSpec) -> None:
        if self.schema_version != 1 or self.algorithm != "masac":
            raise ValueError("MASAC config 的 algorithm/schema_version 不匹配")
        if spec.action_kind != ActionKind.CONTINUOUS:
            raise ValueError("MASAC 不支持当前动作类型")

    def build(self, spec: EnvironmentSpec) -> MASAC:
        return MASAC(spec, self)


class MASAC(BaseMARLAlgorithm):
    """逐智能体双 Q 与可学习温度的连续动作算法。

    critic 拟合各自的 soft TD target；actor 通过冻结 critic 对自己的动作求导；
    temperature 只优化 log_alpha。三种 loss 共用公式但独立构图/更新，避免
    诊断用的总 loss 被误当作单 optimizer 训练入口。
    """

    def __init__(self, spec: EnvironmentSpec, config: MASACConfig) -> None:
        super().__init__(spec)
        config.validate(spec)
        self.config = config
        self.policy = config.policy.build(spec)
        self.critics = config.critic.build(spec)
        self.target_critics = frozen_target(self.critics)
        self.td_loss = TDLossObjective(config.loss.td_coefficient)
        self.target_scale = TargetScale(spec.num_agents, config.loss.normalize_targets)
        self.entropy_objective = SACEntropyObjective(
            spec.num_agents, spec.action_dim,
            initial_alpha=config.loss.initial_alpha, target_entropy=config.loss.target_entropy,
        )
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
        actor_result, temperature_result = self._actor_and_temperature_results(batch)
        actor = LossBundle.combine((actor_result,))
        temperature = LossBundle.combine((temperature_result,))
        terms = {**critic.terms, **actor.terms, **temperature.terms}
        terms["loss"] = critic.total + actor_result.loss + temperature_result.loss
        return LossBundle(terms["loss"], terms)

    def _validate_training_batch(self, batch: MARLBatch) -> None:
        batch.validate(self.spec.num_agents, self.spec.observation_dim)
        if batch.actions is None or batch.rewards is None or batch.next_observations is None:
            raise ValueError("MASAC 训练需要 actions/rewards/next_observations")

    def compute_critic_loss_bundle(
        self, batch: MARLBatch, *, update_statistics: bool = False
    ) -> LossBundle:
        """使用 replay 动作更新 twin centralized critics。"""

        self._validate_training_batch(batch)
        assert batch.actions is not None and batch.rewards is not None
        assert batch.next_observations is not None
        terminated, _ = batch.terminal_flags()
        q1, q2 = self.critics(
            centralized_critic_input(batch.observations, batch.actions)
        )
        with torch.no_grad():
            next_output = self.policy.act(batch.next_observations)
            assert next_output.log_prob is not None
            next_q1, next_q2 = self.target_critics(
                centralized_critic_input(batch.next_observations, next_output.actions)
            )
            soft_next = (
                torch.minimum(next_q1, next_q2)
                - self.entropy_objective.alpha * next_output.log_prob
            )
            target_q = self.return_estimator.estimate(
                batch.rewards, soft_next, terminated
            )
        if update_statistics:
            self.target_scale.update(target_q)
        scaled_target = self.target_scale(target_q)
        first_result = self.td_loss(self.target_scale(q1), scaled_target)
        second_result = self.td_loss(self.target_scale(q2), scaled_target)
        critic_loss = first_result.loss + second_result.loss
        critic_result = ObjectiveResult(
            critic_loss,
            {"critic_loss": critic_loss.detach()},
        )
        bundle = LossBundle.combine((critic_result,))
        return LossBundle(bundle.total, {**bundle.terms,
            **value_diagnostics(q1, target_q), **agent_statistics("q2", q2),
            **agent_statistics("reward", batch.rewards)})

    def compute_actor_loss_bundle(self, batch: MARLBatch) -> LossBundle:
        """固定 twin critics 与其他 actor 动作，更新当前策略。"""

        actor_result, _ = self._actor_and_temperature_results(batch)
        return LossBundle.combine((actor_result,))

    def _actor_and_temperature_results(
        self, batch: MARLBatch
    ) -> tuple[ObjectiveResult, ObjectiveResult]:
        self._validate_training_batch(batch)
        policy_output = self.policy.act(batch.observations)
        assert policy_output.log_prob is not None
        q_terms = []
        for index in range(self.spec.num_agents):
            joint_actions = isolate_agent_action(policy_output.actions, index)
            policy_input = centralized_critic_input(batch.observations, joint_actions)
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
        temperature_result = self.entropy_objective.temperature(policy_output.log_prob)
        actor_result = ObjectiveResult(actor_result.loss, {**actor_result.metrics,
            **agent_statistics("entropy", -policy_output.log_prob),
            **agent_statistics("action_saturation",
                (policy_output.actions.abs() >= 0.99).float().mean(-1))})
        temperature_result = ObjectiveResult(temperature_result.loss, {
            **temperature_result.metrics,
            **agent_statistics("alpha", self.entropy_objective.alpha)})
        return actor_result, temperature_result

    def compute_temperature_loss_bundle(self, batch: MARLBatch) -> LossBundle:
        """只更新 log alpha；策略 log-prob 在目标内部显式 detach。"""

        self._validate_training_batch(batch)
        # 必须在 actor 更新后重新采样；温度目标不需要任何 critic 前向。
        with torch.no_grad():
            output = self.policy.act(batch.observations)
        assert output.log_prob is not None
        temperature_result = self.entropy_objective.temperature(output.log_prob)
        temperature_result = ObjectiveResult(temperature_result.loss, {
            **temperature_result.metrics,
            **agent_statistics("alpha", self.entropy_objective.alpha)})
        return LossBundle.combine((temperature_result,))

    def update(
        self, batch: MARLBatch, runtime: OptimizerRuntime
    ) -> dict[str, float]:
        """按 twin critic → actor → temperature → target 的 MASAC 顺序更新。"""

        device = batch.observations.device
        with runtime.autocast(device):
            critic = self.compute_critic_loss_bundle(batch, update_statistics=True)
        critic_norm = self.optimize(
            runtime.optimizer("critic"),
            critic,
            runtime.max_grad_norm("critic"),
            parameters=runtime.parameters("critic"),
        )
        runtime.record_optimizer_step("critic", critic_norm)
        with runtime.autocast(device):
            actor = self.compute_actor_loss_bundle(batch)
        actor_norm = self.optimize(
            runtime.optimizer("actor"),
            actor,
            runtime.max_grad_norm("actor"),
            parameters=runtime.parameters("actor"),
        )
        runtime.record_optimizer_step("actor", actor_norm)
        with runtime.autocast(device):
            temperature = self.compute_temperature_loss_bundle(batch)
        temperature_norm = self.optimize(
            runtime.optimizer("temperature"),
            temperature,
            runtime.max_grad_norm("temperature"),
            parameters=runtime.parameters("temperature"),
        )
        runtime.record_optimizer_step("temperature", temperature_norm)
        runtime.finish(self.target_pairs())
        metrics = {**critic.terms, **actor.terms, **temperature.terms}
        metrics["loss"] = (
            critic.total.detach() + actor.total.detach() + temperature.total.detach()
        )
        metrics["critic_gradient_norm"] = critic_norm
        metrics["actor_gradient_norm"] = actor_norm
        metrics["temperature_gradient_norm"] = temperature_norm
        metrics["gradient_norm"] = torch.stack(
            (critic_norm, actor_norm, temperature_norm)
        ).max()
        return runtime.export_metrics(metrics)

    def target_pairs(self) -> tuple[tuple[nn.Module, nn.Module], ...]:
        return ((self.get_submodule("target_critics"), self.get_submodule("critics")),)
