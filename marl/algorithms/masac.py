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
        """检查 TD 损失系数为非负，初始熵温度 initial_alpha 必须为正。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            None；非法配置在构造阶段抛 ValueError。
        """
        TDLossObjective(self.td_coefficient)
        if self.initial_alpha <= 0:
            raise ValueError("initial_alpha 必须大于 0")


@dataclass(frozen=True, slots=True)
class MASACConfig:
    """MASAC 配置；环境尺寸由 spec 注入。"""

    schema_version: Literal[2] = 2
    algorithm: Literal["masac"] = "masac"
    policy: Buildable[PolicyTopology] = IndependentGaussianConfig()
    critic: Buildable[TwinQEnsemble] = TwinQConfig()
    loss: MASACLossConfig = MASACLossConfig()
    value_target: TD0Config = TD0Config()
    replay: ReplayConfig = ReplayConfig()
    update: TemperatureActorCriticUpdateConfig = TemperatureActorCriticUpdateConfig()
    target_update: SoftTargetConfig | HardTargetConfig = SoftTargetConfig()


    def validate(self, spec: EnvironmentSpec) -> None:
        """验证配置版本和算法标识，并要求环境使用连续动作。

        Args:
            spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。

        Returns:
            None；不兼容时抛 ValueError。
        """
        if self.schema_version != 2 or self.algorithm != "masac":
            raise ValueError("MASAC config 的 algorithm/schema_version 不匹配")
        if spec.action_kind != ActionKind.CONTINUOUS:
            raise ValueError("MASAC 不支持当前动作类型")

    def build(self, spec: EnvironmentSpec) -> MASAC:
        """以当前配置和环境尺寸直接构造 MASAC。

        Args:
            spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。

        Returns:
            新算法实例；optimizer/trainer 由 build_experiment 另外装配。
        """
        return MASAC(spec, self)


class MASAC(BaseMARLAlgorithm):
    """逐智能体双 Q 与可学习温度的连续动作算法。

    critic 拟合各自的 soft TD target；actor 通过冻结 critic 对自己的动作求导；
    temperature 只优化 log_alpha。三种 loss 共用公式但独立构图/更新，避免
    诊断用的总 loss 被误当作单 optimizer 训练入口。
    """

    def __init__(self, spec: EnvironmentSpec, config: MASACConfig) -> None:
        """校验配置并装配 MASAC 的在线网络、数学目标及所需 target/统计模块。

        Args:
            spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。
            config: 当前算法的冻结配置，提供网络、数学目标和更新超参数。

        Returns:
            None；可训练参数和 buffers 注册到当前 nn.Module。
        """
        super().__init__(spec)
        config.validate(spec)
        self.config = config
        self.policy = config.policy.build(spec)
        if bool(getattr(self.policy, "is_recurrent", False)):
            raise ValueError("离策略 policy 不支持跨环境步循环 backbone；请使用 history encoder")
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
        """将各智能体观测映射为联合动作。随机模式使用 tanh-Gaussian
            采样，确定性模式使用策略确定性输出。

        Args:
            observations: 浮点观测 [...,N,O]；循环单步为 [B,N,O]，序列为 [B,T,N,O]。
            deterministic: 是否采用确定性动作；具体算法的例外见本函数说明。
            action_mask: 可选 bool [...,N,A]，True 表示合法；连续动作必须为 None。
            kwargs: 保留的接口扩展关键字；当前具体算法 act 不读取这些值。

        Returns:
            动作 Tensor：离散 [...,N]，连续 [...,N,A]；此接口不返回 log-prob/value。
        """
        return self.policy.act(
            observations,
            deterministic=deterministic,
            action_mask=action_mask,
        ).actions

    def compute_loss_bundle(self, batch: MARLBatch) -> LossBundle:
        """汇总各子目标用于数学诊断；只构图，不执行 backward、step 或 target 统计更新。

        Args:
            batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。

        Returns:
            LossBundle；total 是标量总目标，terms 是各项日志；正常训练仍应调用 update。
        """
        critic = self.compute_critic_loss_bundle(batch)
        actor_result, temperature_result = self._actor_and_temperature_results(batch)
        actor = LossBundle.combine((actor_result,))
        temperature = LossBundle.combine((temperature_result,))
        terms = {**critic.terms, **actor.terms, **temperature.terms}
        terms["loss"] = critic.total + actor_result.loss + temperature_result.loss
        return LossBundle(terms["loss"], terms)

    def _validate_training_batch(self, batch: MARLBatch) -> None:
        """校验公共 batch 形状以及 MASAC 所需的 actions/rewards/next_observations。

        Args:
            batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。

        Returns:
            None；字段缺失或形状不符抛 ValueError。
        """
        batch.validate(self.spec.num_agents, self.spec.observation_dim)
        if batch.actions is None or batch.rewards is None or batch.next_observations is None:
            raise ValueError("MASAC 训练需要 actions/rewards/next_observations")

    def compute_critic_loss_bundle(
        self, batch: MARLBatch, *, update_statistics: bool = False
    ) -> LossBundle:
        """使用 replay 联合动作拟合逐智能体 TD
            target；下一值来自当前策略、target twin-Q 最小值及熵温度。

        使用 replay 动作更新 twin centralized critics。

        Args:
            batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。
            update_statistics: 是否更新逐智能体 target 尺度统计；诊断调用默认 False。

        Returns:
            critic LossBundle；target 无梯度，只有 terminated 阻止 bootstrap；可选更新尺度统计。
        """

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
        """冻结 critic 参数并隔离其他 actor 动作，构建每个智能体自己的策略目标。

        固定 twin critics 与其他 actor 动作，更新当前策略。

        Args:
            batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。

        Returns:
            actor LossBundle；只构图，不执行 optimizer.step。
        """

        actor_result, _ = self._actor_and_temperature_results(batch)
        return LossBundle.combine((actor_result,))

    def _actor_and_temperature_results(
        self, batch: MARLBatch
    ) -> tuple[ObjectiveResult, ObjectiveResult]:
        """共享一次当前策略采样，计算隔离智能体动作梯度的 actor 目标与独立温度目标。

        Args:
            batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。

        Returns:
            (actor_result, temperature_result)；两者为 ObjectiveResult，不执行 optimizer。
        """
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
        """无梯度重新采样当前策略，仅构建 log_alpha 的温度调节目标。

        只更新 log alpha；策略 log-prob 在目标内部显式 detach。

        Args:
            batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。

        Returns:
            温度 LossBundle；梯度只流向熵温度参数。
        """

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
        """对一批 replay 经验按 critic → actor → temperature →
            target 顺序更新参数；复用基类 optimize。

        按 twin critic → actor → temperature → target 的 MASAC 顺序更新。

        Args:
            batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。
            runtime: 已装配的命名 optimizer、裁剪上限、目标更新器、随机生成器及计数状态。

        Returns:
            命名 float 指标；runtime.sync_metrics=False 时返回 {}，之后用 flush_metrics 获取。
        """

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
        """声明目标网络与在线网络的配对，供 runtime.finish 统一同步。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            有序 (target, online) 对；基类默认空，不在本函数执行同步。
        """
        return ((self.get_submodule("target_critics"), self.get_submodule("critics")),)
