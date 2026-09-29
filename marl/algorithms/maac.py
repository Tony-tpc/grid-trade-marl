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
from marl.modules.policy import (
    AgentLogitsPolicy,
    DiscretePolicy,
    IndependentDiscreteConfig,
    IndependentDiscretePolicy,
)
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
from marl.training.optimization import OptimizerRuntime
from marl.value_scaling import TargetScale, agent_statistics, value_diagnostics


@dataclass(frozen=True, slots=True)
class MAACLossConfig:
    td_coefficient: float = 1.0
    normalize_targets: bool = False
    entropy_coefficient: float = 0.01

    def __post_init__(self) -> None:
        """检查 TD 与 entropy 系数为非负；不在此处构造网络。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            None；非法配置在构造阶段抛 ValueError。
        """
        TDLossObjective(self.td_coefficient)
        EntropyObjective(self.entropy_coefficient)


@dataclass(frozen=True, slots=True)
class MAACConfig:
    """MAAC 配置；环境尺寸由 spec 注入。"""

    schema_version: Literal[2] = 2
    algorithm: Literal["maac"] = "maac"
    policy: Buildable[DiscretePolicy] = IndependentDiscreteConfig()
    critic: Buildable[AttentionQNetwork] = AttentionQConfig()
    loss: MAACLossConfig = MAACLossConfig()
    value_target: TD0Config = TD0Config()
    replay: ReplayConfig = ReplayConfig()
    update: ActorCriticUpdateConfig = ActorCriticUpdateConfig()
    target_update: SoftTargetConfig | HardTargetConfig = SoftTargetConfig()


    def validate(self, spec: EnvironmentSpec) -> None:
        """验证配置版本和算法标识，并要求环境使用离散动作。

        Args:
            spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。

        Returns:
            None；不兼容时抛 ValueError。
        """
        if self.schema_version != 2 or self.algorithm != "maac":
            raise ValueError("MAAC config 的 algorithm/schema_version 不匹配")
        if spec.action_kind != ActionKind.DISCRETE:
            raise ValueError("MAAC 不支持当前动作类型")

    def build(self, spec: EnvironmentSpec) -> MAAC:
        """以当前配置和环境尺寸直接构造 MAAC。

        Args:
            spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。

        Returns:
            新算法实例；optimizer/trainer 由 build_experiment 另外装配。
        """
        return MAAC(spec, self)


class MAAC(BaseMARLAlgorithm):
    """离散反事实 actor-critic；每个玩家保留自己的策略和 Q head。

    ``compute_*_loss_bundle`` 只构图，``update`` 才决定 critic→各 actor→target。
    policy loss 固定其他玩家动作，不通过 critic 回传；critic loss 使用 replay
    中的真实联合动作。采集和 replay 生命周期仍由通用 trainer 管理。
    """

    def __init__(self, spec: EnvironmentSpec, config: MAACConfig) -> None:
        """校验配置并装配 MAAC 的在线网络、数学目标及所需 target/统计模块。

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
        if getattr(self.policy, "is_recurrent", False):
            raise ValueError("MAAC 尚未支持循环 policy 的序列 replay；请使用 MLP backbone")
        self.critic = config.critic.build(spec)
        self.target_policy = frozen_target(self.policy)
        self.target_critic = frozen_target(self.critic)
        self.td_loss = TDLossObjective(config.loss.td_coefficient)
        self.target_scale = TargetScale(spec.num_agents, config.loss.normalize_targets)
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
        """将各智能体观测映射为联合动作。在合法离散动作上采样或选择最大 logit。

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
        """校验公共 batch 形状以及 MAAC 所需的 actions/rewards/next_observations。

        Args:
            batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。

        Returns:
            None；字段缺失或形状不符抛 ValueError。
        """
        batch.validate(self.spec.num_agents, self.spec.observation_dim)
        if batch.actions is None or batch.rewards is None or batch.next_observations is None:
            raise ValueError("MAAC 训练需要 actions/rewards/next_observations")

    def compute_critic_loss_bundle(
        self, batch: MARLBatch, *, update_statistics: bool = False
    ) -> LossBundle:
        """使用 replay 联合动作拟合逐智能体 TD target；下一值来自target
            policy、attention Q 及熵正则。

        使用 replay 联合动作回归逐智能体 TD target。

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
        if update_statistics:
            self.target_scale.update(target_q)
        critic_result = self.td_loss(self.target_scale(chosen_q), self.target_scale(target_q))
        bundle = LossBundle.combine((critic_result,))
        return LossBundle(bundle.total, {**bundle.terms,
            **value_diagnostics(chosen_q, target_q),
            **agent_statistics("reward", batch.rewards)})

    def compute_actor_loss_bundle(
        self,
        batch: MARLBatch,
        agent_index: int,
        *,
        current_q_values: Tensor | None = None,
    ) -> LossBundle:
        """构建单智能体反事实策略目标；固定 Q 表和其他玩家动作。

        只构建一个智能体的 policy loss；其他 actor 不接收梯度。

        Args:
            batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。
            agent_index: 目标智能体的零起始索引，必须位于 [0,N)。
            current_q_values: 可复用的当前联合动作 Q 表 [...,N,A]；None 时内部无梯度采样并计算。

        Returns:
            actor LossBundle；只构图，不执行 optimizer.step。
        """

        self._validate_training_batch(batch)
        if not 0 <= agent_index < self.spec.num_agents:
            raise IndexError("agent_index 超出智能体范围")
        if current_q_values is None:
            with torch.no_grad():
                current_actions = self.policy.act(
                    batch.observations,
                    action_mask=batch.action_mask,
                ).actions
                current_q_values = self.critic(
                    batch.observations, current_actions
                )
        if isinstance(self.policy, AgentLogitsPolicy):
            selected_logits = self.policy.logits_for_agent(
                batch.observations,
                agent_index,
                batch.action_mask,
            )
        else:
            selected_logits = self.policy.logits(
                batch.observations, batch.action_mask
            )[..., agent_index, :]
        selected_q = current_q_values[..., agent_index, :]
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
        """对一批 replay 经验按 critic → 逐智能体 actor → target
            顺序更新参数；复用基类 optimize。

        按 critic → 每个智能体 policy → target 的 MAAC 顺序更新。

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
        actor_totals: dict[str, Tensor] = {}
        actor_norms: list[Tensor] = []
        agent_metrics: dict[str, Tensor] = {}
        with torch.no_grad():
            current_actions = self.policy.act(
                batch.observations,
                action_mask=batch.action_mask,
            ).actions
            current_q_values = self.critic(batch.observations, current_actions)
        for agent_index in range(self.spec.num_agents):
            with runtime.autocast(device):
                actor = self.compute_actor_loss_bundle(
                    batch,
                    agent_index,
                    current_q_values=current_q_values,
                )
            actor_norms.append(
                self.optimize(
                    runtime.optimizer("actor"),
                    actor,
                    runtime.max_grad_norm("actor"),
                    parameters=runtime.parameters("actor"),
                    # 内置独立 actor 的未参与参数天然 grad=None；外部拓扑保留安全回退。
                    drop_zero_gradients=not isinstance(self.policy, IndependentDiscretePolicy),
                )
            )
            runtime.record_optimizer_step("actor", actor_norms[-1])
            agent_metrics[f"entropy_agent_{agent_index}_mean"] = actor.terms["entropy"].detach()
            for name, value in actor.terms.items():
                actor_totals[name] = (
                    actor_totals.get(name, torch.zeros_like(value)) + value.detach()
                )
        runtime.finish(self.target_pairs())
        metrics = dict(critic.terms)
        metrics.update(agent_metrics)
        metrics.update(
            {name: value / self.spec.num_agents for name, value in actor_totals.items()}
        )
        metrics["loss"] = critic.total.detach() + actor_totals["loss"] / self.spec.num_agents
        metrics["critic_gradient_norm"] = critic_norm
        metrics["actor_gradient_norm"] = torch.stack(actor_norms).mean()
        metrics["gradient_norm"] = torch.maximum(
            metrics["critic_gradient_norm"], metrics["actor_gradient_norm"]
        )
        return runtime.export_metrics(metrics)

    def target_pairs(self) -> tuple[tuple[nn.Module, nn.Module], ...]:
        """声明目标网络与在线网络的配对，供 runtime.finish 统一同步。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            有序 (target, online) 对；基类默认空，不在本函数执行同步。
        """
        return (
            (self.get_submodule("target_policy"), self.get_submodule("policy")),
            (self.get_submodule("target_critic"), self.get_submodule("critic")),
        )
