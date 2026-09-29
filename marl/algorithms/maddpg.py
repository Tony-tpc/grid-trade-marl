"""一般和博弈版 MADDPG：每个智能体优化自己的累计奖励。
    replay 保存观测、联合动作、逐体奖励、下一观测及独立的终止/截断标记。
    下一动作由 target actor 生成；只有真正终止阻止 TD bootstrap。
    一个 Agent 对应一个 Actor 、一个 Critic 
    Critic 是集中式的，读取所有智能体的观测和动作。
    Actor 是独立的，只读取各自智能体的观测。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from marl.algorithms.base import BaseMARLAlgorithm
from marl.core import MARLBatch
from marl.envs.base import ActionKind, EnvironmentSpec
from marl.extensions import Buildable
from marl.modules.critic import IndependentQConfig, QEnsemble, centralized_critic_input
from marl.modules.policy import (
    IndependentDeterministicConfig,
    PolicyTopology,
)
from marl.objectives import (
    DeterministicPolicyObjective,
    LossBundle,
    TDLossObjective,
)
from marl.returns import TD0Config, ValueTargetEstimator
from marl.target_updates import HardTargetConfig, SoftTargetConfig, frozen_target
from marl.training.gradients import frozen_parameters, isolate_agent_action
from marl.training.off_policy import (
    ActorCriticUpdateConfig,
    ReplayConfig,
)
from marl.training.optimization import OptimizerRuntime
from marl.value_scaling import TargetScale, agent_statistics, value_diagnostics


@dataclass(frozen=True, slots=True)
class MADDPGLossConfig:
    td_coefficient: float = 1.0
    normalize_targets: bool = False

    def __post_init__(self) -> None:
        """检查 TD 损失系数为非负；不在此处构造网络。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            None；非法配置在构造阶段抛 ValueError。
        """
        TDLossObjective(self.td_coefficient)


@dataclass(frozen=True, slots=True)
class MADDPGConfig:
    """MADDPG 配置；环境尺寸由 spec 注入。"""

    schema_version: Literal[2] = 2
    algorithm: Literal["maddpg"] = "maddpg"
    policy: Buildable[PolicyTopology] = IndependentDeterministicConfig()
    critic: Buildable[QEnsemble] = IndependentQConfig()
    loss: MADDPGLossConfig = MADDPGLossConfig()
    value_target: TD0Config = TD0Config()
    replay: ReplayConfig = ReplayConfig()
    update: ActorCriticUpdateConfig = ActorCriticUpdateConfig()
    target_update: SoftTargetConfig | HardTargetConfig = SoftTargetConfig()


    def validate(self, spec: EnvironmentSpec) -> None:
        """验证配置版本和算法标识，并要求环境使用连续动作。

        Args:
            spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。

        Returns:
            None；不兼容时抛 ValueError。
        """
        if self.schema_version != 2 or self.algorithm != "maddpg":
            raise ValueError("MADDPG config 的 algorithm/schema_version 不匹配")
        if spec.action_kind != ActionKind.CONTINUOUS:
            raise ValueError("MADDPG 不支持当前动作类型")

    def build(self, spec: EnvironmentSpec) -> MADDPG:
        """以当前配置和环境尺寸直接构造 MADDPG。

        Args:
            spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。

        Returns:
            新算法实例；optimizer/trainer 由 build_experiment 另外装配。
        """
        return MADDPG(spec, self)


class MADDPG(BaseMARLAlgorithm):
    """每个智能体有独立 Actor 和集中式 Critic 的 MADDPG。

    Actor_i 只读取 o_i，输出自己的连续动作 a_i。Critic_i 在训练时读取联合
    (o_1:N, a_1:N)，预测的是智能体 i 的 Q_i。其 Bellman 标签必须使用 r_i，
    不能把不同家庭/玩家的奖励平均。所有 Actor 的输入输出维度目前要求相同，
    但网络参数独立，可学习不同的博弈策略。
    """
    def __init__(self, spec: EnvironmentSpec, config: MADDPGConfig) -> None:
        """校验配置并装配 MADDPG 的在线网络、数学目标及所需 target/统计模块。

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
        self.critics = config.critic.build(spec)
        self.target_policy = frozen_target(self.policy)
        self.target_critics = frozen_target(self.critics)
        self.td_loss = TDLossObjective(config.loss.td_coefficient)
        self.target_scale = TargetScale(spec.num_agents, config.loss.normalize_targets)
        self.policy_objective = DeterministicPolicyObjective()
        self.return_estimator: ValueTargetEstimator = config.value_target.build()

    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
        **kwargs: Tensor,
    ) -> Tensor:
        """将各智能体观测映射为联合动作。始终返回确定性动作，探索噪声应由采集器加入。

        独立 Actor 并行执行；DDPG 探索噪声由采样器加在返回动作上。

        Args:
            observations: 浮点观测 [...,N,O]；循环单步为 [B,N,O]，序列为 [B,T,N,O]。
            deterministic: 是否采用确定性动作；具体算法的例外见本函数说明。
            action_mask: 可选 bool [...,N,A]，True 表示合法；连续动作必须为 None。
            kwargs: 保留的接口扩展关键字；当前具体算法 act 不读取这些值。

        Returns:
            动作 Tensor：离散 [...,N]，连续 [...,N,A]；此接口不返回 log-prob/value。
        """
        return self.policy.act(observations, deterministic=True, action_mask=action_mask).actions

    def compute_loss_bundle(self, batch: MARLBatch) -> LossBundle:
        """汇总各子目标用于数学诊断；只构图，不执行 backward、step 或 target 统计更新。

        Args:
            batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。

        Returns:
            LossBundle；total 是标量总目标，terms 是各项日志；正常训练仍应调用 update。
        """
        critic = self.compute_critic_loss_bundle(batch)
        actor = self.compute_actor_loss_bundle(batch)
        terms = {**critic.terms, **actor.terms}
        terms["loss"] = critic.total + actor.total
        return LossBundle(terms["loss"], terms)

    def _validate_training_batch(self, batch: MARLBatch) -> None:
        """校验公共 batch 形状以及 MADDPG 所需的 actions/rewards/next_observations。

        Args:
            batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。

        Returns:
            None；字段缺失或形状不符抛 ValueError。
        """
        batch.validate(self.spec.num_agents, self.spec.observation_dim)
        if batch.actions is None or batch.rewards is None or batch.next_observations is None:
            raise ValueError("MADDPG 训练需要 actions/rewards/next_observations")

    def compute_critic_loss_bundle(
        self, batch: MARLBatch, *, update_statistics: bool = False
    ) -> LossBundle:
        """使用 replay 联合动作拟合逐智能体 TD target；下一值来自target actor 与独立集中式 Q。

        使用 replay 联合动作更新全部独立 centralized critics。

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
        # ------------------- Critic Loss ------------------------
        # 每个 Q_i 都看完整联合经验，但回归各自的 r_i + gamma Q_i'。
        current_input = centralized_critic_input(batch.observations, batch.actions)
        current_q = self.critics(current_input)
        with torch.no_grad():
            next_actions = self.target_policy.act(batch.next_observations).actions
            next_q = self.target_critics(
                centralized_critic_input(batch.next_observations, next_actions)
            )
            target_q = self.return_estimator.estimate(
                batch.rewards, next_q, terminated
            )
        if update_statistics:
            self.target_scale.update(target_q)
        critic_result = self.td_loss(self.target_scale(current_q), self.target_scale(target_q))
        bundle = LossBundle.combine((critic_result,))
        return LossBundle(bundle.total, {**bundle.terms,
            **value_diagnostics(current_q, target_q),
            **agent_statistics("reward", batch.rewards)})

    def compute_actor_loss_bundle(self, batch: MARLBatch) -> LossBundle:
        """冻结 critic 参数并隔离其他 actor 动作，构建每个智能体自己的策略目标。

        固定每个 Q_i 及其他 actor 动作，只更新对应 Actor_i。

        Args:
            batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。

        Returns:
            actor LossBundle；只构图，不执行 optimizer.step。
        """

        self._validate_training_batch(batch)
        # ------------------- Actor Loss ------------------------
        # 更新 Actor_i 时，只让自己的动作 a_i 保持梯度。其他 Actor 的动作固定，
        # 因为当前优化的是 J_i，而不是把别人的收益也算进同一个目标。
        policy_actions = self.policy.act(batch.observations).actions
        # 存储每一个 Agent 的 actor loss
        q_terms = []
        for index, critic in enumerate(self.critics.critics):
            # joint_actions = [detach(a_(j不等于i),...,a_(i=j),...]
            joint_actions = isolate_agent_action(policy_actions, index)
            with frozen_parameters(critic):
                q_terms.append(
                    critic(
                        centralized_critic_input(batch.observations, joint_actions)
                    ).squeeze(-1)
                )
        # 所有 Agent 的 -Q_i 在 batch 和 agent 两个维度上做平均后的结果
        actor_result = self.policy_objective(torch.stack(q_terms, dim=-1))
        return LossBundle.combine((actor_result,))

    def update(
        self, batch: MARLBatch, runtime: OptimizerRuntime
    ) -> dict[str, float]:
        """对一批 replay 经验按 critic → actor → target 顺序更新参数；复用基类 optimize。

        按 critic → actor → target 的 MADDPG 顺序更新。

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
        runtime.finish(self.target_pairs())
        metrics = {**critic.terms, **actor.terms}
        metrics["loss"] = critic.total.detach() + actor.total.detach()
        metrics["critic_gradient_norm"] = critic_norm
        metrics["actor_gradient_norm"] = actor_norm
        metrics["gradient_norm"] = torch.maximum(critic_norm, actor_norm)
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
            (self.get_submodule("target_critics"), self.get_submodule("critics")),
        )
