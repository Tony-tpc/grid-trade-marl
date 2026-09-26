"""一般和博弈版 MADDPG：每个智能体优化自己的累计奖励。
    replay_buffer: (ot​,at​,rt​,ot+1​,donet​) at+1由TD目标网络生成
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
from marl.training.optimization import OptimizerRuntime, metrics_to_float


@dataclass(frozen=True, slots=True)
class MADDPGUpdateConfig(ActorCriticUpdateConfig):
    """MADDPG 的 critic→actor 更新参数。"""


@dataclass(frozen=True, slots=True)
class MADDPGLossConfig:
    td_coefficient: float = 1.0

    def __post_init__(self) -> None:
        TDLossObjective(self.td_coefficient)


@dataclass(frozen=True, slots=True)
class MADDPGConfig:
    """MADDPG 配置；环境尺寸由 spec 注入。"""

    schema_version: Literal[1] = 1
    algorithm: Literal["maddpg"] = "maddpg"
    policy: Buildable[PolicyTopology] = IndependentDeterministicConfig()
    critic: Buildable[QEnsemble] = IndependentQConfig()
    loss: MADDPGLossConfig = MADDPGLossConfig()
    value_target: TD0Config = TD0Config()
    replay: ReplayConfig = ReplayConfig()
    update: MADDPGUpdateConfig = MADDPGUpdateConfig()
    target_update: SoftTargetConfig | HardTargetConfig = SoftTargetConfig()


    def validate(self, spec: EnvironmentSpec) -> None:
        if self.schema_version != 1 or self.algorithm != "maddpg":
            raise ValueError("MADDPG config 的 algorithm/schema_version 不匹配")
        if spec.action_kind != ActionKind.CONTINUOUS:
            raise ValueError("MADDPG 不支持当前动作类型")

    def build(self, spec: EnvironmentSpec) -> MADDPG:
        return MADDPG(spec, self)


class MADDPG(BaseMARLAlgorithm):
    """每个智能体有独立 Actor 和集中式 Critic 的 MADDPG。

    Actor_i 只读取 o_i，输出自己的连续动作 a_i。Critic_i 在训练时读取联合
    (o_1:N, a_1:N)，预测的是智能体 i 的 Q_i。其 Bellman 标签必须使用 r_i，
    不能把不同家庭/玩家的奖励平均。所有 Actor 的输入输出维度目前要求相同，
    但网络参数独立，可学习不同的博弈策略。
    """
    def __init__(self, spec: EnvironmentSpec, config: MADDPGConfig) -> None:
        super().__init__(spec)
        config.validate(spec)
        self.config = config
        self.policy = config.policy.build(spec)
        self.critics = config.critic.build(spec)
        self.target_policy = frozen_target(self.policy)
        self.target_critics = frozen_target(self.critics)
        self.td_loss = TDLossObjective(config.loss.td_coefficient)
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
        """独立 Actor 并行执行；DDPG 探索噪声由采样器加在返回动作上。"""
        return self.policy.act(observations, deterministic=True, action_mask=action_mask).actions

    def compute_loss_bundle(self, batch: MARLBatch) -> LossBundle:
        critic = self.compute_critic_loss_bundle(batch)
        actor = self.compute_actor_loss_bundle(batch)
        terms = {**critic.terms, **actor.terms}
        terms["loss"] = critic.total + actor.total
        return LossBundle(terms["loss"], terms)

    def _validate_training_batch(self, batch: MARLBatch) -> None:
        batch.validate(self.spec.num_agents, self.spec.observation_dim)
        if batch.actions is None or batch.rewards is None or batch.next_observations is None:
            raise ValueError("MADDPG 训练需要 actions/rewards/next_observations")

    def compute_critic_loss_bundle(self, batch: MARLBatch) -> LossBundle:
        """使用 replay 联合动作更新全部独立 centralized critics。"""

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
        critic_result = self.td_loss(current_q, target_q)
        return LossBundle.combine((critic_result,))

    def compute_actor_loss_bundle(self, batch: MARLBatch) -> LossBundle:
        """固定每个 Q_i 及其他 actor 动作，只更新对应 Actor_i。"""

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
        """按 critic → actor → target 的 MADDPG 顺序更新。"""

        device = batch.observations.device
        with runtime.autocast(device):
            critic = self.compute_critic_loss_bundle(batch)
        critic_norm = self.optimize(
            runtime.optimizer("critic"),
            critic,
            runtime.max_grad_norm("critic"),
        )
        runtime.record_optimizer_step()
        with runtime.autocast(device):
            actor = self.compute_actor_loss_bundle(batch)
        actor_norm = self.optimize(
            runtime.optimizer("actor"),
            actor,
            runtime.max_grad_norm("actor"),
        )
        runtime.record_optimizer_step()
        runtime.finish(self.target_pairs())
        metrics = {**critic.terms, **actor.terms}
        metrics["loss"] = critic.total.detach() + actor.total.detach()
        metrics["critic_gradient_norm"] = critic_norm
        metrics["actor_gradient_norm"] = actor_norm
        metrics["gradient_norm"] = torch.maximum(critic_norm, actor_norm)
        return metrics_to_float(metrics)

    def target_pairs(self) -> tuple[tuple[nn.Module, nn.Module], ...]:
        return (
            (self.get_submodule("target_policy"), self.get_submodule("policy")),
            (self.get_submodule("target_critics"), self.get_submodule("critics")),
        )
