"""一般和博弈版 MADDPG：每个智能体优化自己的累计奖励。
    replay_buffer: (ot​,at​,rt​,ot+1​,donet​) at+1由TD目标网络生成
    一个 Agent 对应一个 Actor 、一个 Critic 
    Critic 是集中式的，读取所有智能体的观测和动作。
    Actor 是独立的，只读取各自智能体的观测。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import Tensor, nn

from marl.algorithms.assembly import assemble_algorithm_components, require_one
from marl.algorithms.base import BaseMARLAlgorithm
from marl.core import MARLBatch
from marl.envs.base import EnvironmentSpec
from marl.modules.critic import IndependentCentralizedCritics, IndependentQConfig
from marl.modules.policy import IndependentDeterministicConfig, IndependentDeterministicPolicy
from marl.objectives import (
    DeterministicPolicyObjective,
    LossBundle,
    TDLossObjective,
)
from marl.recipes import AlgorithmRecipe, CompiledRecipe, ComponentRecipe
from marl.registry import DEFAULT_COMPONENT_REGISTRY, ComponentRegistry
from marl.returns import TD0Config, TD0Estimator
from marl.target_updates import HardTargetConfig, SoftTargetConfig
from marl.training.gradients import frozen_parameters
from marl.training.off_policy import OffPolicyUpdateConfig, ReplayConfig


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
    policy: IndependentDeterministicConfig = IndependentDeterministicConfig()
    critic: IndependentQConfig = IndependentQConfig()
    loss: MADDPGLossConfig = MADDPGLossConfig()
    value_target: TD0Config = TD0Config()
    replay: ReplayConfig = ReplayConfig()
    update: OffPolicyUpdateConfig = OffPolicyUpdateConfig()
    target_update: SoftTargetConfig | HardTargetConfig = SoftTargetConfig()


def default_maddpg_recipe() -> AlgorithmRecipe:
    return AlgorithmRecipe(
        schema_version=1,
        algorithm="maddpg",
        policy=ComponentRecipe("independent_deterministic", {"hidden_dim": 128}),
        critic=ComponentRecipe("independent_centralized_q", {"hidden_dim": 128}),
        objectives=(
            ComponentRecipe("td_mse", {"coefficient": 1.0}),
            ComponentRecipe("deterministic_policy", {}),
        ),
        returns=ComponentRecipe("td0", {"gamma": 0.99}),
        experience=ComponentRecipe("replay", {"capacity": 100_000, "batch_size": 256}),
        update=ComponentRecipe(
            "off_policy_update",
            {"learning_rate": 3e-4, "max_grad_norm": 10.0, "amp_dtype": None},
        ),
        target_update=ComponentRecipe("soft", {"tau": 0.005}),
    )


class MADDPG(BaseMARLAlgorithm):
    """每个智能体有独立 Actor 和集中式 Critic 的 MADDPG。

    Actor_i 只读取 o_i，输出自己的连续动作 a_i。Critic_i 在训练时读取联合
    (o_1:N, a_1:N)，预测的是智能体 i 的 Q_i。其 Bellman 标签必须使用 r_i，
    不能把不同家庭/玩家的奖励平均。所有 Actor 的输入输出维度目前要求相同，
    但网络参数独立，可学习不同的博弈策略。
    """
    def __init__(
        self,
        spec: EnvironmentSpec,
        policy: IndependentDeterministicPolicy,
        critics: IndependentCentralizedCritics,
        td_loss: TDLossObjective,
        policy_objective: DeterministicPolicyObjective,
        return_estimator: TD0Estimator,
        compiled_recipe: CompiledRecipe,
    ) -> None:
        super().__init__(spec)
        self.policy = policy
        self.critics = critics
        self.target_policy = deepcopy(policy).requires_grad_(False)
        self.target_critics = deepcopy(critics).requires_grad_(False)
        self.td_loss = td_loss
        self.policy_objective = policy_objective
        self.return_estimator = return_estimator
        self.compiled_recipe = compiled_recipe

    @classmethod
    def from_recipe(
        cls,
        spec: EnvironmentSpec,
        recipe: AlgorithmRecipe,
        *,
        registry: ComponentRegistry = DEFAULT_COMPONENT_REGISTRY,
    ) -> MADDPG:
        parts = assemble_algorithm_components("maddpg", spec, recipe, registry)
        if not isinstance(parts.policy, IndependentDeterministicPolicy):
            raise TypeError("MADDPG policy 必须是 IndependentDeterministicPolicy")
        if not isinstance(parts.critic, IndependentCentralizedCritics):
            raise TypeError("MADDPG critic 必须是 IndependentCentralizedCritics")
        if not isinstance(parts.returns, TD0Estimator):
            raise TypeError("MADDPG returns 必须是 TD0Estimator")
        return cls(
            spec,
            parts.policy,
            parts.critic,
            cast(TDLossObjective, require_one(parts.objectives, TDLossObjective, "MADDPG")),
            cast(
                DeterministicPolicyObjective,
                require_one(parts.objectives, DeterministicPolicyObjective, "MADDPG"),
            ),
            parts.returns,
            parts.compiled,
        )

    @staticmethod
    def _critic_input(observations: Tensor, actions: Tensor) -> Tensor:
        """把联合观测和动作展平后拼成 ``[B, N * (O + A)]``。"""
        return torch.cat((observations.flatten(-2), actions.flatten(-2)), dim=-1)

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
        batch.validate(self.spec.num_agents, self.spec.observation_dim)
        if batch.actions is None or batch.rewards is None or batch.next_observations is None:
            raise ValueError("MADDPG 训练需要 actions/rewards/next_observations")
        terminated, _ = batch.terminal_flags()
        # ------------------- Critic Loss ------------------------
        # 每个 Q_i 都看完整联合经验，但回归各自的 r_i + gamma Q_i'。
        current_input = self._critic_input(batch.observations, batch.actions)
        current_q = self.critics(current_input)
        with torch.no_grad():
            next_actions = self.target_policy.act(batch.next_observations).actions
            next_q = self.target_critics(
                self._critic_input(batch.next_observations, next_actions)
            )
            target_q = self.return_estimator.estimate(
                batch.rewards, next_q, terminated
            )
        critic_result = self.td_loss(current_q, target_q)

        # ------------------- Actor Loss ------------------------
        # 更新 Actor_i 时，只让自己的动作 a_i 保持梯度。其他 Actor 的动作固定，
        # 因为当前优化的是 J_i，而不是把别人的收益也算进同一个目标。
        policy_actions = self.policy.act(batch.observations).actions
        # 存储每一个 Agent 的 actor loss
        q_terms = []
        for index, critic in enumerate(self.critics.critics):
            # joint_actions = [detach(a_(j不等于i),...,a_(i=j),...]
            joint_actions = torch.stack(
                [
                    # 当前 Agent 的动作保留梯度，其他 Agent 的动作视作常量。
                    policy_actions[..., other, :]
                    if other == index
                    else policy_actions[..., other, :].detach()
                    for other in range(self.spec.num_agents)
                ],
                dim=-2,
            )
            with frozen_parameters(critic):
                q_terms.append(
                    critic(self._critic_input(batch.observations, joint_actions)).squeeze(-1)
                )
        # 所有 Agent 的 -Q_i 在 batch 和 agent 两个维度上做平均后的结果
        actor_result = self.policy_objective(torch.stack(q_terms, dim=-1))
        return LossBundle.combine((critic_result, actor_result))

    def target_pairs(self) -> tuple[tuple[nn.Module, nn.Module], ...]:
        return (
            (self.target_policy, self.policy),
            (self.target_critics, self.critics),
        )
