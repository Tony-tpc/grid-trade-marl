"""一般和博弈版 MADDPG：每个智能体优化自己的累计奖励。
    replay_buffer: (ot​,at​,rt​,ot+1​,donet​) at+1由TD目标网络生成
    一个 Agent 对应一个 Actor 、一个 Critic 
    Critic 是集中式的，读取所有智能体的观测和动作。
    Actor 是独立的，只读取各自智能体的观测。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from marl.algorithms.base import AlgorithmConfig, BaseMARLAlgorithm
from marl.core.batch import MARLBatch
from marl.models.mlp import MLPBackbone
from marl.modules.action_head import DeterministicActionHead
from marl.modules.actor import Actor
from marl.modules.critic import CentralizedCritic


@dataclass(frozen=True, slots=True)
class MADDPGConfig(AlgorithmConfig):
    num_agents: int = 2
    observation_dim: int = 16
    action_dim: int = 2
    hidden_dim: int = 128


class MADDPG(BaseMARLAlgorithm):
    """每个智能体有独立 Actor 和集中式 Critic 的 MADDPG。

    Actor_i 只读取 o_i，输出自己的连续动作 a_i。Critic_i 在训练时读取联合
    (o_1:N, a_1:N)，预测的是智能体 i 的 Q_i。其 Bellman 标签必须使用 r_i，
    不能把不同家庭/玩家的奖励平均。所有 Actor 的输入输出维度目前要求相同，
    但网络参数独立，可学习不同的博弈策略。
    """

    def __init__(self, config: MADDPGConfig) -> None:
        super().__init__(config)
        self.actors = nn.ModuleList(
            Actor(
                MLPBackbone(config.observation_dim, output_dim=config.hidden_dim),
                DeterministicActionHead(config.hidden_dim, config.action_dim),
            )
            for _ in range(config.num_agents)
        )
        critic_input = config.num_agents * (
            config.observation_dim + config.action_dim
        )  # [observation,action]
        self.critics = nn.ModuleList(
            CentralizedCritic(MLPBackbone(critic_input, output_dim=config.hidden_dim))
            for _ in range(config.num_agents)
        )
        self.target_actors = deepcopy(self.actors).requires_grad_(False)
        self.target_critics = deepcopy(self.critics).requires_grad_(False)

    @property
    def cfg(self) -> MADDPGConfig:
        return self.config  # type: ignore[return-value]

    @staticmethod
    def _critic_input(observations: Tensor, actions: Tensor) -> Tensor:
        """把联合观测和动作展平后拼成 ``[B, N * (O + A)]``。"""
        return torch.cat(
            (
                observations.flatten(-2),
                actions.flatten(-2),
            ),
            dim=-1,
        )

    @staticmethod
    def _joint_actions(actors: nn.ModuleList, observations: Tensor) -> Tensor:
        return torch.stack(
            [
                # 第 i 个 Actor 只读取第 i 个 Agent 的局部观测。
                actor.forward(observations[..., i, :], deterministic=True).actions
                for i, actor in enumerate(actors)
            ],
            dim=-2,
            # 将所有智能体的动作堆叠成 [batch, num_agents, action_dim]
        )

    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
        **kwargs: Tensor,
    ) -> Tensor:
        """独立 Actor 并行执行；DDPG 探索噪声由采样器加在返回动作上。"""
        return self._joint_actions(self.actors, observations)

    def compute_loss(self, batch: MARLBatch) -> dict[str, Tensor]:
        if batch.actions is None or batch.rewards is None or batch.next_observations is None:
            raise ValueError("MADDPG 训练需要 actions/rewards/next_observations")
        if batch.rewards.shape[-1] != self.cfg.num_agents:
            raise ValueError("MADDPG 需要每个智能体各自的奖励 [B,N]")
        dones = batch.dones if batch.dones is not None else torch.zeros_like(batch.rewards)


        # ------------------- Critic Loss ------------------------
        # 每个 Q_i 都看完整联合经验，但回归各自的 r_i + gamma Q_i'。
        current_input = self._critic_input(batch.observations, batch.actions)
        current_q = torch.stack(
            [critic.forward(current_input).squeeze(-1) for critic in self.critics], dim=-1
        )
        with torch.no_grad():
            next_actions = self._joint_actions(self.target_actors, batch.next_observations)
            next_input = self._critic_input(batch.next_observations, next_actions)
            # 计算每个actor的Q_value 
            next_q = torch.stack(
                [critic.forward(next_input).squeeze(-1) for critic in self.target_critics],
                dim=-1,
            )
            target_q = batch.rewards + self.cfg.gamma * (1.0 - dones.float()) * next_q
        # TD-ERROR
        critic_loss = F.mse_loss(current_q, target_q)

        # ------------------- Actor Loss ------------------------
        # 更新 Actor_i 时，只让自己的动作 a_i 保持梯度。其他 Actor 的动作固定，
        # 因为当前优化的是 J_i，而不是把别人的收益也算进同一个目标。
        policy_actions = self._joint_actions(self.actors, batch.observations)
        # 存储每一个 Agent 的 actor loss
        actor_terms = []
        for i, critic in enumerate(self.critics):
            # joint_actions = [detach(a_(j不等于i),...,a_(i=j),...]
            joint_actions = torch.stack(
                [
                    # 当前 Agent 的动作保留梯度，其他 Agent 的动作视作常量。
                    policy_actions[..., j, :] if j == i else policy_actions[..., j, :].detach()
                    for j in range(self.cfg.num_agents)
                ],
                dim=-2,
            )
            with self.frozen(critic):
                q_i = critic.forward(self._critic_input(batch.observations, joint_actions))
            actor_terms.append(-q_i.mean())
        actor_loss = torch.stack(actor_terms).mean()


        return {
            "loss": actor_loss + critic_loss,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
        }

    def update_targets(self) -> None:
        for target, online in zip(self.target_actors, self.actors, strict=True):
            self.soft_update(target, online)
        for target, online in zip(self.target_critics, self.critics, strict=True):
            self.soft_update(target, online)
