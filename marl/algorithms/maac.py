"""论文启发的离散动作 MAAC，保留每个智能体的独立目标。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import functional as F

from marl.algorithms.base import AlgorithmConfig, BaseMARLAlgorithm
from marl.core.batch import MARLBatch
from marl.models.mlp import MLPBackbone
from marl.modules.action_head import DiscreteActionHead
from marl.modules.actor import Actor


@dataclass(frozen=True, slots=True)
class MAACConfig(AlgorithmConfig):
    num_agents: int = 2
    observation_dim: int = 16
    action_dim: int = 5
    hidden_dim: int = 128
    attention_heads: int = 4
    entropy_coef: float = 0.01


class AttentionCritic(nn.Module):
    """共享注意力层 + 每个智能体各自的 Q_i 输出。

    对智能体 i，注意力只聚合其他智能体的 (o_j,a_j)，不读取 a_i。然后把
    自己的观测编码和这个上下文拼起来，输出所有候选动作的 Q_i(a_i)。
    因此固定 a_-i 时，可以直接比较不同 a_i 的反事实价值。
    形状：observations [B,N,O]，actions [B,N]，输出 [B,N,A]。
    """

    def __init__(self, config: MAACConfig) -> None:
        super().__init__()
        if config.hidden_dim % config.attention_heads:
            raise ValueError("hidden_dim 必须能被 attention_heads 整除")
        self.action_dim = config.action_dim
        self.own_projection = nn.Linear(config.observation_dim, config.hidden_dim)
        self.other_projection = nn.Linear(
            config.observation_dim + config.action_dim, config.hidden_dim
        )
        self.attention = nn.MultiheadAttention(
            config.hidden_dim, config.attention_heads, batch_first=True
        )
        self.q_head = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

    def forward(self, observations: Tensor, actions: Tensor) -> Tensor:
        leading, agents = observations.shape[:-2], observations.shape[-2]
        own = self.own_projection(observations).reshape(
            -1, agents, self.own_projection.out_features
        )
        one_hot = F.one_hot(actions.long(), self.action_dim).to(observations.dtype)
        others = self.other_projection(torch.cat((observations, one_hot), dim=-1))
        others = others.reshape(-1, agents, others.shape[-1])
        if agents == 1:
            context = torch.zeros_like(own)
        else:
            # 对角线 True 表示 i 不能把自己的 replay 动作 a_i 当作“别人信息”读取。
            exclude_self = torch.eye(agents, device=observations.device, dtype=torch.bool)
            context, _ = self.attention(
                own, others, others, attn_mask=exclude_self, need_weights=False
            )
        q_values = self.q_head(torch.cat((own, context), dim=-1))
        return cast(Tensor, q_values.reshape(*leading, agents, self.action_dim))


class MAAC(BaseMARLAlgorithm):
    """每个智能体独立 Actor，Critic 通过共享注意力参数联合训练。

    Bellman 标签 y_i 使用环境返回的 r_i，Actor_i 优化自己的优势函数。
    当前 Actor 是离散分类分布；论文原始连续+二元混合动作还需专用动作头。
    """

    def __init__(self, config: MAACConfig) -> None:
        super().__init__(config)
        self.actors = nn.ModuleList(
            Actor(
                MLPBackbone(config.observation_dim, output_dim=config.hidden_dim),
                DiscreteActionHead(config.hidden_dim, config.action_dim),
            )
            for _ in range(config.num_agents)
        )
        self.critic = AttentionCritic(config)
        self.target_actors = deepcopy(self.actors).requires_grad_(False)
        self.target_critic = deepcopy(self.critic).requires_grad_(False)

    @property
    def cfg(self) -> MAACConfig:
        return self.config  # type: ignore[return-value]

    @staticmethod
    def _joint_policy(
        actors: nn.ModuleList,
        observations: Tensor,
        action_mask: Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        outputs = [
            actor.forward(
                observations[..., i, :],
                deterministic=deterministic,
                action_mask=(action_mask[..., i, :] if action_mask is not None else None),
            )
            for i, actor in enumerate(actors)
        ]
        return (
            torch.stack([output.actions for output in outputs], dim=-1),
            torch.stack([output.log_prob for output in outputs], dim=-1),
            torch.stack([output.distribution_params["logits"] for output in outputs], dim=-2),
        )

    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
        **kwargs: Tensor,
    ) -> Tensor:
        return self._joint_policy(self.actors, observations, action_mask, deterministic)[0]

    def compute_loss(self, batch: MARLBatch) -> dict[str, Tensor]:
        if batch.actions is None or batch.rewards is None or batch.next_observations is None:
            raise ValueError("MAAC 训练需要 actions/rewards/next_observations")
        if batch.rewards.shape[-1] != self.cfg.num_agents:
            raise ValueError("MAAC 需要每个智能体各自的奖励 [B,N]")
        dones = batch.dones if batch.dones is not None else torch.zeros_like(batch.rewards)
        q_all = self.critic(batch.observations, batch.actions)
        chosen_q = q_all.gather(-1, batch.actions.long().unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_actions, next_log_prob, _ = self._joint_policy(
                self.target_actors, batch.next_observations, batch.next_action_mask
            )
            next_q_all = self.target_critic(batch.next_observations, next_actions)
            next_q = next_q_all.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)
            target_q = batch.rewards + self.cfg.gamma * (1.0 - dones.float()) * (
                next_q - self.cfg.entropy_coef * next_log_prob
            )
        critic_loss = F.mse_loss(chosen_q, target_q)

        # 反事实 baseline：固定其他智能体动作，按 pi_i 对候选 a_i 求期望。
        _, _, logits = self._joint_policy(self.actors, batch.observations, batch.action_mask)
        distribution = Categorical(logits=logits)
        policy_q = self.critic(batch.observations, batch.actions).detach()
        probabilities = distribution.probs.detach()
        baseline = (probabilities * policy_q).sum(dim=-1, keepdim=True)
        advantage = (policy_q - baseline).detach()
        actor_loss = -(probabilities * advantage * distribution.logits).sum(dim=-1).mean()
        entropy = distribution.entropy().mean()
        total = actor_loss + critic_loss - self.cfg.entropy_coef * entropy
        return {
            "loss": total,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy,
        }

    def update_targets(self) -> None:
        for target, online in zip(self.target_actors, self.actors, strict=True):
            self.soft_update(target, online)
        self.soft_update(self.target_critic, self.critic)
