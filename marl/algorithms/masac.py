"""一般和博弈版多智能体 SAC，每个智能体有自己的策略与 Q 值。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from marl.algorithms.base import AlgorithmConfig, BaseMARLAlgorithm
from marl.core.batch import MARLBatch
from marl.models.mlp import MLPBackbone
from marl.modules.action_head import GaussianActionHead
from marl.modules.actor import Actor
from marl.modules.critic import CentralizedCritic


@dataclass(frozen=True, slots=True)
class MASACConfig(AlgorithmConfig):
    num_agents: int = 2
    observation_dim: int = 16
    action_dim: int = 2
    hidden_dim: int = 128
    initial_alpha: float = 0.2
    target_entropy: float | None = None


class MASAC(BaseMARLAlgorithm):
    """独立随机 Actor、每个智能体各自的集中式 twin-Q 与熵温度。

    Q_i(o_1:N,a_1:N) 回归 r_i，而不是团队奖励。Actor_i 只最大化自身
    Q_i 加自身策略熵。所有网络维度相同，但每个智能体参数独立。
    """

    def __init__(self, config: MASACConfig) -> None:
        super().__init__(config)
        if config.initial_alpha <= 0:
            raise ValueError("initial_alpha 必须大于 0")
        self.actors = nn.ModuleList(
            Actor(
                MLPBackbone(config.observation_dim, output_dim=config.hidden_dim),
                GaussianActionHead(config.hidden_dim, config.action_dim),
            )
            for _ in range(config.num_agents)
        )
        critic_input = config.num_agents * (config.observation_dim + config.action_dim)
        self.critics1 = nn.ModuleList(
            CentralizedCritic(MLPBackbone(critic_input, output_dim=config.hidden_dim))
            for _ in range(config.num_agents)
        )
        self.critics2 = nn.ModuleList(
            CentralizedCritic(MLPBackbone(critic_input, output_dim=config.hidden_dim))
            for _ in range(config.num_agents)
        )
        self.target_critics1 = deepcopy(self.critics1).requires_grad_(False)
        self.target_critics2 = deepcopy(self.critics2).requires_grad_(False)
        # 每个智能体独立调节探索温度 alpha_i；log 参数保证 exp 后为正。
        self.log_alpha = nn.Parameter(
            torch.full((config.num_agents,), float(config.initial_alpha)).log()
        )

    @property
    def cfg(self) -> MASACConfig:
        return self.config  # type: ignore[return-value]

    @staticmethod
    def _critic_input(observations: Tensor, actions: Tensor) -> Tensor:
        return torch.cat((observations.flatten(-2), actions.flatten(-2)), dim=-1)

    @staticmethod
    def _joint_policy(
        actors: nn.ModuleList, observations: Tensor, deterministic: bool = False
    ) -> tuple[Tensor, Tensor]:
        outputs = [
            actor.forward(observations[..., i, :], deterministic=deterministic)
            for i, actor in enumerate(actors)
        ]
        return (
            torch.stack([output.actions for output in outputs], dim=-2),
            torch.stack([output.log_prob for output in outputs], dim=-1),
        )

    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
        **kwargs: Tensor,
    ) -> Tensor:
        return self._joint_policy(self.actors, observations, deterministic)[0]

    def compute_loss(self, batch: MARLBatch) -> dict[str, Tensor]:
        if batch.actions is None or batch.rewards is None or batch.next_observations is None:
            raise ValueError("MASAC 训练需要 actions/rewards/next_observations")
        if batch.rewards.shape[-1] != self.cfg.num_agents:
            raise ValueError("MASAC 需要每个智能体各自的奖励 [B,N]")
        dones = batch.dones if batch.dones is not None else torch.zeros_like(batch.rewards)
        current_input = self._critic_input(batch.observations, batch.actions)
        q1 = torch.stack(
            [critic.forward(current_input).squeeze(-1) for critic in self.critics1], dim=-1
        )
        q2 = torch.stack(
            [critic.forward(current_input).squeeze(-1) for critic in self.critics2], dim=-1
        )
        alpha = self.log_alpha.exp()  # [N]，可广播到 [B,N]。

        # Critic_i 的目标只包含 r_i 和自己的熵项 -alpha_i*log(pi_i)。
        with torch.no_grad():
            next_actions, next_log_prob = self._joint_policy(self.actors, batch.next_observations)
            next_input = self._critic_input(batch.next_observations, next_actions)
            next_q1 = torch.stack(
                [critic.forward(next_input).squeeze(-1) for critic in self.target_critics1],
                dim=-1,
            )
            next_q2 = torch.stack(
                [critic.forward(next_input).squeeze(-1) for critic in self.target_critics2],
                dim=-1,
            )
            target_q = batch.rewards + self.cfg.gamma * (1.0 - dones.float()) * (
                torch.minimum(next_q1, next_q2) - alpha * next_log_prob
            )
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # Actor_i 的 Q_i 对联合动作求值，但只有 a_i 可以收到梯度。
        policy_actions, log_prob = self._joint_policy(self.actors, batch.observations)
        actor_terms = []
        for i in range(self.cfg.num_agents):
            joint_actions = torch.stack(
                [
                    policy_actions[..., j, :] if j == i else policy_actions[..., j, :].detach()
                    for j in range(self.cfg.num_agents)
                ],
                dim=-2,
            )
            policy_input = self._critic_input(batch.observations, joint_actions)
            with self.frozen(self.critics1[i]), self.frozen(self.critics2[i]):
                min_q_i = torch.minimum(
                    self.critics1[i].forward(policy_input),
                    self.critics2[i].forward(policy_input),
                ).squeeze(-1)
            actor_terms.append((alpha[i].detach() * log_prob[..., i] - min_q_i).mean())
        actor_loss = torch.stack(actor_terms).mean()

        target_entropy = (
            self.cfg.target_entropy
            if self.cfg.target_entropy is not None
            else -float(self.cfg.action_dim)
        )
        batch_dims = tuple(range(log_prob.ndim - 1))
        alpha_loss = -(
            self.log_alpha * (log_prob.detach().mean(dim=batch_dims) + target_entropy)
        ).mean()
        total = actor_loss + critic_loss + alpha_loss
        return {
            "loss": total,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "alpha_loss": alpha_loss,
            "alpha": alpha.detach().mean(),
        }

    def update_targets(self) -> None:
        for target, online in zip(self.target_critics1, self.critics1, strict=True):
            self.soft_update(target, online)
        for target, online in zip(self.target_critics2, self.critics2, strict=True):
            self.soft_update(target, online)
