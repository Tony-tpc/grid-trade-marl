"""一般和博弈版 MAPPO：独立策略与逐智能体集中式价值。"""

from __future__ import annotations

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
from marl.modules.critic import CentralizedCritic


@dataclass(frozen=True, slots=True)
class MAPPOConfig(AlgorithmConfig):
    num_agents: int = 2
    observation_dim: int = 16
    action_dim: int = 5
    state_dim: int | None = None
    hidden_dim: int = 128
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01


class MAPPO(BaseMARLAlgorithm):
    """每个智能体优化自己的 clipped objective 与 return。

    Actor_i 仅输入 o_i；集中式 Critic 的第 i 个输出 V_i(s) 只拟合 return_i。
    batch.extras 必须提供 old_log_prob、advantages、returns，均为 [B,N]。
    这些量由当前策略的 rollout/GAE 生成；不能直接从长期 replay 中采样。
    """

    def __init__(self, config: MAPPOConfig) -> None:
        super().__init__(config)
        self.actors = nn.ModuleList(
            Actor(
                MLPBackbone(config.observation_dim, output_dim=config.hidden_dim),
                DiscreteActionHead(config.hidden_dim, config.action_dim),
            )
            for _ in range(config.num_agents)
        )
        critic_input = config.state_dim or config.num_agents * config.observation_dim
        self.critic = CentralizedCritic(
            MLPBackbone(critic_input, output_dim=config.hidden_dim),
            output_dim=config.num_agents,
        )

    @property
    def cfg(self) -> MAPPOConfig:
        return self.config  # type: ignore[return-value]

    def _policy_logits(self, observations: Tensor, action_mask: Tensor | None) -> Tensor:
        """堆叠独立 Actor 的离散分布参数，得到 [B,N,A]。"""
        return torch.stack(
            [
                cast(Actor, actor).discrete_logits(
                    observations[..., i, :],
                    action_mask=(action_mask[..., i, :] if action_mask is not None else None),
                )
                for i, actor in enumerate(self.actors)
            ],
            dim=-2,
        )

    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
        **kwargs: Tensor,
    ) -> Tensor:
        logits = self._policy_logits(observations, action_mask)
        return (
            logits.argmax(dim=-1)
            if deterministic
            else Categorical(logits=logits, validate_args=False).sample()
        )

    def compute_loss(self, batch: MARLBatch) -> dict[str, Tensor]:
        if batch.actions is None:
            raise ValueError("MAPPO 训练需要 actions")
        required = ("old_log_prob", "advantages", "returns")
        if any(name not in batch.extras for name in required):
            raise ValueError(f"MAPPO batch.extras 必须包含 {required}")
        expected_shape = batch.observations.shape[:-1]  # [B,N] 或 [...,N]
        for name in required:
            if batch.extras[name].shape != expected_shape:
                raise ValueError(f"MAPPO {name} 必须具有逐智能体形状 {expected_shape}")

        # 每个智能体分别计算自己的概率比与 advantage，再对训练样本求平均。
        distribution = Categorical(
            logits=self._policy_logits(batch.observations, batch.action_mask),
            validate_args=False,
        )
        new_log_prob = distribution.log_prob(batch.actions.long())
        ratio = (new_log_prob - batch.extras["old_log_prob"]).exp()
        advantages = batch.extras["advantages"]
        unclipped = ratio * advantages
        clipped = ratio.clamp(1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * advantages
        actor_loss = -torch.minimum(unclipped, clipped).mean()

        critic_input = batch.state if batch.state is not None else batch.observations.flatten(-2)
        values = self.critic.forward(critic_input)  # [B,N]，分别对应 V_1 ... V_N。
        value_loss = F.mse_loss(values, batch.extras["returns"])
        entropy = distribution.entropy().mean()
        total = actor_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy
        return {
            "loss": total,
            "actor_loss": actor_loss,
            "value_loss": value_loss,
            "entropy": entropy,
        }
