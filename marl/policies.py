"""可复用的策略参数共享拓扑。"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from marl.core import MARLModelOutput
from marl.models.mlp import MLPBackbone
from marl.modules.action_head import DiscreteActionHead
from marl.modules.actor import Actor


class IndependentDiscretePolicy(nn.Module):
    """每个同构智能体拥有独立 MLP Actor 的离散策略。"""

    def __init__(
        self, num_agents: int, observation_dim: int, action_dim: int, hidden_dim: int
    ) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.actors = nn.ModuleList(
            Actor(
                MLPBackbone(observation_dim, output_dim=hidden_dim),
                DiscreteActionHead(hidden_dim, action_dim),
            )
            for _ in range(num_agents)
        )

    def logits(self, observations: Tensor, action_mask: Tensor | None = None) -> Tensor:
        if observations.shape[-2:] != (self.num_agents, self.observation_dim):
            raise ValueError(
                "observations 末两维应为 "
                f"({self.num_agents}, {self.observation_dim})，实际为 {observations.shape[-2:]}"
            )
        if action_mask is not None and action_mask.shape != (
            *observations.shape[:-1],
            self.action_dim,
        ):
            raise ValueError("action_mask 应为 [...,N,A] 且前置维与 observations 一致")
        return torch.stack(
            [
                cast(Actor, actor).discrete_logits(
                    observations[..., index, :],
                    action_mask=(
                        action_mask[..., index, :] if action_mask is not None else None
                    ),
                )
                for index, actor in enumerate(self.actors)
            ],
            dim=-2,
        )

    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
    ) -> MARLModelOutput:
        logits = self.logits(observations, action_mask)
        distribution = Categorical(logits=logits, validate_args=False)
        actions = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return MARLModelOutput(
            actions=actions,
            logits=logits,
            log_prob=distribution.log_prob(actions),
            entropy=distribution.entropy(),
        )

    def evaluate(
        self,
        observations: Tensor,
        actions: Tensor,
        *,
        action_mask: Tensor | None = None,
    ) -> MARLModelOutput:
        logits = self.logits(observations, action_mask)
        if actions.shape != observations.shape[:-1]:
            raise ValueError("离散 actions 应为 [...,N]")
        distribution = Categorical(logits=logits, validate_args=False)
        return MARLModelOutput(
            actions=actions,
            logits=logits,
            log_prob=distribution.log_prob(actions.long()),
            entropy=distribution.entropy(),
        )
