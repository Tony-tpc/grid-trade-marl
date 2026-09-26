"""可复用的策略参数共享拓扑。"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from marl.core import MARLModelOutput
from marl.models.mlp import MLPBackbone
from marl.modules.action_head import (
    DeterministicActionHead,
    DiscreteActionHead,
    GaussianActionHead,
)
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


class IndependentDeterministicPolicy(nn.Module):
    """每个智能体拥有独立确定性连续 Actor。"""

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
                DeterministicActionHead(hidden_dim, action_dim),
            )
            for _ in range(num_agents)
        )

    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool = True,
        action_mask: Tensor | None = None,
    ) -> MARLModelOutput:
        if action_mask is not None:
            raise ValueError("连续策略不使用 action_mask")
        if observations.shape[-2:] != (self.num_agents, self.observation_dim):
            raise ValueError("observations 末两维与策略 EnvironmentSpec 不一致")
        actions = torch.stack(
            [
                actor(observations[..., index, :], deterministic=True).actions
                for index, actor in enumerate(self.actors)
            ],
            dim=-2,
        )
        zeros = torch.zeros_like(actions[..., 0])
        return MARLModelOutput(actions=actions, log_prob=zeros, entropy=zeros)


class IndependentGaussianPolicy(nn.Module):
    """每个智能体拥有独立 tanh-Gaussian Actor。"""

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
                GaussianActionHead(hidden_dim, action_dim),
            )
            for _ in range(num_agents)
        )

    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
    ) -> MARLModelOutput:
        if action_mask is not None:
            raise ValueError("连续策略不使用 action_mask")
        if observations.shape[-2:] != (self.num_agents, self.observation_dim):
            raise ValueError("observations 末两维与策略 EnvironmentSpec 不一致")
        outputs = [
            actor(observations[..., index, :], deterministic=deterministic)
            for index, actor in enumerate(self.actors)
        ]
        return MARLModelOutput(
            actions=torch.stack([output.actions for output in outputs], dim=-2),
            log_prob=torch.stack([output.log_prob for output in outputs], dim=-1),
            entropy=torch.stack([output.entropy for output in outputs], dim=-1),
        )


class SharedDiscreteQPolicy(nn.Module):
    """同构智能体共享局部 Q 网络的离散执行策略。"""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.backbone = MLPBackbone(observation_dim, output_dim=hidden_dim)
        self.q_head = nn.Linear(hidden_dim, action_dim)

    def q_values(self, observations: Tensor) -> Tensor:
        if observations.shape[-1] != self.observation_dim:
            raise ValueError("observations 最后一维与策略 EnvironmentSpec 不一致")
        return cast(Tensor, self.q_head(self.backbone(observations).features))

    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool = True,
        action_mask: Tensor | None = None,
    ) -> MARLModelOutput:
        q_values = self.q_values(observations)
        if action_mask is not None:
            if action_mask.shape != q_values.shape:
                raise ValueError("action_mask 必须与离散 Q 值形状一致")
            q_values = q_values.masked_fill(
                ~action_mask.bool(), torch.finfo(q_values.dtype).min
            )
        return MARLModelOutput(actions=q_values.argmax(dim=-1), logits=q_values)
