"""多智能体策略拓扑。

``Actor`` 负责一个智能体的 backbone 和 action head；本模块只负责参数共享关系、
逐智能体输入选择、联合形状校验和输出堆叠，不拥有 rollout 或 loss 生命周期。
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from marl.core import MARLModelOutput
from marl.models.mlp import MLPBackbone
from marl.modules.action_head import (
    ActionHeadOutput,
    DeterministicActionHead,
    DiscreteActionHead,
    GaussianActionHead,
)
from marl.modules.actor import Actor


def _stack_scalar(
    outputs: list[ActionHeadOutput], attribute: str
) -> Tensor:
    """把每个 Actor 的 ``[*B]`` 结果堆叠为 ``[*B, N]``。"""

    return torch.stack(
        [cast(Tensor, getattr(output, attribute)) for output in outputs],
        dim=-1,
    )


def _stack_parameter(outputs: list[ActionHeadOutput], name: str) -> Tensor:
    """把每个 Actor 的 ``[*B, A]`` 分布参数堆叠为 ``[*B, N, A]``。"""

    return torch.stack(
        [output.distribution_params[name] for output in outputs],
        dim=-2,
    )


class IndependentDiscretePolicy(nn.Module):
    """每个同构智能体拥有独立离散 Actor 的策略拓扑。"""

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

    def _validate(
        self, observations: Tensor, action_mask: Tensor | None
    ) -> None:
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

    def _mask_for(self, action_mask: Tensor | None, index: int) -> Tensor | None:
        return action_mask[..., index, :] if action_mask is not None else None

    def logits(self, observations: Tensor, action_mask: Tensor | None = None) -> Tensor:
        """返回联合分类参数 ``[*B, N, A]``，不采样动作。"""

        self._validate(observations, action_mask)
        return torch.stack(
            [
                cast(Actor, actor).discrete_logits(
                    observations[..., index, :],
                    action_mask=self._mask_for(action_mask, index),
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
        """对各 Actor 采样并返回 ``actions/log_prob/entropy`` 的逐智能体结果。"""

        self._validate(observations, action_mask)
        outputs = [
            cast(Actor, actor)(
                observations[..., index, :],
                deterministic=deterministic,
                action_mask=self._mask_for(action_mask, index),
            )
            for index, actor in enumerate(self.actors)
        ]
        return MARLModelOutput(
            actions=_stack_scalar(outputs, "actions"),
            logits=_stack_parameter(outputs, "logits"),
            log_prob=_stack_scalar(outputs, "log_prob"),
            entropy=_stack_scalar(outputs, "entropy"),
        )

    def evaluate(
        self,
        observations: Tensor,
        actions: Tensor,
        *,
        action_mask: Tensor | None = None,
    ) -> MARLModelOutput:
        """评估给定联合动作；不会重新采样。"""

        self._validate(observations, action_mask)
        if actions.shape != observations.shape[:-1]:
            raise ValueError("离散 actions 应为 [...,N]")
        outputs = [
            cast(Actor, actor).evaluate_actions(
                observations[..., index, :],
                actions[..., index],
                action_mask=self._mask_for(action_mask, index),
            )
            for index, actor in enumerate(self.actors)
        ]
        return MARLModelOutput(
            actions=_stack_scalar(outputs, "actions"),
            logits=_stack_parameter(outputs, "logits"),
            log_prob=_stack_scalar(outputs, "log_prob"),
            entropy=_stack_scalar(outputs, "entropy"),
        )


class IndependentDeterministicPolicy(nn.Module):
    """每个智能体拥有独立确定性连续 Actor 的策略拓扑。"""

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
        outputs = [
            cast(Actor, actor)(
                observations[..., index, :], deterministic=deterministic
            )
            for index, actor in enumerate(self.actors)
        ]
        return MARLModelOutput(
            actions=torch.stack([output.actions for output in outputs], dim=-2),
            log_prob=_stack_scalar(outputs, "log_prob"),
            entropy=_stack_scalar(outputs, "entropy"),
        )


class IndependentGaussianPolicy(nn.Module):
    """每个智能体拥有独立 tanh-Gaussian Actor 的策略拓扑。"""

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
            cast(Actor, actor)(
                observations[..., index, :], deterministic=deterministic
            )
            for index, actor in enumerate(self.actors)
        ]
        return MARLModelOutput(
            actions=torch.stack([output.actions for output in outputs], dim=-2),
            log_prob=_stack_scalar(outputs, "log_prob"),
            entropy=_stack_scalar(outputs, "entropy"),
        )


class SharedDiscreteQPolicy(nn.Module):
    """同构智能体共享局部 Q 网络的 value-based 执行拓扑。"""

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
        del deterministic
        q_values = self.q_values(observations)
        if action_mask is not None:
            if action_mask.shape != q_values.shape:
                raise ValueError("action_mask 必须与离散 Q 值形状一致")
            q_values = q_values.masked_fill(
                ~action_mask.bool(), torch.finfo(q_values.dtype).min
            )
        return MARLModelOutput(actions=q_values.argmax(dim=-1), logits=q_values)
