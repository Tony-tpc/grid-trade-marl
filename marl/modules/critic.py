from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from marl.models.base import BaseBackbone
from marl.models.mlp import MLPBackbone


class ValueCritic(nn.Module):
    """对每个输入位置输出标量 V/Q，适用于独立 critic。"""

    def __init__(self, backbone: BaseBackbone) -> None:
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Linear(backbone.output_dim, 1)

    def forward(self, inputs: Tensor, **backbone_kwargs: Tensor) -> Tensor:
        features = self.backbone.forward(inputs, **backbone_kwargs).features
        return cast(Tensor, self.value_head(features).squeeze(-1))


class CentralizedCritic(nn.Module):
    """CTDE 的集中式 critic。

    调用者负责把 global state、联合观测或联合动作拼成最后一维；这种显式约定避免
    critic 猜测各算法不同的数据布局。
    """

    def __init__(self, backbone: BaseBackbone, output_dim: int = 1) -> None:
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Linear(backbone.output_dim, output_dim)  # 输出Q值

    def forward(self, centralized_input: Tensor, **backbone_kwargs: Tensor) -> Tensor:
        """
        centralized_input :
            batch{
                [o1,o2,o3,...,a1,a2,a3,...],
                [o1,o2,o3,...,a1,a2,a3,...]
                ...
            }

        centralized_input -> backbone -> features -> value_head -> Q_value
        """
        features = self.backbone.forward(centralized_input, **backbone_kwargs).features
        return cast(Tensor, self.value_head(features))


class IndependentCentralizedCritics(nn.Module):
    """参数独立的逐智能体集中式 Q 网络。"""

    def __init__(self, num_agents: int, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.critics = nn.ModuleList(
            CentralizedCritic(MLPBackbone(input_dim, output_dim=hidden_dim))
            for _ in range(num_agents)
        )

    def forward(self, centralized_input: Tensor) -> Tensor:
        return torch.stack(
            [critic(centralized_input).squeeze(-1) for critic in self.critics],
            dim=-1,
        )


class TwinIndependentCentralizedCritics(nn.Module):
    """MASAC 使用的两组独立集中式 Q 网络。"""

    def __init__(self, num_agents: int, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.first = IndependentCentralizedCritics(num_agents, input_dim, hidden_dim)
        self.second = IndependentCentralizedCritics(num_agents, input_dim, hidden_dim)

    def forward(self, centralized_input: Tensor) -> tuple[Tensor, Tensor]:
        return self.first(centralized_input), self.second(centralized_input)


class AttentionCritic(nn.Module):
    """共享注意力上下文和逐智能体离散候选 Q。"""

    def __init__(
        self,
        num_agents: int,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int,
        attention_heads: int,
    ) -> None:
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim 必须能被 attention_heads 整除")
        self.action_dim = action_dim
        self.own_projection = nn.Linear(observation_dim, hidden_dim)
        self.other_projection = nn.Linear(observation_dim + action_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, attention_heads, batch_first=True
        )
        self.register_buffer(
            "_self_attention_mask",
            torch.eye(num_agents, dtype=torch.bool),
            persistent=False,
        )
        self.q_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
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
            context, _ = self.attention(
                own,
                others,
                others,
                attn_mask=self._self_attention_mask,
                need_weights=False,
            )
        q_values = self.q_head(torch.cat((own, context), dim=-1))
        return cast(Tensor, q_values.reshape(*leading, agents, self.action_dim))
