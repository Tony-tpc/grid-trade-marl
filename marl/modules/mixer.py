from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn


class VDNMixer(nn.Module):
    """Value Decomposition Network：直接求和个体 Q 值。"""

    def forward(self, agent_q_values: Tensor, state: Tensor | None = None) -> Tensor:
        return agent_q_values.sum(dim=-1, keepdim=True)


class QMixer(nn.Module):
    """QMIX 单调 mixing network。

    Hypernetwork 由全局 state 生成非负权重，从结构上保证
    ``d Q_tot / d Q_agent >= 0``。
    """

    def __init__(self, num_agents: int, state_dim: int, mixing_dim: int = 32) -> None:
        super().__init__()
        self.num_agents, self.state_dim, self.mixing_dim = num_agents, state_dim, mixing_dim
        self.hyper_w1 = nn.Linear(state_dim, num_agents * mixing_dim)
        self.hyper_b1 = nn.Linear(state_dim, mixing_dim)
        self.hyper_w2 = nn.Linear(state_dim, mixing_dim)
        self.value = nn.Sequential(
            nn.Linear(state_dim, mixing_dim), nn.ReLU(), nn.Linear(mixing_dim, 1)
        )

    def forward(self, agent_q_values: Tensor, state: Tensor) -> Tensor:
        if agent_q_values.shape[-1] != self.num_agents or state.shape[-1] != self.state_dim:
            raise ValueError("agent_q_values/state 最后一维与 QMixer 配置不一致")
        leading = agent_q_values.shape[:-1]
        flat_q = agent_q_values.reshape(-1, 1, self.num_agents)
        flat_state = state.reshape(-1, self.state_dim)
        w1 = self.hyper_w1(flat_state).abs().view(-1, self.num_agents, self.mixing_dim)
        b1 = self.hyper_b1(flat_state).view(-1, 1, self.mixing_dim)
        hidden = torch.nn.functional.elu(torch.bmm(flat_q, w1) + b1)
        w2 = self.hyper_w2(flat_state).abs().view(-1, self.mixing_dim, 1)
        total = torch.bmm(hidden, w2) + self.value(flat_state).view(-1, 1, 1)
        return cast(Tensor, total.view(*leading, 1))
