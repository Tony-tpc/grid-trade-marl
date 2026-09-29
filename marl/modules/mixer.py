from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from marl.envs.base import EnvironmentSpec
from marl.models import EncoderConfig, GNNBackboneConfig, IdentityEncoderConfig, MLPBackboneConfig
from marl.models.encoder import IdentityEncoder, InputEncoder
from marl.modules.encoding import state_encoder

_DEFAULT_VALUE = MLPBackboneConfig(hidden_dims=(), output_dim=32)


def _hypernetwork(input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]) -> nn.Module:
    """线性输出头；空隐藏层保持原始 Linear 参数与初始化顺序。"""
    if not hidden_dims:
        return nn.Linear(input_dim, output_dim)
    layers: list[nn.Module] = []
    for width in hidden_dims:
        layers.extend((nn.Linear(input_dim, width), nn.ReLU()))
        input_dim = width
    return nn.Sequential(*layers, nn.Linear(input_dim, output_dim))


@runtime_checkable
class MixingNetwork(Protocol):
    def __call__(self, agent_q_values: Tensor, state: Tensor) -> Tensor: ...


class VDNMixer(nn.Module):
    """Value Decomposition Network：直接求和个体 Q 值。"""

    def forward(self, agent_q_values: Tensor, state: Tensor | None = None) -> Tensor:
        return agent_q_values.sum(dim=-1, keepdim=True)


class QMixer(nn.Module):
    """QMIX 单调 mixing network。

    Hypernetwork 由全局 state 生成非负权重，从结构上保证
    ``d Q_tot / d Q_agent >= 0``。
    """

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        mixing_dim: int = 32,
        encoder: InputEncoder | None = None,
        hyper_hidden_dims: tuple[int, ...] = (),
        value_backbone: MLPBackboneConfig = _DEFAULT_VALUE,
    ) -> None:
        super().__init__()
        self.num_agents, self.state_dim, self.mixing_dim = num_agents, state_dim, mixing_dim
        self.encoder = encoder if encoder is not None else IdentityEncoder(state_dim)
        encoded_dim = self.encoder.output_dim
        self.hyper_w1 = _hypernetwork(encoded_dim, num_agents * mixing_dim, hyper_hidden_dims)
        self.hyper_b1 = _hypernetwork(encoded_dim, mixing_dim, hyper_hidden_dims)
        self.hyper_w2 = _hypernetwork(encoded_dim, mixing_dim, hyper_hidden_dims)
        self.value = nn.Sequential(
            *value_backbone.build(encoded_dim).network, nn.Linear(value_backbone.output_dim, 1)
        )

    def forward(self, agent_q_values: Tensor, state: Tensor) -> Tensor:
        if agent_q_values.shape[:-1] != state.shape[:-1]:
            raise ValueError("agent_q_values/state 的批次维必须一致，不能仅展平后大小相同")
        if agent_q_values.shape[-1] != self.num_agents or state.shape[-1] != self.state_dim:
            raise ValueError("agent_q_values/state 最后一维与 QMixer 配置不一致")
        leading = agent_q_values.shape[:-1]
        # 仅合并样本维，不混合智能体；非负 mixing 权重保证逐体 Q 的单调性。
        flat_q = agent_q_values.reshape(-1, 1, self.num_agents)
        flat_state = self.encoder(state).reshape(-1, self.encoder.output_dim)
        w1 = self.hyper_w1(flat_state).abs().view(-1, self.num_agents, self.mixing_dim)
        b1 = self.hyper_b1(flat_state).view(-1, 1, self.mixing_dim)
        hidden = torch.nn.functional.elu(torch.bmm(flat_q, w1) + b1)
        w2 = self.hyper_w2(flat_state).abs().view(-1, self.mixing_dim, 1)
        total = torch.bmm(hidden, w2) + self.value(flat_state).view(-1, 1, 1)
        result: Tensor = total.view(*leading, 1)
        return result


@dataclass(frozen=True, slots=True)
class QMixerConfig:
    kind: Literal["qmix_mixer"] = "qmix_mixer"
    mixing_dim: int = 32
    encoder: EncoderConfig = IdentityEncoderConfig()
    graph: GNNBackboneConfig | None = None
    hyper_hidden_dims: tuple[int, ...] = ()
    value_backbone: MLPBackboneConfig = MLPBackboneConfig(hidden_dims=(), output_dim=32)

    def __post_init__(self) -> None:
        if self.mixing_dim < 1:
            raise ValueError("mixing_dim 必须大于 0")
        if not isinstance(self.hyper_hidden_dims, tuple) or any(
            type(width) is not int or width < 1 for width in self.hyper_hidden_dims
        ):
            raise ValueError("hyper_hidden_dims 必须为正整数 tuple，允许空")
        if not isinstance(self.value_backbone, MLPBackboneConfig):
            raise ValueError("mixer value_backbone 必须为 MLP")

    def build(self, spec: EnvironmentSpec) -> QMixer:
        return QMixer(
            spec.num_agents,
            spec.state_dim,
            self.mixing_dim,
            state_encoder(spec, self.encoder, self.graph),
            self.hyper_hidden_dims,
            self.value_backbone,
        )
