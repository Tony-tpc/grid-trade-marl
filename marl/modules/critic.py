from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol, overload, runtime_checkable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from marl.core.recurrent import RecurrentState
from marl.envs.base import EnvironmentSpec
from marl.models import (
    BackboneConfig,
    EncoderConfig,
    GNNBackboneConfig,
    GRUBackboneConfig,
    IdentityEncoderConfig,
    LSTMBackboneConfig,
    MLPBackboneConfig,
)
from marl.models.base import BaseBackbone
from marl.models.encoder import IdentityEncoder, InputEncoder
from marl.modules.encoding import JointActionEncoder, ObservationEncoder, state_encoder

_DEFAULT_ENCODER = IdentityEncoderConfig()
_DEFAULT_BACKBONE = MLPBackboneConfig()
_DEFAULT_EMBEDDING = MLPBackboneConfig(hidden_dims=())


def centralized_critic_input(observations: Tensor, actions: Tensor) -> Tensor:
    """把 ``[..., N, O]`` 与 ``[..., N, A]`` 拼成 ``[..., N*O + N*A]``。"""

    if observations.ndim < 2 or actions.ndim < 2:
        raise ValueError("observations 和 actions 至少需要智能体维与特征维")
    if observations.shape[:-2] != actions.shape[:-2]:
        raise ValueError("observations 和 actions 的批次维必须一致")
    if observations.shape[-2] != actions.shape[-2]:
        raise ValueError("observations 和 actions 的智能体数量必须一致")
    return torch.cat((observations.flatten(-2), actions.flatten(-2)), dim=-1)


@runtime_checkable
class ValueNetwork(Protocol):
    def __call__(self, inputs: Tensor) -> Tensor: ...
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...


@runtime_checkable
class RecurrentValueNetwork(Protocol):
    """可选的状态返回能力；外部无状态 nn.Sequential 无需实现。"""

    is_recurrent: bool

    def __call__(
        self,
        inputs: Tensor,
        *,
        hidden_state: RecurrentState = None,
        return_state: Literal[True],
    ) -> tuple[Tensor, RecurrentState]: ...


@runtime_checkable
class AttentionQNetwork(Protocol):
    def __call__(self, observations: Tensor, actions: Tensor) -> Tensor: ...
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...


@runtime_checkable
class QEnsemble(ValueNetwork, Protocol):
    @property
    def critics(self) -> nn.ModuleList: ...


@runtime_checkable
class TwinQEnsemble(Protocol):
    @property
    def first(self) -> QEnsemble: ...

    @property
    def second(self) -> QEnsemble: ...

    def __call__(self, inputs: Tensor) -> tuple[Tensor, Tensor]: ...
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...


class ValueCritic(nn.Module):
    """对每个输入位置输出标量 V/Q，适用于独立 critic。"""

    def __init__(self, backbone: BaseBackbone) -> None:
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Linear(backbone.output_dim, 1)

    def forward(self, inputs: Tensor, **backbone_kwargs: Tensor) -> Tensor:
        features = self.backbone(inputs, **backbone_kwargs).features
        values: Tensor = self.value_head(features).squeeze(-1)
        return values


class CentralizedCritic(nn.Module):
    """CTDE 的集中式 critic。

    调用者负责把 global state、联合观测或联合动作拼成最后一维；这种显式约定避免
    critic 猜测各算法不同的数据布局。
    """

    def __init__(
        self, backbone: BaseBackbone, output_dim: int = 1, encoder: InputEncoder | None = None
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else IdentityEncoder(backbone.input_dim)
        self.backbone = backbone
        self.is_recurrent = backbone.is_recurrent
        # 输入是 state 时可表示 V，输入含联合动作时可表示 Q；网络不猜测算法。
        self.value_head = nn.Linear(backbone.output_dim, output_dim)

    def initial_state(self, batch_size: int) -> RecurrentState:
        return self.backbone.initial_state(batch_size)

    @overload
    def forward(
        self,
        centralized_input: Tensor,
        *,
        hidden_state: RecurrentState = None,
        return_state: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        centralized_input: Tensor,
        *,
        hidden_state: RecurrentState = None,
        return_state: Literal[True],
    ) -> tuple[Tensor, RecurrentState]: ...

    def forward(
        self,
        centralized_input: Tensor,
        *,
        hidden_state: RecurrentState = None,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, RecurrentState]:
        """单步 [B,S] 或连续序列 [B,T,S]，value 最后一维始终保留 N。"""
        centralized_input = self.encoder(centralized_input)
        single_step = self.is_recurrent and centralized_input.ndim == 2
        encoded = self.backbone(
            centralized_input.unsqueeze(1) if single_step else centralized_input,
            hidden_state=hidden_state,
        )
        values: Tensor = self.value_head(encoded.features)
        if single_step:
            values = values.squeeze(1)
        return (values, encoded.hidden_state) if return_state else values


class IndependentCentralizedCritics(nn.Module):
    """参数独立的逐智能体集中式 Q 网络。"""

    def __init__(
        self,
        spec: EnvironmentSpec,
        backbone: MLPBackboneConfig = _DEFAULT_BACKBONE,
        encoder: EncoderConfig = _DEFAULT_ENCODER,
        graph: GNNBackboneConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, MLPBackboneConfig):
            raise ValueError("离策略 critic 不支持循环 backbone；请使用 history encoder")
        self.critics = nn.ModuleList()
        for _ in range(spec.num_agents):
            inputs = JointActionEncoder(spec, encoder, graph)
            self.critics.append(
                CentralizedCritic(backbone.build(inputs.output_dim), encoder=inputs)
            )

    def forward(self, centralized_input: Tensor) -> Tensor:
        return torch.stack(
            [critic(centralized_input).squeeze(-1) for critic in self.critics],
            dim=-1,
        )


class TwinIndependentCentralizedCritics(nn.Module):
    """MASAC 使用的两组独立集中式 Q 网络。"""

    def __init__(
        self,
        spec: EnvironmentSpec,
        backbone: MLPBackboneConfig = _DEFAULT_BACKBONE,
        encoder: EncoderConfig = _DEFAULT_ENCODER,
        graph: GNNBackboneConfig | None = None,
    ) -> None:
        super().__init__()
        self.first = IndependentCentralizedCritics(spec, backbone, encoder, graph)
        self.second = IndependentCentralizedCritics(spec, backbone, encoder, graph)

    def forward(self, centralized_input: Tensor) -> tuple[Tensor, Tensor]:
        return self.first(centralized_input), self.second(centralized_input)


class AttentionCritic(nn.Module):
    """MAAC 的逐智能体离散候选 Q 网络。

    每个智能体拥有独立的 observation encoder、state-action encoder 与 Q head；
    只有 ``nn.MultiheadAttention`` 内的 query/key/value 投影在智能体间共享。
    输入为 observations ``[*B,N,O]`` 与离散 actions ``[*B,N]``，输出为
    每个智能体所有候选动作的 Q 值 ``[*B,N,A]``。
    """

    def __init__(
        self,
        spec: EnvironmentSpec,
        embedding: MLPBackboneConfig = _DEFAULT_EMBEDDING,
        backbone: MLPBackboneConfig = _DEFAULT_EMBEDDING,
        encoder: EncoderConfig = _DEFAULT_ENCODER,
        attention_heads: int = 4,
        graph: GNNBackboneConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(embedding, MLPBackboneConfig) or not isinstance(
            backbone, MLPBackboneConfig
        ):
            raise ValueError("MAAC embedding/Q backbone 必须为 MLP；时序编码使用 encoder")
        hidden_dim = embedding.output_dim
        num_agents, observation_dim, action_dim = (
            spec.num_agents,
            spec.observation_dim,
            spec.action_dim,
        )
        if attention_heads < 1 or hidden_dim % attention_heads:
            raise ValueError("hidden_dim 必须能被 attention_heads 整除")
        self.num_agents = num_agents
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.encoder = ObservationEncoder(spec, encoder, graph)
        encoded_dim = self.encoder.output_dim
        self.own_encoders = nn.ModuleList(
            embedding.build(encoded_dim).network for _ in range(num_agents)
        )
        self.state_action_encoders = nn.ModuleList(
            embedding.build(encoded_dim + action_dim).network for _ in range(num_agents)
        )
        self.attention = nn.MultiheadAttention(hidden_dim, attention_heads, batch_first=True)
        self.register_buffer(
            "_self_attention_mask",
            torch.eye(num_agents, dtype=torch.bool),
            persistent=False,
        )
        self.q_heads = nn.ModuleList(
            nn.Sequential(
                *backbone.build(2 * hidden_dim).network,
                nn.Linear(backbone.output_dim, action_dim),
            )
            for _ in range(num_agents)
        )

    def forward(self, observations: Tensor, actions: Tensor) -> Tensor:
        expected_observation_shape = (self.num_agents, self.observation_dim)
        if observations.shape[-2:] != expected_observation_shape:
            raise ValueError(
                "observations 末两维应为 "
                f"{expected_observation_shape}，实际为 {tuple(observations.shape[-2:])}"
            )
        if actions.shape != observations.shape[:-1]:
            raise ValueError("离散 actions 应为 [...,N] 且前置维与 observations 一致")
        leading = observations.shape[:-2]
        observations = self.encoder(observations)
        own = torch.stack(
            [
                encoder(observations[..., index, :])
                for index, encoder in enumerate(self.own_encoders)
            ],
            dim=-2,
        )
        one_hot = F.one_hot(actions.long(), self.action_dim).to(observations.dtype)
        state_actions = torch.cat((observations, one_hot), dim=-1)
        others = torch.stack(
            [
                encoder(state_actions[..., index, :])
                for index, encoder in enumerate(self.state_action_encoders)
            ],
            dim=-2,
        )
        flat_own = own.reshape(-1, self.num_agents, self.hidden_dim)
        flat_others = others.reshape(-1, self.num_agents, self.hidden_dim)
        if self.num_agents == 1:
            context = torch.zeros_like(own)
        else:
            context, _ = self.attention(
                flat_own,
                flat_others,
                flat_others,
                attn_mask=self._self_attention_mask,
                need_weights=False,
            )
            context = context.reshape(*leading, self.num_agents, self.hidden_dim)
        q_input = torch.cat((own, context), dim=-1)
        q_values = torch.stack(
            [head(q_input[..., index, :]) for index, head in enumerate(self.q_heads)],
            dim=-2,
        )
        result: Tensor = q_values.reshape(*leading, self.num_agents, self.action_dim)
        return result


@dataclass(frozen=True, slots=True)
class CentralizedValueConfig:
    kind: Literal["centralized_value"] = "centralized_value"
    backbone: BackboneConfig = MLPBackboneConfig()
    encoder: EncoderConfig = IdentityEncoderConfig()
    graph: GNNBackboneConfig | None = None

    def build(self, spec: EnvironmentSpec) -> CentralizedCritic:
        if not isinstance(
            self.backbone, (MLPBackboneConfig, GRUBackboneConfig, LSTMBackboneConfig)
        ):
            raise ValueError("value backbone 只支持 MLP/GRU/LSTM；历史/图使用 encoder/graph")
        inputs = state_encoder(spec, self.encoder, self.graph)
        return CentralizedCritic(
            self.backbone.build(inputs.output_dim),
            output_dim=spec.num_agents,
            encoder=inputs,
        )


@dataclass(frozen=True, slots=True)
class AttentionQConfig:
    kind: Literal["attention_q"] = "attention_q"
    encoder: EncoderConfig = IdentityEncoderConfig()
    embedding: MLPBackboneConfig = MLPBackboneConfig(hidden_dims=())
    backbone: MLPBackboneConfig = MLPBackboneConfig(hidden_dims=())
    graph: GNNBackboneConfig | None = None
    attention_heads: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.embedding, MLPBackboneConfig) or not isinstance(
            self.backbone, MLPBackboneConfig
        ):
            raise ValueError("MAAC embedding/backbone 必须为 MLP")
        if self.attention_heads < 1 or self.embedding.output_dim % self.attention_heads:
            raise ValueError("attention_heads 必须为正且整除 embedding.output_dim")

    def build(self, spec: EnvironmentSpec) -> AttentionCritic:
        return AttentionCritic(
            spec,
            self.embedding,
            self.backbone,
            self.encoder,
            self.attention_heads,
            self.graph,
        )


@dataclass(frozen=True, slots=True)
class IndependentQConfig:
    kind: Literal["independent_centralized_q"] = "independent_centralized_q"
    encoder: EncoderConfig = IdentityEncoderConfig()
    backbone: MLPBackboneConfig = MLPBackboneConfig()
    graph: GNNBackboneConfig | None = None

    def build(self, spec: EnvironmentSpec) -> IndependentCentralizedCritics:
        return IndependentCentralizedCritics(
            spec,
            self.backbone,
            self.encoder,
            self.graph,
        )


@dataclass(frozen=True, slots=True)
class TwinQConfig:
    kind: Literal["twin_independent_centralized_q"] = "twin_independent_centralized_q"
    encoder: EncoderConfig = IdentityEncoderConfig()
    backbone: MLPBackboneConfig = MLPBackboneConfig()
    graph: GNNBackboneConfig | None = None

    def build(self, spec: EnvironmentSpec) -> TwinIndependentCentralizedCritics:
        return TwinIndependentCentralizedCritics(
            spec,
            self.backbone,
            self.encoder,
            self.graph,
        )
