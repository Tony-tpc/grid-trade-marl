"""多智能体策略拓扑。

``Actor`` 负责一个智能体的 backbone 和 action head；本模块只负责参数共享关系、
逐智能体输入选择、联合形状校验和输出堆叠，不拥有 rollout 或 loss 生命周期。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from marl.core import MARLModelOutput
from marl.core.layout import HistoryLayout
from marl.core.recurrent import RecurrentState, map_state, stack_states
from marl.envs.base import EnvironmentSpec
from marl.models import (
    BackboneConfig,
    EncoderConfig,
    GRUBackboneConfig,
    IdentityEncoderConfig,
    LSTMBackboneConfig,
    MLPBackboneConfig,
)
from marl.modules.action_head import (
    ActionHeadOutput,
    DeterministicActionHead,
    DiscreteActionHead,
    GaussianActionHead,
)
from marl.modules.actor import Actor

_DEFAULT_ENCODER = IdentityEncoderConfig()
_DEFAULT_BACKBONE = MLPBackboneConfig()


@runtime_checkable
class PolicyTopology(Protocol):
    """网络能力协议，不是算法基类；外部 nn.Module 可用结构化类型直接接入。"""

    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
    ) -> MARLModelOutput: ...

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...


@runtime_checkable
class StochasticPolicy(PolicyTopology, Protocol):
    """可采样且可评估固定动作的 PPO 策略；连续样本可携带 pre-tanh 值。"""
    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
        hidden_state: RecurrentState = None,
    ) -> MARLModelOutput: ...

    def evaluate(
        self,
        observations: Tensor,
        actions: Tensor,
        *,
        action_mask: Tensor | None = None,
        hidden_state: RecurrentState = None,
        raw_actions: Tensor | None = None,
    ) -> MARLModelOutput: ...


@runtime_checkable
class DiscretePolicy(StochasticPolicy, Protocol):
    def logits(self, observations: Tensor, action_mask: Tensor | None = None) -> Tensor: ...


@runtime_checkable
class AgentLogitsPolicy(Protocol):
    """可选的单智能体 logits 快速路径；外部策略无需强制实现。"""

    def logits_for_agent(
        self,
        observations: Tensor,
        agent_index: int,
        action_mask: Tensor | None = None,
    ) -> Tensor: ...


@runtime_checkable
class LocalQPolicy(PolicyTopology, Protocol):
    def q_values(self, observations: Tensor) -> Tensor: ...


def _stack_scalar(values: list[Tensor | None]) -> Tensor:
    """把各 Actor 的 [*B] 标量结果堆叠为 [*B,N]。"""
    if any(value is None for value in values):
        raise ValueError("Actor 必须返回 log_prob 和 entropy")
    return torch.stack([value for value in values if value is not None], dim=-1)


def _stack_parameter(outputs: list[ActionHeadOutput], name: str) -> Tensor:
    """把每个 Actor 的 ``[*B, A]`` 分布参数堆叠为 ``[*B, N, A]``。"""

    return torch.stack(
        [output.distribution_params[name] for output in outputs],
        dim=-2,
    )


class IndependentDiscretePolicy(nn.Module):
    """每个同构智能体拥有独立离散 Actor 的策略拓扑。"""

    def __init__(
        self,
        num_agents: int,
        observation_dim: int,
        action_dim: int,
        backbone: BackboneConfig | None = None,
        encoder: EncoderConfig = _DEFAULT_ENCODER,
        history: HistoryLayout | None = None,
    ) -> None:
        super().__init__()
        backbone = backbone or MLPBackboneConfig()
        if not isinstance(backbone, (MLPBackboneConfig, GRUBackboneConfig, LSTMBackboneConfig)):
            raise ValueError(
                "本地 actor backbone 只支持 MLP/GRU/LSTM；Transformer 使用 history encoder"
            )
        self.num_agents = num_agents
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.actors = nn.ModuleList()
        for _ in range(num_agents):
            inputs = encoder.build(observation_dim, history)
            network = backbone.build(inputs.output_dim)
            self.actors.append(
                Actor(network, DiscreteActionHead(network.output_dim, action_dim), inputs)
            )
        self.is_recurrent = network.is_recurrent

    def initial_state(self, batch_size: int) -> RecurrentState:
        states = []
        for actor in self.actors:
            assert isinstance(actor, Actor)
            states.append(actor.backbone.initial_state(batch_size))
        return stack_states(states, dim=2)

    def _agent_state(self, state: RecurrentState, index: int, batch: int) -> RecurrentState:
        def select(value: Tensor) -> Tensor:
            if value.ndim != 4 or value.shape[1:3] != (batch, self.num_agents):
                raise ValueError("policy_state 必须为 [K,B,N,H]")
            return value[:, :, index, :].contiguous()

        return map_state(state, select)

    def _validate(self, observations: Tensor, action_mask: Tensor | None) -> None:
        if self.is_recurrent and observations.ndim not in (3, 4):
            raise ValueError("循环 policy 输入必须为 [B,N,O] 或 [B,T,N,O]")
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
        logits = []
        for index, actor in enumerate(self.actors):
            assert isinstance(actor, Actor)
            logits.append(
                actor.discrete_logits(
                    observations[..., index, :],
                    action_mask=self._mask_for(action_mask, index),
                )
            )
        return torch.stack(logits, dim=-2)

    def logits_for_agent(
        self,
        observations: Tensor,
        agent_index: int,
        action_mask: Tensor | None = None,
    ) -> Tensor:
        """只执行目标智能体 Actor，返回 ``[*B,A]`` logits。"""

        self._validate(observations, action_mask)
        if not 0 <= agent_index < self.num_agents:
            raise IndexError("agent_index 超出智能体范围")
        actor = self.actors[agent_index]
        assert isinstance(actor, Actor)
        return actor.discrete_logits(
            observations[..., agent_index, :],
            action_mask=self._mask_for(action_mask, agent_index),
        )

    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
        hidden_state: RecurrentState = None,
    ) -> MARLModelOutput:
        """对各 Actor 采样并返回 ``actions/log_prob/entropy`` 的逐智能体结果。"""

        self._validate(observations, action_mask)
        outputs = [
            actor(
                observations[..., index, :],
                deterministic=deterministic,
                action_mask=self._mask_for(action_mask, index),
                hidden_state=self._agent_state(hidden_state, index, observations.shape[0]),
            )
            for index, actor in enumerate(self.actors)
        ]
        return MARLModelOutput(
            actions=_stack_scalar([output.actions for output in outputs]),
            logits=_stack_parameter(outputs, "logits"),
            log_prob=_stack_scalar([output.log_prob for output in outputs]),
            entropy=_stack_scalar([output.entropy for output in outputs]),
            policy_state=stack_states([output.hidden_state for output in outputs], dim=2),
        )

    def evaluate(
        self,
        observations: Tensor,
        actions: Tensor,
        *,
        action_mask: Tensor | None = None,
        hidden_state: RecurrentState = None,
        raw_actions: Tensor | None = None,
    ) -> MARLModelOutput:
        """评估给定联合动作；不会重新采样。"""

        if raw_actions is not None:
            raise ValueError("离散策略不接受 raw_actions")
        self._validate(observations, action_mask)
        if actions.shape != observations.shape[:-1]:
            raise ValueError("离散 actions 应为 [...,N]")
        outputs = []
        for index, actor in enumerate(self.actors):
            assert isinstance(actor, Actor)
            outputs.append(
                actor.evaluate_actions(
                    observations[..., index, :],
                    actions[..., index],
                    action_mask=self._mask_for(action_mask, index),
                    hidden_state=self._agent_state(hidden_state, index, observations.shape[0]),
                )
            )
        return MARLModelOutput(
            actions=_stack_scalar([output.actions for output in outputs]),
            logits=_stack_parameter(outputs, "logits"),
            log_prob=_stack_scalar([output.log_prob for output in outputs]),
            entropy=_stack_scalar([output.entropy for output in outputs]),
            policy_state=stack_states([output.hidden_state for output in outputs], dim=2),
        )


class IndependentDeterministicPolicy(nn.Module):
    """每个智能体拥有独立确定性 Actor；不把参数共享当作批处理优化。

    act 与高斯拓扑的组装步骤相似，但动作分布、默认确定性和探索机制不同。
    保留这段直观组装，不为少量堆叠代码另建策略基类或万能分布工厂。
    """

    def __init__(
        self,
        num_agents: int,
        observation_dim: int,
        action_dim: int,
        backbone: MLPBackboneConfig = _DEFAULT_BACKBONE,
        encoder: EncoderConfig = _DEFAULT_ENCODER,
        history: HistoryLayout | None = None,
    ) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        if not isinstance(backbone, MLPBackboneConfig):
            raise ValueError("离策略 policy 不支持跨环境步循环 backbone；请使用 history encoder")
        self.actors = nn.ModuleList()
        for _ in range(num_agents):
            inputs = encoder.build(observation_dim, history)
            network = backbone.build(inputs.output_dim)
            self.actors.append(
                Actor(network, DeterministicActionHead(network.output_dim, action_dim), inputs)
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
            actor(observations[..., index, :], deterministic=deterministic)
            for index, actor in enumerate(self.actors)
        ]
        return MARLModelOutput(
            actions=torch.stack([output.actions for output in outputs], dim=-2),
            log_prob=_stack_scalar([output.log_prob for output in outputs]),
            entropy=_stack_scalar([output.entropy for output in outputs]),
        )


class IndependentGaussianPolicy(nn.Module):
    """每个智能体拥有独立 tanh-Gaussian Actor 的策略拓扑。"""

    def __init__(
        self,
        num_agents: int,
        observation_dim: int,
        action_dim: int,
        backbone: BackboneConfig = _DEFAULT_BACKBONE,
        encoder: EncoderConfig = _DEFAULT_ENCODER,
        history: HistoryLayout | None = None,
    ) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        if not isinstance(backbone, (MLPBackboneConfig, GRUBackboneConfig, LSTMBackboneConfig)):
            raise ValueError("连续 actor backbone 只支持 MLP/GRU/LSTM")
        self.actors = nn.ModuleList()
        for _ in range(num_agents):
            inputs = encoder.build(observation_dim, history)
            network = backbone.build(inputs.output_dim)
            self.actors.append(
                Actor(network, GaussianActionHead(network.output_dim, action_dim), inputs)
            )
        self.is_recurrent = network.is_recurrent

    def initial_state(self, batch_size: int) -> RecurrentState:
        """返回各独立 actor 初态 [K,B,N,H]；MLP 返回 None。"""
        states = []
        for actor in self.actors:
            assert isinstance(actor, Actor)
            states.append(actor.backbone.initial_state(batch_size))
        return stack_states(states, dim=2)

    def _agent_state(self, state: RecurrentState, index: int, batch: int) -> RecurrentState:
        """从 [K,B,N,H] 的 h/c 中选择目标 actor，返回连续的 [K,B,H]。"""
        def select(value: Tensor) -> Tensor:
            """验证状态的层/批/智能体维并提取 index，MLP 的 None 由 map_state 保留。"""
            if value.ndim != 4 or value.shape[1:3] != (batch, self.num_agents):
                raise ValueError("policy_state 必须为 [K,B,N,H]")
            return value[:, :, index, :].contiguous()
        return map_state(state, select)

    def _validate(self, observations: Tensor, action_mask: Tensor | None) -> None:
        if action_mask is not None:
            raise ValueError("连续策略不使用 action_mask")
        if observations.shape[-2:] != (self.num_agents, self.observation_dim):
            raise ValueError("observations 末两维与策略 EnvironmentSpec 不一致")
        if self.is_recurrent and observations.ndim not in (3, 4):
            raise ValueError("循环 policy 输入必须为 [B,N,O] 或 [B,T,N,O]")

    @staticmethod
    def _combine(outputs: list[ActionHeadOutput]) -> MARLModelOutput:
        """将 N 个连续 actor 输出堆叠为 [...,N,A]，保留 raw_actions 与独立 h/c 状态。"""
        return MARLModelOutput(
            actions=torch.stack([output.actions for output in outputs], dim=-2),
            log_prob=_stack_scalar([output.log_prob for output in outputs]),
            entropy=_stack_scalar([output.entropy for output in outputs]),
            policy_state=stack_states([output.hidden_state for output in outputs], dim=2),
            extras={"raw_actions": _stack_parameter(outputs, "raw_actions")},
        )

    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
        hidden_state: RecurrentState = None,
    ) -> MARLModelOutput:
        self._validate(observations, action_mask)
        outputs = [
            actor(
                observations[..., index, :], deterministic=deterministic,
                hidden_state=self._agent_state(hidden_state, index, observations.shape[0]),
            )
            for index, actor in enumerate(self.actors)
        ]
        return self._combine(outputs)

    def evaluate(
        self,
        observations: Tensor,
        actions: Tensor,
        *,
        action_mask: Tensor | None = None,
        hidden_state: RecurrentState = None,
        raw_actions: Tensor | None = None,
    ) -> MARLModelOutput:
        """评估 [...,N,A] 固定样本；逐 agent 概率/熵为 [...,N]。"""
        self._validate(observations, action_mask)
        expected = (*observations.shape[:-1], self.action_dim)
        if actions.shape != expected or (raw_actions is not None and raw_actions.shape != expected):
            raise ValueError("连续 actions/raw_actions 应为 [...,N,A]")
        outputs = []
        for index, actor in enumerate(self.actors):
            assert isinstance(actor, Actor)
            outputs.append(actor.evaluate_actions(
                observations[..., index, :], actions[..., index, :],
                hidden_state=self._agent_state(hidden_state, index, observations.shape[0]),
                raw_actions=raw_actions[..., index, :] if raw_actions is not None else None,
            ))
        return self._combine(outputs)


class SharedDiscreteQPolicy(nn.Module):
    """同构智能体共享局部 Q 网络的 value-based 执行拓扑。"""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        backbone: MLPBackboneConfig = _DEFAULT_BACKBONE,
        encoder: EncoderConfig = _DEFAULT_ENCODER,
        history: HistoryLayout | None = None,
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        if not isinstance(backbone, MLPBackboneConfig):
            raise ValueError("离策略 Q policy 不支持循环 backbone；请使用 history encoder")
        self.encoder = encoder.build(observation_dim, history)
        self.backbone = backbone.build(self.encoder.output_dim)
        self.q_head = nn.Linear(self.backbone.output_dim, action_dim)

    def q_values(self, observations: Tensor) -> Tensor:
        if observations.shape[-1] != self.observation_dim:
            raise ValueError("observations 最后一维与策略 EnvironmentSpec 不一致")
        values: Tensor = self.q_head(self.backbone(self.encoder(observations)).features)
        return values

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
            q_values = q_values.masked_fill(~action_mask.bool(), torch.finfo(q_values.dtype).min)
        return MARLModelOutput(actions=q_values.argmax(dim=-1), logits=q_values)


@dataclass(frozen=True, slots=True)
class IndependentDiscreteConfig:
    kind: Literal["independent_discrete"] = "independent_discrete"
    backbone: BackboneConfig = MLPBackboneConfig()
    encoder: EncoderConfig = IdentityEncoderConfig()

    def build(self, spec: EnvironmentSpec) -> IndependentDiscretePolicy:
        return IndependentDiscretePolicy(
            spec.num_agents,
            spec.observation_dim,
            spec.action_dim,
            self.backbone,
            self.encoder,
            spec.observation_history,
        )


@dataclass(frozen=True, slots=True)
class IndependentDeterministicConfig:
    kind: Literal["independent_deterministic"] = "independent_deterministic"
    backbone: MLPBackboneConfig = MLPBackboneConfig()
    encoder: EncoderConfig = IdentityEncoderConfig()

    def build(self, spec: EnvironmentSpec) -> IndependentDeterministicPolicy:
        return IndependentDeterministicPolicy(
            spec.num_agents,
            spec.observation_dim,
            spec.action_dim,
            self.backbone,
            self.encoder,
            spec.observation_history,
        )


@dataclass(frozen=True, slots=True)
class IndependentGaussianConfig:
    kind: Literal["independent_gaussian"] = "independent_gaussian"
    backbone: BackboneConfig = MLPBackboneConfig()
    encoder: EncoderConfig = IdentityEncoderConfig()

    def build(self, spec: EnvironmentSpec) -> IndependentGaussianPolicy:
        return IndependentGaussianPolicy(
            spec.num_agents,
            spec.observation_dim,
            spec.action_dim,
            self.backbone,
            self.encoder,
            spec.observation_history,
        )


@dataclass(frozen=True, slots=True)
class SharedDiscreteQConfig:
    kind: Literal["shared_discrete_q"] = "shared_discrete_q"
    backbone: MLPBackboneConfig = MLPBackboneConfig()
    encoder: EncoderConfig = IdentityEncoderConfig()

    def build(self, spec: EnvironmentSpec) -> SharedDiscreteQPolicy:
        return SharedDiscreteQPolicy(
            spec.observation_dim,
            spec.action_dim,
            self.backbone,
            self.encoder,
            spec.observation_history,
        )
