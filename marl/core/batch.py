from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from marl.core.recurrent import RecurrentState, map_state


@dataclass(slots=True)
class MARLBatch:
    """MARL 各模块之间传递的标准批数据。

    形状约定：
        observations: `[*B, N, O]`，N 为智能体数，O 为单体观测维度。
        actions: `[*B, N]`（离散）或 `[*B, N, A]`（连续）。
        rewards: `[*B, N]`，个体收益任务保留各自的 `r_i`。
        terminated: `[*B, N]`，MDP 真正终止，会阻止 bootstrap。
        truncated: `[*B, N]`，时间限制边界，保留当前下一状态 bootstrap。
        state: `[*B, S]`，仅集中式 critic/mixer 需要。
        action_mask: `[*B, N, A]`，True 表示离散动作可用。

    `*B` 可同时包含时间、并行环境和 batch 维。算法不得把 `terminated` 与
    `truncated` 合并成一个模糊 done 后用于 Bellman/GAE bootstrap。
    """

    observations: Tensor
    actions: Tensor | None = None
    rewards: Tensor | None = None
    next_observations: Tensor | None = None
    terminated: Tensor | None = None
    truncated: Tensor | None = None
    state: Tensor | None = None
    next_state: Tensor | None = None
    action_mask: Tensor | None = None
    next_action_mask: Tensor | None = None
    extras: dict[str, Tensor] = field(default_factory=dict)
    # 仅序列训练使用；状态是每条序列开始前的 [K,B,(N),H]，不是每个时间步的状态。
    policy_state: RecurrentState = None
    value_state: RecurrentState = None
    sequence_mask: Tensor | None = None
    valid_sample_count: int | None = None

    def valid(self, value: Tensor) -> Tensor:
        """完整序列前向之后选择有效位置，保留 N，不对 padding 计算目标。"""
        return value[self.sequence_mask] if self.sequence_mask is not None else value

    def terminal_flags(self) -> tuple[Tensor, Tensor]:
        """返回逐智能体终止标记，缺失时使用全 False。"""

        reference = self.rewards
        if reference is None:
            reference = torch.zeros(
                self.observations.shape[:-1],
                dtype=torch.float32,
                device=self.observations.device,
            )
        terminated = (
            self.terminated
            if self.terminated is not None
            else torch.zeros_like(reference, dtype=torch.bool)
        )
        truncated = (
            self.truncated
            if self.truncated is not None
            else torch.zeros_like(reference, dtype=torch.bool)
        )
        if terminated.shape != reference.shape or truncated.shape != reference.shape:
            raise ValueError("terminated/truncated 必须与逐智能体 rewards 形状一致")
        return terminated.bool(), truncated.bool()

    def validate(self, num_agents: int, observation_dim: int) -> None:
        """尽早检查公共维度，避免错误在损失计算深处才暴露。"""

        if self.observations.ndim < 2:
            raise ValueError("observations 至少需要 [agent, observation] 两个维度")
        if self.observations.shape[-2] != num_agents:
            raise ValueError(f"期望 {num_agents} 个智能体，实际为 {self.observations.shape[-2]}")
        if self.observations.shape[-1] != observation_dim:
            raise ValueError(
                f"期望 observation_dim={observation_dim}，实际为 {self.observations.shape[-1]}"
            )
        # 只检查 shape 元数据，不对 GPU tensor 做 .item()/isfinite 等同步。
        # 动作类别、合法范围和 extras 的数学含义仍由具体算法验证。
        if self.sequence_mask is not None:
            if self.observations.ndim != 4 or (
                self.sequence_mask.shape != self.observations.shape[:2]
                or self.sequence_mask.dtype != torch.bool
                or self.sequence_mask.device != self.observations.device
            ):
                raise ValueError("sequence_mask 必须是与 [B,T,N,O] 对应的 bool [B,T]")
            if self.valid_sample_count is None or self.valid_sample_count < 1:
                raise ValueError("序列 batch 必须声明正的 valid_sample_count")
        agent_shape = self.observations.shape[:-1]
        leading_shape = self.observations.shape[:-2]
        for name in ("rewards", "terminated", "truncated"):
            value = getattr(self, name)
            if value is not None and value.shape != agent_shape:
                raise ValueError(f"{name} 必须具有逐智能体形状 {tuple(agent_shape)}")
        if self.next_observations is not None and (
            self.next_observations.shape != self.observations.shape
        ):
            raise ValueError("next_observations 必须与 observations 形状一致")
        if self.actions is not None and not (
            self.actions.shape == agent_shape
            or (
                self.actions.ndim == self.observations.ndim
                and self.actions.shape[:-1] == agent_shape
            )
        ):
            raise ValueError("actions 必须为 [...,N] 或 [...,N,A]")
        for name in ("state", "next_state"):
            value = getattr(self, name)
            if value is not None and (
                value.ndim != self.observations.ndim - 1 or value.shape[:-1] != leading_shape
            ):
                raise ValueError(f"{name} 的批次维必须与 observations 一致")
        for name in ("action_mask", "next_action_mask"):
            value = getattr(self, name)
            if value is not None and value.shape[:-1] != agent_shape:
                raise ValueError(f"{name} 的前置维度必须与 observations 一致")

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> MARLBatch:
        """返回移动到目标设备的新批次，不在原对象上做隐式修改。"""

        def move(value: Tensor | None) -> Tensor | None:
            return value.to(device, non_blocking=non_blocking) if value is not None else None

        return MARLBatch(
            observations=self.observations.to(device, non_blocking=non_blocking),
            actions=move(self.actions),
            rewards=move(self.rewards),
            next_observations=move(self.next_observations),
            terminated=move(self.terminated),
            truncated=move(self.truncated),
            state=move(self.state),
            next_state=move(self.next_state),
            action_mask=move(self.action_mask),
            next_action_mask=move(self.next_action_mask),
            policy_state=map_state(
                self.policy_state, lambda t: t.to(device, non_blocking=non_blocking)
            ),
            value_state=map_state(
                self.value_state, lambda t: t.to(device, non_blocking=non_blocking)
            ),
            sequence_mask=move(self.sequence_mask),
            valid_sample_count=self.valid_sample_count,
            extras={
                key: tensor.to(device, non_blocking=non_blocking)
                for key, tensor in self.extras.items()
            },
        )

    def pin_memory(self) -> MARLBatch:
        """返回 pinned-CPU 批次，供 CUDA non-blocking 整批传输。"""

        def pin(value: Tensor | None) -> Tensor | None:
            return value.pin_memory() if value is not None else None

        return MARLBatch(
            observations=self.observations.pin_memory(),
            actions=pin(self.actions),
            rewards=pin(self.rewards),
            next_observations=pin(self.next_observations),
            terminated=pin(self.terminated),
            truncated=pin(self.truncated),
            state=pin(self.state),
            next_state=pin(self.next_state),
            action_mask=pin(self.action_mask),
            next_action_mask=pin(self.next_action_mask),
            extras={key: tensor.pin_memory() for key, tensor in self.extras.items()},
            policy_state=map_state(
                self.policy_state, lambda t: t.pin_memory() if not t.is_cuda else t
            ),
            value_state=map_state(
                self.value_state, lambda t: t.pin_memory() if not t.is_cuda else t
            ),
            sequence_mask=pin(self.sequence_mask),
            valid_sample_count=self.valid_sample_count,
        )
