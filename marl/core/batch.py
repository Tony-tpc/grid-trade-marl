from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


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
        if (
            self.action_mask is not None
            and self.action_mask.shape[:-1] != self.observations.shape[:-1]
        ):
            raise ValueError("action_mask 的前置维度必须与 observations 一致")
        if self.rewards is not None:
            self.terminal_flags()

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
            extras={
                key: tensor.to(device, non_blocking=non_blocking)
                for key, tensor in self.extras.items()
            },
        )
