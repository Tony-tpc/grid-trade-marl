from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass(slots=True)
class MARLBatch:
    """MARL 各模块之间传递的标准批数据。

    形状约定：
        observations: ``[*B, N, O]``，N 为智能体数，O 为单体观测维度。
        actions: ``[*B, N]``（离散动作）或 ``[*B, N, A]``（连续动作）。
        rewards: ``[*B, N]``。个体收益博弈保留各自的 r_i；只有共享奖励任务
            才把同一个团队奖励广播到 N 个智能体。
        dones: ``[*B, N]``；若环境只返回一个终止标记，也可以传 ``[*B, 1]``。
        state: ``[*B, S]``，仅集中式 critic 需要；去中心化算法可以不传。
        action_mask: ``[*B, N, A]``，True 表示该离散动作可用。

    ``*B`` 可以是单个 batch 维，也可以同时包含时间和 batch 维。这样回放缓冲区、
    rollout 和 recurrent 算法可共享同一容器，而无需为每一种算法重新定义数据类。
    """

    observations: Tensor
    actions: Tensor | None = None
    rewards: Tensor | None = None
    next_observations: Tensor | None = None
    dones: Tensor | None = None
    state: Tensor | None = None
    next_state: Tensor | None = None
    action_mask: Tensor | None = None
    next_action_mask: Tensor | None = None
    # 算法专用张量放在 extras，例如 PPO 的 old_log_prob/advantages/returns。
    extras: dict[str, Tensor] = field(default_factory=dict)

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

    def to(self, device: torch.device | str) -> MARLBatch:
        """返回移动到目标设备的新批次，不在原对象上做隐式修改。"""

        def move(value: Tensor | None) -> Tensor | None:
            return value.to(device) if value is not None else None

        return MARLBatch(
            observations=self.observations.to(device),
            actions=move(self.actions),
            rewards=move(self.rewards),
            next_observations=move(self.next_observations),
            dones=move(self.dones),
            state=move(self.state),
            next_state=move(self.next_state),
            action_mask=move(self.action_mask),
            next_action_mask=move(self.next_action_mask),
            extras={key: tensor.to(device) for key, tensor in self.extras.items()},
        )
