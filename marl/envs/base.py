"""所有论文环境都要遵守的接口约定。

这里不引用任何特定环境的类。算法训练器只需要认识 EnvironmentSpec、
EnvironmentStep 和 MARLBatch，因此以后换论文环境时不用改算法源码。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import Tensor

from marl.core.batch import MARLBatch
from marl.core.layout import HistoryLayout, StateLayout


class ActionKind(str, Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class RewardStructure(str, Enum):
    """环境给谁定义目标：个体收益博弈，或全体共享同一团队奖励。"""

    INDIVIDUAL = "individual"
    SHARED = "shared"


@dataclass(frozen=True, slots=True)
class EnvironmentSpec:
    """从环境推导的模型尺寸，而非训练脚本里手填的常量。"""

    num_agents: int
    observation_dim: int
    action_dim: int
    state_dim: int
    action_kind: ActionKind
    horizon: int
    reward_structure: RewardStructure = RewardStructure.INDIVIDUAL
    observation_history: HistoryLayout | None = None
    state_layout: StateLayout | None = None
    adjacency: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        for name in ("num_agents", "observation_dim", "action_dim", "state_dim", "horizon"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if self.observation_history is not None:
            self.observation_history.validate(self.observation_dim)
        if self.state_layout is not None:
            self.state_layout.validate(self.state_dim, self.num_agents)
        if self.adjacency is not None:
            if not isinstance(self.adjacency, tuple) or any(
                not isinstance(row, tuple) for row in self.adjacency
            ):
                raise TypeError("adjacency 必须为不可变 tuple[tuple[float,...],...]")
            adjacency = np.asarray(self.adjacency, dtype=float)
            if (
                adjacency.shape != (self.num_agents, self.num_agents)
                or not np.isfinite(adjacency).all()
                or (adjacency < 0).any()
            ):
                raise ValueError("adjacency 必须为有限非负 [N,N] 静态矩阵")


@dataclass(slots=True)
class EnvironmentStep:
    """适配器返回的统一格式。观测 [N,O]，状态 [S]，奖励 [N]。"""

    observations: np.ndarray
    state: np.ndarray
    rewards: np.ndarray
    terminated: bool
    truncated: bool
    action_mask: np.ndarray | None = None  # 离散动作 [N,A]；True 为合法。
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class EnvironmentAdapter(ABC):
    """训练器依赖的唯一环境接口；具体论文环境继承并实现三个方法。"""

    def validate_step(self, step: EnvironmentStep) -> EnvironmentStep:
        """在环境边界检查形状和数值，便于定位新论文适配器中的数据错误。"""
        spec = self.spec
        if step.observations.shape != (spec.num_agents, spec.observation_dim):
            raise ValueError("observations 形状与 EnvironmentSpec 不一致")
        if step.state.shape != (spec.state_dim,):
            raise ValueError("state 形状与 EnvironmentSpec 不一致")
        if step.rewards.shape != (spec.num_agents,):
            raise ValueError("rewards 形状与 EnvironmentSpec 不一致")
        if not all(
            np.isfinite(array).all() for array in (step.observations, step.state, step.rewards)
        ):
            raise ValueError("环境返回了 NaN 或无穷大")
        if spec.reward_structure == RewardStructure.SHARED and not np.allclose(
            step.rewards, step.rewards[0]
        ):
            raise ValueError("共享奖励环境必须给每个智能体相同奖励")
        if spec.action_kind == ActionKind.DISCRETE:
            if step.action_mask is None or step.action_mask.shape != (
                spec.num_agents,
                spec.action_dim,
            ):
                raise ValueError("离散环境必须返回 [N,A] action_mask")
            if not step.action_mask.any(axis=-1).all():
                raise ValueError("每个智能体至少需要一个合法动作")
        elif step.action_mask is not None:
            raise ValueError("连续环境的 action_mask 应为 None")
        return step

    @property
    @abstractmethod
    def spec(self) -> EnvironmentSpec:
        """返回智能体数、观测/动作/状态尺寸和动作类型。"""

    @abstractmethod
    def reset(self, seed: int | None = None) -> EnvironmentStep:
        """开启一个 episode。reset 的奖励应为全零。"""

    @abstractmethod
    def step(self, actions: np.ndarray) -> EnvironmentStep:
        """执行标准动作 [N] 或 [N,A]，并返回下一时刻数据。"""


@dataclass(slots=True)
class Transition:
    """环境交互得到的一步经验，供 replay buffer 或 rollout 保存。"""

    current: EnvironmentStep
    actions: np.ndarray
    next: EnvironmentStep


def transitions_to_batch(
    transitions: list[Transition], device: torch.device | str = "cpu"
) -> MARLBatch:
    """把多步经验转换成算法统一读取的 MARLBatch。

    ``terminated`` 会停止 Bellman bootstrap；``truncated`` 只表示时间限制边界。
    二者作为 MARLBatch 的一等字段传递，算法不得重新合并成模糊的 done。
    """

    if not transitions:
        raise ValueError("不能从空 transition 列表构造 batch")

    def stack(field_name: str, *, next_step: bool = False) -> Tensor:
        arrays = [
            getattr(item.next if next_step else item.current, field_name) for item in transitions
        ]
        return torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=device)

    masks = [item.current.action_mask for item in transitions]
    next_masks = [item.next.action_mask for item in transitions]
    if any(mask is None for mask in masks) != all(mask is None for mask in masks):
        raise ValueError("一个 batch 内的 action_mask 必须全部存在或全部不存在")
    if any(mask is None for mask in next_masks) != all(mask is None for mask in next_masks):
        raise ValueError("一个 batch 内的 next_action_mask 必须全部存在或全部不存在")
    present_masks = [mask for mask in masks if mask is not None]
    present_next_masks = [mask for mask in next_masks if mask is not None]

    action_array = np.stack([item.actions for item in transitions])
    discrete = action_array.ndim == 2
    action_dtype = torch.long if discrete else torch.float32
    num_agents = transitions[0].current.observations.shape[0]
    terminated = (
        torch.tensor(
            [item.next.terminated for item in transitions], dtype=torch.bool, device=device
        )
        .unsqueeze(-1)
        .expand(-1, num_agents)
    )
    truncated = (
        torch.tensor([item.next.truncated for item in transitions], dtype=torch.bool, device=device)
        .unsqueeze(-1)
        .expand(-1, num_agents)
    )
    return MARLBatch(
        observations=stack("observations"),
        actions=torch.as_tensor(action_array, dtype=action_dtype, device=device),
        rewards=stack("rewards", next_step=True),
        next_observations=stack("observations", next_step=True),
        terminated=terminated,
        truncated=truncated,
        state=stack("state"),
        next_state=stack("state", next_step=True),
        action_mask=(
            torch.as_tensor(np.stack(present_masks), dtype=torch.bool, device=device)
            if present_masks
            else None
        ),
        next_action_mask=(
            torch.as_tensor(np.stack(present_next_masks), dtype=torch.bool, device=device)
            if present_next_masks
            else None
        ),
    )
