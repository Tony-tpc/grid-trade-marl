"""Device placement, vectorized environment adapters and replay storage.

This layer consumes the public environment, batch and algorithm interfaces. It
does not change the behavior or network layout of any particular algorithm.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from marl.core.batch import MARLBatch
from marl.envs.base import (
    ActionKind,
    EnvironmentAdapter,
    EnvironmentSpec,
    EnvironmentStep,
    Transition,
)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "当前 PyTorch 未提供 CUDA；请安装 CUDA 版 PyTorch 后再使用 --device cuda"
        )
    return device


class SyncVectorEnv:
    """Run multiple adapters together and present one policy inference batch."""

    def __init__(self, adapters: Sequence[EnvironmentAdapter]) -> None:
        if not adapters:
            raise ValueError("至少需要一个环境适配器")
        self.adapters = tuple(adapters)
        self.spec = self.adapters[0].spec
        if any(adapter.spec != self.spec for adapter in self.adapters[1:]):
            raise ValueError("并行环境必须具有相同的 EnvironmentSpec")

    def reset(self, seeds: Sequence[int]) -> list[EnvironmentStep]:
        if len(seeds) != len(self.adapters):
            raise ValueError("seeds 数量必须与环境数量一致")
        return [adapter.reset(seed) for adapter, seed in zip(self.adapters, seeds, strict=True)]

    def step(self, actions: np.ndarray) -> list[EnvironmentStep]:
        if len(actions) != len(self.adapters):
            raise ValueError("动作批量的首维必须等于环境数量")
        return [
            adapter.step(action)
            for adapter, action in zip(self.adapters, actions, strict=True)
        ]


class TensorReplayBuffer:
    """Preallocated CPU ring buffer for fixed-shape environment transitions."""

    def __init__(self, spec: EnvironmentSpec, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity 必须大于零")
        self.spec = spec
        self.capacity = capacity
        self.size = 0
        self.position = 0
        def allocate(*shape: int, dtype: torch.dtype) -> Tensor:
            return torch.empty((capacity, *shape), dtype=dtype)
        n, o, s = spec.num_agents, spec.observation_dim, spec.state_dim
        action_shape = (n,) if spec.action_kind == ActionKind.DISCRETE else (n, spec.action_dim)
        action_dtype = torch.long if spec.action_kind == ActionKind.DISCRETE else torch.float32
        self.observations = allocate(n, o, dtype=torch.float32)
        self.next_observations = allocate(n, o, dtype=torch.float32)
        self.states = allocate(s, dtype=torch.float32)
        self.next_states = allocate(s, dtype=torch.float32)
        self.actions = allocate(*action_shape, dtype=action_dtype)
        self.rewards = allocate(n, dtype=torch.float32)
        self.terminated = allocate(1, dtype=torch.bool)
        self.truncated = allocate(1, dtype=torch.bool)
        self.action_masks: Tensor | None = None
        self.next_action_masks: Tensor | None = None
        if spec.action_kind == ActionKind.DISCRETE:
            self.action_masks = allocate(n, spec.action_dim, dtype=torch.bool)
            self.next_action_masks = allocate(n, spec.action_dim, dtype=torch.bool)

    def __len__(self) -> int:
        return self.size

    def add(self, transition: Transition) -> None:
        index = self.position
        current, next_step = transition.current, transition.next
        for target, source in (
            (self.observations, current.observations),
            (self.next_observations, next_step.observations),
            (self.states, current.state),
            (self.next_states, next_step.state),
            (self.actions, transition.actions),
            (self.rewards, next_step.rewards),
        ):
            target[index].copy_(torch.as_tensor(source, dtype=target.dtype))
        self.terminated[index, 0] = next_step.terminated
        self.truncated[index, 0] = next_step.truncated
        if self.action_masks is not None and self.next_action_masks is not None:
            if current.action_mask is None or next_step.action_mask is None:
                raise ValueError("离散动作经验必须包含当前及下一步 action_mask")
            self.action_masks[index].copy_(torch.as_tensor(current.action_mask))
            self.next_action_masks[index].copy_(torch.as_tensor(next_step.action_mask))
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        *,
        device: torch.device | str = "cpu",
    ) -> MARLBatch:
        if batch_size < 1 or batch_size > self.size:
            raise ValueError("batch_size 必须位于 [1, 当前经验数] 内")
        indices = torch.as_tensor(
            rng.choice(self.size, size=batch_size, replace=False), dtype=torch.long
        )
        def take(values: Tensor) -> Tensor:
            return values.index_select(0, indices)
        terminated = take(self.terminated).expand(-1, self.spec.num_agents)
        truncated = take(self.truncated).expand(-1, self.spec.num_agents)
        batch = MARLBatch(
            observations=take(self.observations),
            actions=take(self.actions),
            rewards=take(self.rewards),
            next_observations=take(self.next_observations),
            terminated=terminated,
            truncated=truncated,
            state=take(self.states),
            next_state=take(self.next_states),
            action_mask=take(self.action_masks) if self.action_masks is not None else None,
            next_action_mask=(
                take(self.next_action_masks) if self.next_action_masks is not None else None
            ),
        )
        return batch.to(device, non_blocking=True)

    def state_dict(self) -> dict[str, object]:
        """保存完整环形缓冲区，以便 checkpoint 后精确继续离策略训练。"""

        state: dict[str, object] = {
            "spec": self.spec,
            "capacity": self.capacity,
            "size": self.size,
            "position": self.position,
        }
        for name in (
            "observations",
            "next_observations",
            "states",
            "next_states",
            "actions",
            "rewards",
            "terminated",
            "truncated",
            "action_masks",
            "next_action_masks",
        ):
            value = getattr(self, name)
            state[name] = value.clone() if value is not None else None
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("spec") != self.spec or state.get("capacity") != self.capacity:
            raise ValueError("replay checkpoint 的 EnvironmentSpec 或 capacity 不一致")
        size = state.get("size")
        position = state.get("position")
        if not isinstance(size, int) or not isinstance(position, int):
            raise TypeError("replay checkpoint 缺少合法 size/position")
        if not 0 <= size <= self.capacity or not 0 <= position < self.capacity:
            raise ValueError("replay checkpoint 的 size/position 越界")
        for name in (
            "observations",
            "next_observations",
            "states",
            "next_states",
            "actions",
            "rewards",
            "terminated",
            "truncated",
            "action_masks",
            "next_action_masks",
        ):
            target = getattr(self, name)
            source = state.get(name)
            if target is None:
                if source is not None:
                    raise ValueError(f"replay checkpoint 不应包含 {name}")
            elif not isinstance(source, Tensor) or source.shape != target.shape:
                raise ValueError(f"replay checkpoint 的 {name} 形状不一致")
            else:
                target.copy_(source)
        self.size = size
        self.position = position
