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


def environment_steps_to_tensors(
    steps: Sequence[EnvironmentStep],
    spec: EnvironmentSpec,
    *,
    device: torch.device | str,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """批量转换环境输出，供 on/off-policy collector 共用一次 actor 前向。"""

    if not steps:
        raise ValueError("steps 不能为空")
    observations = torch.as_tensor(
        np.stack([step.observations for step in steps]),
        dtype=torch.float32,
        device=device,
    )
    states = torch.as_tensor(
        np.stack([step.state for step in steps]),
        dtype=torch.float32,
        device=device,
    )
    action_masks = None
    if spec.action_kind == ActionKind.DISCRETE:
        if any(step.action_mask is None for step in steps):
            raise ValueError("离散环境的每个 step 都必须提供 action_mask")
        action_masks = torch.as_tensor(
            np.stack(
                [step.action_mask for step in steps if step.action_mask is not None]
            ),
            dtype=torch.bool,
            device=device,
        )
    return observations, states, action_masks


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

    def __init__(
        self, spec: EnvironmentSpec, capacity: int, *, pin_memory: bool = False
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity 必须大于零")
        self.spec = spec
        self.capacity = capacity
        self.pin_memory = pin_memory
        self.size = 0
        self.position = 0
        self._sample_staging: dict[int, Tensor] = {}
        self._sample_copy_done: torch.cuda.Event | None = None
        def allocate(*shape: int, dtype: torch.dtype) -> Tensor:
            return torch.empty(
                (capacity, *shape), dtype=dtype, pin_memory=pin_memory
            )
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
        self.record_many((transition,))

    def record_many(self, transitions: Sequence[Transition]) -> None:
        """按环形顺序批量写入 transition，等价于依次调用 ``add``。"""

        if not transitions:
            return
        original_count = len(transitions)
        kept = tuple(transitions[-self.capacity :])
        start = (self.position + original_count - len(kept)) % self.capacity
        indices = (torch.arange(len(kept), dtype=torch.long) + start) % self.capacity
        current_steps = [transition.current for transition in kept]
        next_steps = [transition.next for transition in kept]
        if self.action_masks is not None and self.next_action_masks is not None:
            if any(step.action_mask is None for step in (*current_steps, *next_steps)):
                raise ValueError("离散动作经验必须包含当前及下一步 action_mask")
        sources = (
            (self.observations, np.stack([step.observations for step in current_steps])),
            (
                self.next_observations,
                np.stack([step.observations for step in next_steps]),
            ),
            (self.states, np.stack([step.state for step in current_steps])),
            (self.next_states, np.stack([step.state for step in next_steps])),
            (self.actions, np.stack([transition.actions for transition in kept])),
            (self.rewards, np.stack([step.rewards for step in next_steps])),
        )
        for target, source in sources:
            target.index_copy_(
                0,
                indices,
                torch.as_tensor(source, dtype=target.dtype),
            )
        self.terminated.index_copy_(
            0,
            indices,
            torch.as_tensor(
                [[step.terminated] for step in next_steps], dtype=torch.bool
            ),
        )
        self.truncated.index_copy_(
            0,
            indices,
            torch.as_tensor(
                [[step.truncated] for step in next_steps], dtype=torch.bool
            ),
        )
        if self.action_masks is not None and self.next_action_masks is not None:
            self.action_masks.index_copy_(
                0,
                indices,
                torch.as_tensor(
                    np.stack(
                        [step.action_mask for step in current_steps if step.action_mask is not None]
                    ),
                    dtype=torch.bool,
                ),
            )
            self.next_action_masks.index_copy_(
                0,
                indices,
                torch.as_tensor(
                    np.stack(
                        [step.action_mask for step in next_steps if step.action_mask is not None]
                    ),
                    dtype=torch.bool,
                ),
            )
        self.position = (self.position + original_count) % self.capacity
        self.size = min(self.size + original_count, self.capacity)

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
        destination = torch.device(device)
        reuse_staging = self.pin_memory and destination.type == "cuda"
        if reuse_staging and self._sample_copy_done is not None:
            # 先前异步 H2D 完成才能改写 host staging；返回的 GPU batch 始终独立拥有数据。
            self._sample_copy_done.synchronize()
        def take(values: Tensor) -> Tensor:
            if not self.pin_memory:
                return values.index_select(0, indices)
            # 仅私有 host staging 可复用；CPU 公共 sample 返回独立 pinned 张量。
            shape = (batch_size, *values.shape[1:])
            sampled = self._sample_staging.get(id(values)) if reuse_staging else None
            if sampled is None or sampled.shape != shape:
                sampled = torch.empty(shape, dtype=values.dtype, pin_memory=True)
                if reuse_staging:
                    self._sample_staging[id(values)] = sampled
            return torch.index_select(values, 0, indices, out=sampled)
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
        result = batch.to(destination, non_blocking=True)
        if reuse_staging:
            self._sample_copy_done = torch.cuda.Event()
            self._sample_copy_done.record(torch.cuda.current_stream(destination))
        return result

    def state_dict(self) -> dict[str, object]:
        """保存完整环形缓冲区，以便 checkpoint 后精确继续离策略训练。"""

        state: dict[str, object] = {
            "spec": self.spec,
            "capacity": self.capacity,
            "size": self.size,
            "position": self.position,
            "pin_memory": self.pin_memory,
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
        if state.get("pin_memory", False) != self.pin_memory:
            raise ValueError("replay checkpoint 的 pin_memory 配置不一致")
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
