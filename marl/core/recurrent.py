"""循环状态的值操作；不拥有模型、环境或训练生命周期。"""

from collections.abc import Callable, Sequence
from typing import TypeAlias

import torch
from torch import Tensor

RecurrentState: TypeAlias = Tensor | tuple[Tensor, Tensor] | None


def map_state(state: RecurrentState, function: Callable[[Tensor], Tensor]) -> RecurrentState:
    """对 GRU 的 h 或 LSTM 的 (h,c) 做同一种搬运、索引或 detach 操作。"""
    if state is None:
        return None
    if isinstance(state, tuple):
        if len(state) != 2:
            raise ValueError("LSTM 状态必须为 (h,c)")
        return function(state[0]), function(state[1])
    return function(state)


def stack_states(states: Sequence[RecurrentState], dim: int) -> RecurrentState:
    """堆叠同构 actor 状态，不混合 MLP/GRU/LSTM 类型。"""
    if not states:
        raise ValueError("states 不能为空")
    first = states[0]
    if first is None:
        if any(state is not None for state in states):
            raise ValueError("同一拓扑的循环状态类型必须一致")
        return None
    if isinstance(first, tuple):
        if not all(isinstance(state, tuple) and len(state) == 2 for state in states):
            raise ValueError("LSTM 状态必须全部为 (h,c)")
        pairs = [state for state in states if isinstance(state, tuple)]
        return torch.stack([s[0] for s in pairs], dim), torch.stack([s[1] for s in pairs], dim)
    if not all(isinstance(state, Tensor) for state in states):
        raise ValueError("GRU 状态必须全部为 Tensor")
    return torch.stack([state for state in states if isinstance(state, Tensor)], dim)


def validate_state(
    state: RecurrentState, shape: tuple[int, ...], reference: Tensor, *, lstm: bool,
) -> None:
    """只验证元数据，不产生 GPU 数值同步；None 代表新序列的零初态。"""
    if state is None:
        return
    tensors: tuple[Tensor, ...]
    if lstm:
        if not isinstance(state, tuple) or len(state) != 2:
            raise ValueError("LSTM hidden_state 必须为 (h,c)")
        tensors = state
    else:
        if not isinstance(state, Tensor):
            raise ValueError("GRU hidden_state 必须为 Tensor")
        tensors = (state,)
    for value in tensors:
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"循环状态形状必须为 {shape}")
        if value.device != reference.device or value.dtype != reference.dtype:
            raise ValueError("循环状态必须与输入具有相同的 device/dtype")
