"""训练目标共享的梯度边界工具。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import Tensor, nn


def isolate_agent_action(actions: Tensor, agent_index: int) -> Tensor:
    """固定其他智能体动作，只保留目标智能体动作到 actor 的梯度。

    只保留 ``agent_index`` 动作的梯度，其他智能体动作视作常量。

    Args:
        actions: 连续联合动作 Tensor [...,N,A]，至少包含智能体维和动作维。
        agent_index: 目标智能体的零起始索引，必须位于 [0,N)。

    Returns:
        数值与输入相同的 Tensor [...,N,A]，只有选中智能体切片保留梯度。
    """

    if actions.ndim < 2:
        raise ValueError("actions 至少需要智能体维与动作维")
    num_agents = actions.shape[-2]
    if not 0 <= agent_index < num_agents:
        raise IndexError("agent_index 超出智能体范围")
    return torch.stack(
        [
            actions[..., index, :]
            if index == agent_index
            else actions[..., index, :].detach()
            for index in range(num_agents)
        ],
        dim=-2,
    )


@contextmanager
def frozen_parameters(module: nn.Module) -> Iterator[None]:
    """临时关闭模块参数梯度，保留输出对输入（例如 actor 动作）的可导性。

    临时冻结模块参数，同时保留输出对输入的梯度。

    Args:
        module: 暂时冻结参数的 nn.Module；退出上下文时逐参数恢复 requires_grad。

    Returns:
        上下文管理器，yield None；正常或异常退出均恢复原 requires_grad。
    """

    original = [parameter.requires_grad for parameter in module.parameters()]
    try:
        module.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(module.parameters(), original, strict=True):
            parameter.requires_grad_(requires_grad)
