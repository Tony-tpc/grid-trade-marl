"""与算法解耦的 target network 更新策略。"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

import torch
from torch import nn

ModuleT = TypeVar("ModuleT")


def frozen_target(module: ModuleT) -> ModuleT:
    """复制 online 网络并冻结参数，作为注册到算法上的 target 网络。"""

    target = deepcopy(module)
    if not isinstance(target, nn.Module):
        raise TypeError("target network 必须是 nn.Module")
    target.requires_grad_(False)
    return target


@dataclass(frozen=True, slots=True)
class SoftTargetUpdate:
    """参数执行 Polyak 平均，非参数 buffer（如 BN 统计）完整复制。

    buffer 可以是整数计数器，不能统一执行 lerp。目标网络是否处于 eval 模式由
    调用者决定；requires_grad=False 本身不会关闭 dropout 或 BN 统计更新。
    """

    tau: float = 0.005

    def __post_init__(self) -> None:
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau 必须位于 (0, 1]")

    @torch.no_grad()
    def step(self, pairs: Iterable[tuple[nn.Module, nn.Module]]) -> None:
        for target, online in pairs:
            for target_parameter, online_parameter in zip(
                target.parameters(), online.parameters(), strict=True
            ):
                target_parameter.lerp_(online_parameter, self.tau)
            for target_buffer, online_buffer in zip(
                target.buffers(), online.buffers(), strict=True
            ):
                target_buffer.copy_(online_buffer)


@dataclass(frozen=True, slots=True)
class HardTargetUpdate:
    """按固定更新次数完整复制 online 参数。"""

    interval: int = 1

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError("interval 必须大于 0")

    def should_step(self, update_count: int) -> bool:
        return update_count % self.interval == 0

    @torch.no_grad()
    def step(self, pairs: Iterable[tuple[nn.Module, nn.Module]]) -> None:
        for target, online in pairs:
            target.load_state_dict(online.state_dict())

class TargetUpdate(Protocol):
    def step(self, pairs: Iterable[tuple[nn.Module, nn.Module]]) -> None: ...


@dataclass(frozen=True, slots=True)
class SoftTargetConfig:
    kind: Literal["soft"] = "soft"
    tau: float = 0.005

    def __post_init__(self) -> None:
        if not 0 < self.tau <= 1:
            raise ValueError("tau 必须位于 (0,1]")

    def build(self) -> SoftTargetUpdate:
        return SoftTargetUpdate(self.tau)


@dataclass(frozen=True, slots=True)
class HardTargetConfig:
    kind: Literal["hard"] = "hard"
    interval: int = 1

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError("interval 必须大于 0")

    def build(self) -> HardTargetUpdate:
        return HardTargetUpdate(self.interval)
