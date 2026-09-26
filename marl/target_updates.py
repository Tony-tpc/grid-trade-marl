"""与算法解耦的 target network 更新策略。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class NoTargetUpdate:
    def step(self, pairs: Iterable[tuple[nn.Module, nn.Module]] = ()) -> None:
        tuple(pairs)


@dataclass(frozen=True, slots=True)
class SoftTargetUpdate:
    """每次更新后执行 Polyak 平均：target <- (1-tau)target + tau*online。"""

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
