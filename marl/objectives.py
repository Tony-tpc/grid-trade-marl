"""可组合的训练目标与统一损失结果。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.nn import functional as F


def _scalar(value: Tensor, name: str) -> Tensor:
    if value.numel() != 1:
        raise ValueError(f"{name} 必须是标量，实际形状为 {tuple(value.shape)}")
    return value.reshape(())


@dataclass(frozen=True, slots=True)
class ObjectiveResult:
    """一个目标对总损失的贡献及其只读监控指标。"""

    loss: Tensor
    metrics: Mapping[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _scalar(self.loss, "objective loss")
        for name, value in self.metrics.items():
            _scalar(value, f"metric {name}")


@dataclass(frozen=True, slots=True)
class LossBundle:
    """反向传播使用的总损失与命名指标。"""

    total: Tensor
    terms: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        _scalar(self.total, "total loss")
        for name, value in self.terms.items():
            _scalar(value, f"loss term {name}")

    @classmethod
    def combine(cls, results: Sequence[ObjectiveResult]) -> LossBundle:
        if not results:
            raise ValueError("至少需要一个 ObjectiveResult")
        total = torch.stack([result.loss.reshape(()) for result in results]).sum()
        terms: dict[str, Tensor] = {}
        for result in results:
            for name, value in result.metrics.items():
                if name in terms:
                    raise ValueError(f"重复的 loss/metric 名称：{name}")
                terms[name] = value
        terms["loss"] = total
        return cls(total=total, terms=terms)


@dataclass(frozen=True, slots=True)
class PPOClipObjective:
    """标准 PPO clipped surrogate objective。"""

    clip_ratio: float = 0.2

    def __post_init__(self) -> None:
        if not 0.0 < self.clip_ratio < 1.0:
            raise ValueError("clip_ratio 必须位于 (0, 1)")

    def __call__(
        self, new_log_prob: Tensor, old_log_prob: Tensor, advantages: Tensor
    ) -> ObjectiveResult:
        if new_log_prob.shape != old_log_prob.shape or new_log_prob.shape != advantages.shape:
            raise ValueError("new/old log-prob 与 advantages 必须具有相同形状 [...,N]")
        log_ratio = new_log_prob - old_log_prob
        ratio = log_ratio.exp()
        unclipped = ratio * advantages
        clipped = ratio.clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.clip_ratio).float().mean()
        return ObjectiveResult(
            loss=policy_loss,
            metrics={
                "policy_loss": policy_loss.detach(),
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
            },
        )


@dataclass(frozen=True, slots=True)
class ValueMSEObjective:
    """逐智能体 value MSE；coefficient 只缩放总损失贡献。"""

    coefficient: float = 0.5

    def __post_init__(self) -> None:
        if self.coefficient < 0.0:
            raise ValueError("value coefficient 不能为负")

    def __call__(self, values: Tensor, returns: Tensor) -> ObjectiveResult:
        if values.shape != returns.shape:
            raise ValueError("values 与 returns 必须具有相同形状 [...,N]")
        raw_loss = F.mse_loss(values, returns)
        return ObjectiveResult(
            loss=self.coefficient * raw_loss,
            metrics={"value_loss": raw_loss.detach()},
        )


@dataclass(frozen=True, slots=True)
class EntropyObjective:
    """最大化策略熵，因此对最小化目标贡献负值。"""

    coefficient: float = 0.01

    def __post_init__(self) -> None:
        if self.coefficient < 0.0:
            raise ValueError("entropy coefficient 不能为负")

    def __call__(self, entropy: Tensor) -> ObjectiveResult:
        mean_entropy = entropy.mean()
        return ObjectiveResult(
            loss=-self.coefficient * mean_entropy,
            metrics={"entropy": mean_entropy.detach()},
        )
