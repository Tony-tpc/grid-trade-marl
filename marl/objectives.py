"""可组合的训练目标与统一损失结果。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
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
class ClippedValueObjective:
    """PPO value clipping，使用 rollout 中保存的 old value 约束单次更新幅度。"""

    coefficient: float = 0.5
    clip_ratio: float = 0.2

    def __post_init__(self) -> None:
        if self.coefficient < 0.0:
            raise ValueError("value coefficient 不能为负")
        if self.clip_ratio <= 0.0:
            raise ValueError("value clip ratio 必须大于 0")

    def __call__(
        self, values: Tensor, old_values: Tensor, returns: Tensor, *, scale: Tensor | None = None
    ) -> ObjectiveResult:
        if values.shape != returns.shape or old_values.shape != returns.shape:
            raise ValueError("values/old_values/returns 必须具有相同逐智能体形状")
        clipped = old_values + (values - old_values).clamp(
            -self.clip_ratio, self.clip_ratio
        )
        errors = torch.maximum(
            (values - returns).pow(2), (clipped - returns).pow(2)
        )
        # clip_ratio 始终在原始 value 单位下应用，尺度仅改变误差权重。
        raw_loss = (errors if scale is None else errors / scale.detach().square()).mean()
        clip_fraction = ((values - old_values).abs() > self.clip_ratio).float().mean()
        return ObjectiveResult(
            loss=self.coefficient * raw_loss,
            metrics={
                "value_loss": raw_loss.detach(),
                "value_clip_fraction": clip_fraction.detach(),
            },
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


@dataclass(frozen=True, slots=True)
class TDLossObjective:
    """一步 TD 回归使用的均方误差。"""

    coefficient: float = 1.0

    def __post_init__(self) -> None:
        if self.coefficient < 0.0:
            raise ValueError("TD loss coefficient 不能为负")

    def __call__(
        self, prediction: Tensor, target: Tensor, *, name: str = "critic_loss"
    ) -> ObjectiveResult:
        if prediction.shape != target.shape:
            raise ValueError("TD prediction 与 target 形状必须一致")
        raw_loss = F.mse_loss(prediction, target)
        return ObjectiveResult(
            loss=self.coefficient * raw_loss,
            metrics={name: raw_loss.detach()},
        )


@dataclass(frozen=True, slots=True)
class CounterfactualPolicyObjective:
    """固定其他智能体动作的离散反事实策略目标。"""

    def __call__(self, logits: Tensor, q_values: Tensor) -> ObjectiveResult:
        if logits.shape != q_values.shape:
            raise ValueError("counterfactual logits 与 q_values 形状必须一致")
        probabilities = logits.softmax(dim=-1).detach()
        baseline = (probabilities * q_values.detach()).sum(dim=-1, keepdim=True)
        advantage = (q_values.detach() - baseline).detach()
        log_probabilities = logits.log_softmax(dim=-1)
        actor_loss = -(probabilities * advantage * log_probabilities).sum(dim=-1).mean()
        return ObjectiveResult(
            loss=actor_loss,
            metrics={"actor_loss": actor_loss.detach()},
        )


@dataclass(frozen=True, slots=True)
class DeterministicPolicyObjective:
    """最大化逐智能体集中式 Q 的确定性策略目标。
        input: Q值大小 (多个Batch，多个智能体)
        output: 所有智能体的 actor loss (-Q.mean())
    """

    def __call__(self, q_values: Tensor) -> ObjectiveResult:
        actor_loss = -q_values.mean()
        return ObjectiveResult(
            loss=actor_loss,
            metrics={"actor_loss": actor_loss.detach()},
        )


class SACEntropyObjective(nn.Module):
    """逐智能体 SAC actor/temperature 目标及可训练 log alpha。"""

    def __init__(
        self,
        num_agents: int,
        action_dim: int,
        *,
        initial_alpha: float = 0.2,
        target_entropy: float | None = None,
    ) -> None:
        super().__init__()
        if initial_alpha <= 0.0:
            raise ValueError("initial_alpha 必须大于 0")
        self.log_alpha = nn.Parameter(
            torch.full((num_agents,), float(initial_alpha)).log()
        )
        self.target_entropy = (
            -float(action_dim) if target_entropy is None else float(target_entropy)
        )

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def actor(self, log_prob: Tensor, min_q: Tensor) -> ObjectiveResult:
        if log_prob.shape != min_q.shape:
            raise ValueError("SAC log_prob 与 min_q 必须具有相同逐智能体形状")
        actor_loss = (self.alpha.detach() * log_prob - min_q).mean()
        return ObjectiveResult(
            loss=actor_loss,
            metrics={"actor_loss": actor_loss.detach()},
        )

    def temperature(self, log_prob: Tensor) -> ObjectiveResult:
        batch_dims = tuple(range(log_prob.ndim - 1))
        alpha_loss = -(
            self.log_alpha
            * (log_prob.detach().mean(dim=batch_dims) + self.target_entropy)
        ).mean()
        return ObjectiveResult(
            loss=alpha_loss,
            metrics={
                "alpha_loss": alpha_loss.detach(),
                "alpha": self.alpha.detach().mean(),
            },
        )
