"""与具体算法解耦的 return/advantage 估计。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class TD0Estimator:
    """逐智能体一步 Bellman target，只由 true termination 阻止 bootstrap。
        y_t = r_t + gamma (1-done_t)V(s_{t+1})
    """

    gamma: float = 0.99

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma 必须位于 [0, 1]")

    def estimate(
        self, rewards: Tensor, next_values: Tensor, terminated: Tensor
    ) -> Tensor:
        if rewards.shape != next_values.shape or rewards.shape != terminated.shape:
            raise ValueError("rewards/next_values/terminated 必须具有相同逐智能体形状")
        return rewards + self.gamma * (~terminated.bool()).to(rewards.dtype) * next_values


@dataclass(frozen=True, slots=True)
class GAEEstimator:
    """逐智能体 Generalized Advantage Estimation。

    输入第一维必须是时间 ``T``，其余批维保持不变。真实 termination 会阻止
    bootstrap；time-limit truncation 仍在当前 delta 中 bootstrap，但会停止 advantage
    向下一个 episode 递推。
    """

    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize: bool = True
    normalization_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma 必须位于 [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda 必须位于 [0, 1]")
        if self.normalization_epsilon <= 0.0:
            raise ValueError("normalization_epsilon 必须大于 0")

    def estimate(
        self,
        rewards: Tensor,
        values: Tensor,
        next_value: Tensor,
        terminated: Tensor,
        truncated: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if rewards.ndim < 2:
            raise ValueError("rewards 至少需要 [T,N] 两个维度")
        if rewards.shape != values.shape:
            raise ValueError("rewards 与 values 形状必须一致")
        if terminated.shape != rewards.shape or truncated.shape != rewards.shape:
            raise ValueError("terminated/truncated 必须与 rewards 形状一致")
        if next_value.shape != rewards.shape[1:]:
            raise ValueError(
                f"next_value 应为 {tuple(rewards.shape[1:])}，实际为 {tuple(next_value.shape)}"
            )
        if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
            raise TypeError("terminated/truncated 必须是 bool Tensor")

        advantages = torch.empty_like(rewards)
        following_value = next_value
        following_advantage = torch.zeros_like(next_value)
        for index in range(rewards.shape[0] - 1, -1, -1):
            bootstrap_mask = ~terminated[index]
            continuation_mask = ~(terminated[index] | truncated[index])
            delta = (
                rewards[index]
                + self.gamma * following_value * bootstrap_mask.to(values.dtype)
                - values[index]
            )
            following_advantage = (
                delta
                + self.gamma
                * self.gae_lambda
                * continuation_mask.to(values.dtype)
                * following_advantage
            )
            advantages[index] = following_advantage
            following_value = values[index]

        returns = advantages + values
        if self.normalize:
            raw_advantages = advantages
            advantages = (raw_advantages - raw_advantages.mean()) / torch.sqrt(
                raw_advantages.var(unbiased=False) + self.normalization_epsilon
            )
        return advantages, returns

class AdvantageEstimator(Protocol):
    """输入 [T,...,N]；输出相同形状的 advantage 和 return。"""

    def estimate(
        self, rewards: Tensor, values: Tensor, next_value: Tensor,
        terminated: Tensor, truncated: Tensor,
    ) -> tuple[Tensor, Tensor]: ...


class ValueTargetEstimator(Protocol):
    """逐智能体 Bellman target 能力，与 GAE 时序接口分开。"""

    def estimate(
        self, rewards: Tensor, next_values: Tensor, terminated: Tensor,
    ) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class GAEConfig:
    kind: Literal["gae"] = "gae"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma/gae_lambda 必须位于 [0,1]")

    def build(self) -> GAEEstimator:
        return GAEEstimator(self.gamma, self.gae_lambda, self.normalize)


@dataclass(frozen=True, slots=True)
class TD0Config:
    kind: Literal["td0"] = "td0"
    gamma: float = 0.99

    def __post_init__(self) -> None:
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma 必须位于 [0,1]")

    def build(self) -> TD0Estimator:
        return TD0Estimator(self.gamma)
