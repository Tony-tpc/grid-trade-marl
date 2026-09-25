from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Normal


@dataclass(slots=True)
class ActionHeadOutput:
    """策略动作头的统一结果，支持离散和连续动作。"""

    actions: Tensor
    log_prob: Tensor
    entropy: Tensor
    distribution_params: dict[str, Tensor]


class BaseActionHead(nn.Module, ABC):
    @abstractmethod
    def forward(
        self, features: Tensor, deterministic: bool = False, action_mask: Tensor | None = None
    ) -> ActionHeadOutput:
        """从特征构造动作分布并采样。"""


class DiscreteActionHead(BaseActionHead):
    """带非法动作 mask 的分类动作头。"""

    def __init__(self, feature_dim: int, action_dim: int) -> None:
        super().__init__()
        self.logits_layer = nn.Linear(feature_dim, action_dim)

    def logits(self, features: Tensor, action_mask: Tensor | None = None) -> Tensor:
        """返回分类分布参数，供只需要 logits 的训练路径使用。

        调用方须确保每行 action_mask 至少有一个合法动作。
        """
        logits = self.logits_layer(features)
        if action_mask is not None:
            if action_mask.shape != logits.shape:
                raise ValueError("action_mask 与 logits 的形状必须一致")
            logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
        return cast(Tensor, logits)

    def forward(
        self, features: Tensor, deterministic: bool = False, action_mask: Tensor | None = None
    ) -> ActionHeadOutput:
        logits = self.logits(features, action_mask)
        distribution = Categorical(logits=logits, validate_args=False)
        actions = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return ActionHeadOutput(
            actions, distribution.log_prob(actions), distribution.entropy(), {"logits": logits}
        )


class DeterministicActionHead(BaseActionHead):
    """MADDPG/TD3 使用的确定性连续动作头。"""

    def __init__(self, feature_dim: int, action_dim: int) -> None:
        super().__init__()
        self.action_layer = nn.Linear(feature_dim, action_dim)

    def forward(
        self,
        features: Tensor,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
    ) -> ActionHeadOutput:
        if action_mask is not None:
            raise ValueError("连续动作头不使用 action_mask")
        actions = torch.tanh(self.action_layer(features))
        zeros = torch.zeros_like(actions[..., 0])
        return ActionHeadOutput(actions, zeros, zeros, {"mean": actions})


class GaussianActionHead(BaseActionHead):
    """连续动作的对角高斯头；tanh 将环境动作限制在 [-1, 1]。"""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
    ) -> None:
        super().__init__()
        self.mean_layer = nn.Linear(feature_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.min_log_std, self.max_log_std = min_log_std, max_log_std

    def forward(
        self, features: Tensor, deterministic: bool = False, action_mask: Tensor | None = None
    ) -> ActionHeadOutput:
        if action_mask is not None:
            raise ValueError("连续动作头不使用 action_mask")
        mean = self.mean_layer(features)
        log_std = self.log_std.clamp(self.min_log_std, self.max_log_std).expand_as(mean)
        distribution = Normal(mean, log_std.exp(), validate_args=False)
        raw_action = mean if deterministic else distribution.rsample()
        actions = torch.tanh(raw_action)
        # tanh 变量变换的 Jacobian 修正，保证 SAC 等算法的 log_prob 正确。
        correction = torch.log(1.0 - actions.pow(2) + 1e-6)
        log_prob = (distribution.log_prob(raw_action) - correction).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return ActionHeadOutput(actions, log_prob, entropy, {"mean": mean, "log_std": log_std})
