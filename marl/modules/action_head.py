from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Normal, TanhTransform


@dataclass(slots=True)
class ActionHeadOutput:
    """单 actor 结果：离散 actions [...]，连续 actions [...,A]。

    log_prob/entropy 均为 [...]，连续动作已对 A 求和；此处不创建智能体维，
    policy 拓扑负责将多个 actor 的结果堆叠成 MARLModelOutput。
    """

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

    def evaluate_actions(
        self,
        features: Tensor,
        actions: Tensor,
        action_mask: Tensor | None = None,
    ) -> ActionHeadOutput:
        """评估给定动作；不支持该能力的动作头应显式报错。"""

        raise NotImplementedError(
            f"{type(self).__name__} 不支持 evaluate_actions"
        )


class DiscreteActionHead(BaseActionHead):
    """带非法动作 mask 的分类动作头。"""

    def __init__(self, feature_dim: int, action_dim: int) -> None:
        super().__init__()
        self.logits_layer = nn.Linear(feature_dim, action_dim)

    def logits(self, features: Tensor, action_mask: Tensor | None = None) -> Tensor:
        """返回分类分布参数，供只需要 logits 的训练路径使用。

        调用方须确保每行 action_mask 至少有一个合法动作。
        """
        logits: Tensor = self.logits_layer(features)
        if action_mask is not None:
            if action_mask.shape != logits.shape:
                raise ValueError("action_mask 与 logits 的形状必须一致")
            logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
        return logits

    def forward(
        self, features: Tensor, deterministic: bool = False, action_mask: Tensor | None = None
    ) -> ActionHeadOutput:
        logits = self.logits(features, action_mask)
        distribution = Categorical(logits=logits, validate_args=False)
        actions = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return ActionHeadOutput(
            actions, distribution.log_prob(actions), distribution.entropy(), {"logits": logits}
        )

    def evaluate_actions(
        self,
        features: Tensor,
        actions: Tensor,
        action_mask: Tensor | None = None,
    ) -> ActionHeadOutput:
        """返回给定离散动作的 log-prob 和 entropy，不重新采样。"""

        logits = self.logits(features, action_mask)
        if actions.shape != logits.shape[:-1]:
            raise ValueError("离散 actions 形状必须等于 logits 去掉动作维后的形状")
        distribution = Categorical(logits=logits, validate_args=False)
        selected = actions.long()
        return ActionHeadOutput(
            selected,
            distribution.log_prob(selected),
            distribution.entropy(),
            {"logits": logits},
        )


class DeterministicActionHead(BaseActionHead):
    """确定性连续动作头；探索噪声由 collector 添加，不混入网络定义。

    返回的零 log_prob/entropy 仅为统一结果的占位值，不是真实概率密度，
    不能用于 PPO 概率比或 SAC 熵项。evaluate_actions 因而保持显式不支持。
    """

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
    """对角高斯重参数化采样，再经 tanh 限制到 [-1,1]。

    mean 依赖观测，log_std 是每个动作维共享于所有观测的可学习参数；不是
    state-dependent std。当前只实现采样路径，不宣称支持连续 PPO 的固定动作评估。
    """

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
        # 使用 raw action 的稳定公式；饱和时 1-tanh(x)^2 会舍入为零。
        correction = TanhTransform().log_abs_det_jacobian(raw_action, actions)
        log_prob = (distribution.log_prob(raw_action) - correction).sum(dim=-1)
        # squash 后没有解析熵；随机动作上的 -log_prob 是 Monte Carlo 估计。
        # deterministic=True 时它只是该动作的 surprisal，不应当解释为熵。
        return ActionHeadOutput(actions, log_prob, -log_prob, {
            "mean": mean, "log_std": log_std,
            "gaussian_entropy": distribution.entropy().sum(dim=-1),
        })
