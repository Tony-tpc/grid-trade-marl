from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from torch import Tensor, nn

from marl.core import MARLBatch
from marl.envs.base import EnvironmentSpec
from marl.objectives import LossBundle


class BaseMARLAlgorithm(nn.Module, ABC):
    """所有具体 MARL 算法的最小公共接口。

    算法只负责模型所有权、动作推理和损失计算。optimizer、epoch、梯度裁剪、
    target update 和 checkpoint 均由 trainer/update plan 负责。
    """

    def __init__(self, spec: EnvironmentSpec) -> None:
        super().__init__()
        self.spec = spec

    @abstractmethod
    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
        **kwargs: Tensor,
    ) -> Tensor:
        """把 `[...,N,O]` 观测映射为逐智能体动作。"""

    @abstractmethod
    def compute_loss_bundle(self, batch: MARLBatch) -> LossBundle:
        """构建算法计算图，但不执行 backward 或 optimizer step。"""

    def target_pairs(self) -> Iterable[tuple[nn.Module, nn.Module]]:
        """返回 `(target, online)` 模块对；无 target network 时为空。"""

        return ()
