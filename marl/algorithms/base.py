from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import cast

import torch
from torch import Tensor, nn

from marl.core import MARLBatch
from marl.envs.base import EnvironmentSpec
from marl.objectives import LossBundle


class BaseMARLAlgorithm(nn.Module, ABC):
    """所有具体 MARL 算法的最小公共接口。

    具体算法负责自己的数学目标和更新顺序；本基类只复用单个 optimizer 的标准
    zero_grad/backward/clip/step 动作。经验生命周期和 checkpoint 仍由 trainer 负责。
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

    @staticmethod
    def optimize(
        optimizer: torch.optim.Optimizer,
        bundle: LossBundle,
        max_grad_norm: float | None,
        *,
        parameters: tuple[nn.Parameter, ...] | None = None,
        drop_zero_gradients: bool = False,
    ) -> Tensor:
        """执行一次通用优化动作，并返回裁剪前梯度范数。

        ``drop_zero_gradients`` 用于逐智能体更新：把非目标网络产生的全零梯度恢复
        为 ``None``，避免 Adam 历史动量仍然修改这些参数。
        """

        optimizer.zero_grad(set_to_none=True)
        bundle.total.backward()
        if parameters is None:
            parameters = tuple(
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            )
        if drop_zero_gradients:
            for parameter in parameters:
                if parameter.grad is not None and not torch.count_nonzero(parameter.grad):
                    parameter.grad = None
        with_grad = tuple(parameter for parameter in parameters if parameter.grad is not None)
        if not with_grad:
            gradient_norm = torch.zeros((), device=bundle.total.device)
        elif max_grad_norm is not None:
            gradient_norm = cast(
                Tensor, nn.utils.clip_grad_norm_(with_grad, max_grad_norm)
            )
        else:
            gradient_norm = cast(
                Tensor,
                torch.linalg.vector_norm(
                    torch.stack(
                        [cast(Tensor, parameter.grad).detach().norm() for parameter in with_grad]
                    )
                ),
            )
        optimizer.step()
        return gradient_norm.detach()
