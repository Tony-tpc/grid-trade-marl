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
        """初始化 nn.Module 并保存环境规格，不构造具体网络。

        Args:
            spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。

        Returns:
            None；设置 self.spec。
        """
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
        """将各智能体观测映射为联合动作。抽象接口，行为由具体算法定义。

        把 `[...,N,O]` 观测映射为逐智能体动作。

        Args:
            observations: 浮点观测 [...,N,O]；循环单步为 [B,N,O]，序列为 [B,T,N,O]。
            deterministic: 是否采用确定性动作；具体算法的例外见本函数说明。
            action_mask: 可选 bool [...,N,A]，True 表示合法；连续动作必须为 None。
            kwargs: 保留的接口扩展关键字；当前具体算法 act 不读取这些值。

        Returns:
            动作 Tensor：离散 [...,N]，连续 [...,N,A]；此接口不返回 log-prob/value。
        """

    @abstractmethod
    def compute_loss_bundle(self, batch: MARLBatch) -> LossBundle:
        """汇总各子目标用于数学诊断；只构图，不执行 backward、step 或 target 统计更新。

        数学诊断入口：只构图，不做 backward、step 或 target 统计更新。

        actor/critic/temperature 的总和便于单测，但不代表它们能用一个 optimizer
        同时训练。正常训练调用具体算法 update，再由它选择 typed loss 方法。

        Args:
            batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。

        Returns:
            LossBundle；total 是标量总目标，terms 是各项日志；正常训练仍应调用 update。
        """

    def target_pairs(self) -> Iterable[tuple[nn.Module, nn.Module]]:
        """声明目标网络与在线网络的配对，供 runtime.finish 统一同步。

        返回 `(target, online)` 模块对；无 target network 时为空。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            有序 (target, online) 对；基类默认空，不在本函数执行同步。
        """

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
        """执行一次 zero_grad → backward → 可选裁剪 → step。

        执行一次通用优化动作，并返回裁剪前梯度范数。

        ``drop_zero_gradients`` 用于逐智能体更新：把非目标网络产生的全零梯度恢复
        为 ``None``，避免 Adam 历史动量仍然修改这些参数。
        此兜底会触发数值检查；内置 MAAC 的单 actor 前向已从计算图隔离其他 actor，
        不需要兜底。参数列表由 runtime 缓存，不能包含其他 optimizer 的参数。

        Args:
            optimizer: 本次更新唯一负责的优化器；参数归属必须与 bundle 的目标一致。
            bundle: LossBundle；total 为用于 backward 的标量，terms 为命名诊断指标。
            max_grad_norm: 梯度范数裁剪上限；None 表示只测量范数而不裁剪。
            parameters: 该 optimizer 的参数元组；None 时从 param_groups 获取。
            drop_zero_gradients: 是否把全零梯度改回 None，防止无关参数被优化器历史动量更新。

        Returns:
            已 detach 的标量 Tensor，表示裁剪前总梯度范数。
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
