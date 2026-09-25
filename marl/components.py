"""算法实现所依赖的平级组件协议。

这些协议只描述组件之间如何协作，不提供新的算法继承层。带参数的实现仍然使用
``torch.nn.Module``，无参数实现可以使用普通 dataclass。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Protocol, runtime_checkable

from torch import Tensor, nn

from marl.core import MARLBatch, MARLModelOutput


@runtime_checkable
class PolicyTopology(Protocol):
    """把观测映射为动作分布，并暴露可训练参数。"""

    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
    ) -> MARLModelOutput: ...

    def evaluate(
        self,
        observations: Tensor,
        actions: Tensor,
        *,
        action_mask: Tensor | None = None,
    ) -> MARLModelOutput: ...

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...


@runtime_checkable
class Objective(Protocol):
    """根据批数据和模型输出计算一个命名训练目标。"""

    name: str

    def __call__(
        self, batch: MARLBatch, outputs: Mapping[str, Tensor]
    ) -> Tensor: ...


@runtime_checkable
class ReturnEstimator(Protocol):
    """把时序奖励和值估计转换成 advantage 与 return。"""

    def estimate(
        self,
        rewards: Tensor,
        values: Tensor,
        next_value: Tensor,
        terminated: Tensor,
        truncated: Tensor,
    ) -> tuple[Tensor, Tensor]: ...


@runtime_checkable
class ExperienceSource(Protocol):
    """提供一次训练所需的新鲜经验批次。"""

    def batches(self) -> Iterable[MARLBatch]: ...


@runtime_checkable
class UpdatePlan(Protocol):
    """控制参数组、mini-batch 与优化器更新顺序。"""

    def update(self, algorithm: nn.Module, batches: Iterable[MARLBatch]) -> Mapping[str, float]: ...


@runtime_checkable
class TargetUpdate(Protocol):
    """描述 online/target 模块之间的同步策略。"""

    def step(self, pairs: Iterable[tuple[nn.Module, nn.Module]]) -> None: ...
