from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

from marl.core.recurrent import RecurrentState


@dataclass(slots=True)
class MARLModelOutput:
    """不同算法统一返回的模型结果。

    ``extras`` 用来容纳算法特有内容，例如 Q 值、attention 权重等
    调试信息。循环状态使用 policy_state/value_state，不藏入无类型 extras。

    actions 为 [...,N] 或 [...,N,A]；log_prob/entropy/values 为 [...,N]。
    与单智能体 ActionHeadOutput 的区别是本类已组装智能体维，且 value/logits
    并非所有算法都具备。它是前向结果，不是可直接重放的 MARLBatch。
    """

    actions: Tensor
    logits: Tensor | None = None
    log_prob: Tensor | None = None
    entropy: Tensor | None = None
    values: Tensor | None = None
    policy_state: RecurrentState = None
    value_state: RecurrentState = None
    extras: dict[str, Any] = field(default_factory=dict)
