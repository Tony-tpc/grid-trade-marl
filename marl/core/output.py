from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor


@dataclass(slots=True)
class MARLModelOutput:
    """不同算法统一返回的模型结果。

    ``extras`` 用来容纳算法特有内容，例如 Q 值、attention 权重或 recurrent hidden
    state。公共训练流程无需知道这些字段的具体语义，具体算法仍能保留完全的扩展空间。
    """

    actions: Tensor
    logits: Tensor | None = None
    log_prob: Tensor | None = None
    entropy: Tensor | None = None
    values: Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)
