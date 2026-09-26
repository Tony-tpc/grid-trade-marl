"""训练目标共享的梯度边界工具。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from torch import nn


@contextmanager
def frozen_parameters(module: nn.Module) -> Iterator[None]:
    """临时冻结模块参数，同时保留输出对输入的梯度。"""

    original = [parameter.requires_grad for parameter in module.parameters()]
    try:
        module.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(module.parameters(), original, strict=True):
            parameter.requires_grad_(requires_grad)
