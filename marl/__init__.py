"""本地 MARL 基础设施。

公开 API 保持精简：算法代码通常只需要从这里导入配置、批数据和基础模型。
"""

from marl.core.batch import MARLBatch
from marl.core.output import MARLModelOutput
from marl.models.base import BackboneOutput, BaseBackbone

__all__ = [
    "BackboneOutput",
    "BaseBackbone",
    "MARLBatch",
    "MARLModelOutput",
]
