"""模型、采样器和算法之间共享的数据结构。"""

from marl.core.batch import MARLBatch
from marl.core.output import MARLModelOutput
from marl.core.recurrent import RecurrentState

__all__ = ["MARLBatch", "MARLModelOutput", "RecurrentState"]
