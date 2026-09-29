"""由 adapter 声明的静态输入索引；布局本身不执行网络计算。"""

from dataclasses import dataclass


def _partition(
    groups: tuple[tuple[int, ...], ...], current: tuple[int, ...], input_dim: int
) -> None:
    """确认索引恰好覆盖输入，每个特征只出现一次。"""
    indices = [index for group in groups for index in group] + list(current)
    if any(type(index) is not int for index in indices):
        raise TypeError("布局索引必须是整数")
    if sorted(indices) != list(range(input_dim)):
        raise ValueError("布局索引必须无重复、无遗漏地覆盖输入特征")


@dataclass(frozen=True, slots=True)
class HistoryLayout:
    """history_indices 为 [W,C] 时间优先索引；current_indices 按拼接顺序排列。"""

    history_indices: tuple[tuple[int, ...], ...]
    current_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.history_indices, tuple)
            or not self.history_indices
            or any(not isinstance(row, tuple) or not row for row in self.history_indices)
            or len({len(row) for row in self.history_indices}) != 1
            or not isinstance(self.current_indices, tuple)
        ):
            raise ValueError("历史布局必须是非空矩形 tuple[W,C] 和 current tuple")
        self.validate(sum(map(len, self.history_indices)) + len(self.current_indices))

    def validate(self, input_dim: int) -> None:
        """构建时验证布局与 O 或节点特征维一致。"""
        _partition(self.history_indices, self.current_indices, input_dim)


@dataclass(frozen=True, slots=True)
class StateLayout:
    """全局 state 中的 [N,F] 节点索引、剩余特征和每个节点相同的历史布局。"""

    node_indices: tuple[tuple[int, ...], ...]
    current_indices: tuple[int, ...] = ()
    history: HistoryLayout | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.node_indices, tuple)
            or not self.node_indices
            or any(not isinstance(row, tuple) or not row for row in self.node_indices)
            or len({len(row) for row in self.node_indices}) != 1
            or not isinstance(self.current_indices, tuple)
        ):
            raise ValueError("state 节点布局必须为非空矩形 tuple[N,F]")
        _partition(
            self.node_indices,
            self.current_indices,
            sum(map(len, self.node_indices)) + len(self.current_indices),
        )
        if self.history is not None:
            self.history.validate(len(self.node_indices[0]))

    def validate(self, input_dim: int, num_agents: int) -> None:
        """节点数必须等于 N；完整覆盖 state，禁止根据尺寸猜测节点。"""
        if len(self.node_indices) != num_agents:
            raise ValueError("state 节点数量必须等于 num_agents")
        _partition(self.node_indices, self.current_indices, input_dim)
