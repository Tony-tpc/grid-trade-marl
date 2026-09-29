# marl/core/layout.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/core/layout.py)

由 adapter 声明的静态输入索引；布局本身不执行网络计算。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [_partition](#_partition)
- [HistoryLayout](#historylayout)
- [HistoryLayout.__post_init__](#historylayout-__post_init__)
- [HistoryLayout.validate](#historylayout-validate)
- [StateLayout](#statelayout)
- [StateLayout.__post_init__](#statelayout-__post_init__)
- [StateLayout.validate](#statelayout-validate)

<a id="_partition"></a>

## _partition

[源码位置](../../marl/core/layout.py#L6) · [页内目录](#符号目录)

```python
def _partition(groups: tuple[tuple[int, ...], ...], current: tuple[int, ...], input_dim: int) -> None
```

确认索引恰好覆盖输入，每个特征只出现一次。

<a id="historylayout"></a>

## HistoryLayout

[源码位置](../../marl/core/layout.py#L18) · [页内目录](#符号目录)

`class HistoryLayout()`

history_indices 为 [W,C] 时间优先索引；current_indices 按拼接顺序排列。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
history_indices: tuple[tuple[int, ...], ...]
current_indices: tuple[int, ...]
```

<a id="historylayout-__post_init__"></a>

## HistoryLayout.__post_init__

[源码位置](../../marl/core/layout.py#L24) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 HistoryLayout 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="historylayout-validate"></a>

## HistoryLayout.validate

[源码位置](../../marl/core/layout.py#L35) · [页内目录](#符号目录)

```python
def validate(self, input_dim: int) -> None
```

构建时验证布局与 O 或节点特征维一致。

<a id="statelayout"></a>

## StateLayout

[源码位置](../../marl/core/layout.py#L41) · [页内目录](#符号目录)

`class StateLayout()`

全局 state 中的 [N,F] 节点索引、剩余特征和每个节点相同的历史布局。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
node_indices: tuple[tuple[int, ...], ...]
current_indices: tuple[int, ...] = ()
history: HistoryLayout | None = None
```

<a id="statelayout-__post_init__"></a>

## StateLayout.__post_init__

[源码位置](../../marl/core/layout.py#L48) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 StateLayout 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="statelayout-validate"></a>

## StateLayout.validate

[源码位置](../../marl/core/layout.py#L65) · [页内目录](#符号目录)

```python
def validate(self, input_dim: int, num_agents: int) -> None
```

节点数必须等于 N；完整覆盖 state，禁止根据尺寸猜测节点。
