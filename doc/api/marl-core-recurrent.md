# marl/core/recurrent.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/core/recurrent.py)

循环状态的值操作；不拥有模型、环境或训练生命周期。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [map_state](#map_state)
- [stack_states](#stack_states)
- [validate_state](#validate_state)

<a id="map_state"></a>

## map_state

[源码位置](../../marl/core/recurrent.py#L12) · [页内目录](#符号目录)

```python
def map_state(state: RecurrentState, function: Callable[[Tensor], Tensor]) -> RecurrentState
```

对 GRU 的 h 或 LSTM 的 (h,c) 做同一种搬运、索引或 detach 操作。

<a id="stack_states"></a>

## stack_states

[源码位置](../../marl/core/recurrent.py#L23) · [页内目录](#符号目录)

```python
def stack_states(states: Sequence[RecurrentState], dim: int) -> RecurrentState
```

堆叠同构 actor 状态，不混合 MLP/GRU/LSTM 类型。

<a id="validate_state"></a>

## validate_state

[源码位置](../../marl/core/recurrent.py#L42) · [页内目录](#符号目录)

```python
def validate_state(state: RecurrentState, shape: tuple[int, ...], reference: Tensor, *, lstm: bool) -> None
```

只验证元数据，不产生 GPU 数值同步；None 代表新序列的零初态。
