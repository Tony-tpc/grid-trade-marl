# marl/core/batch.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/core/batch.py)

模块职责见类与函数说明。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [MARLBatch](#marlbatch)
- [MARLBatch.valid](#marlbatch-valid)
- [MARLBatch.terminal_flags](#marlbatch-terminal_flags)
- [MARLBatch.validate](#marlbatch-validate)
- [MARLBatch.to](#marlbatch-to)
- [MARLBatch.to.move](#marlbatch-to-move)
- [MARLBatch.pin_memory](#marlbatch-pin_memory)
- [MARLBatch.pin_memory.pin](#marlbatch-pin_memory-pin)

<a id="marlbatch"></a>

## MARLBatch

[源码位置](../../marl/core/batch.py#L12) · [页内目录](#符号目录)

`class MARLBatch()`

MARL 各模块之间传递的标准批数据。

形状约定：
    observations: `[*B, N, O]`，N 为智能体数，O 为单体观测维度。
    actions: `[*B, N]`（离散）或 `[*B, N, A]`（连续）。
    rewards: `[*B, N]`，个体收益任务保留各自的 `r_i`。
    terminated: `[*B, N]`，MDP 真正终止，会阻止 bootstrap。
    truncated: `[*B, N]`，时间限制边界，保留当前下一状态 bootstrap。
    state: `[*B, S]`，仅集中式 critic/mixer 需要。
    action_mask: `[*B, N, A]`，True 表示离散动作可用。

`*B` 可同时包含时间、并行环境和 batch 维。算法不得把 `terminated` 与
`truncated` 合并成一个模糊 done 后用于 Bellman/GAE bootstrap。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
observations: Tensor
actions: Tensor | None = None
rewards: Tensor | None = None
next_observations: Tensor | None = None
terminated: Tensor | None = None
truncated: Tensor | None = None
state: Tensor | None = None
next_state: Tensor | None = None
action_mask: Tensor | None = None
next_action_mask: Tensor | None = None
extras: dict[str, Tensor] = field(default_factory=dict)
policy_state: RecurrentState = None
value_state: RecurrentState = None
sequence_mask: Tensor | None = None
valid_sample_count: int | None = None
```

<a id="marlbatch-valid"></a>

## MARLBatch.valid

[源码位置](../../marl/core/batch.py#L45) · [页内目录](#符号目录)

```python
def valid(self, value: Tensor) -> Tensor
```

完整序列前向之后选择有效位置，保留 N，不对 padding 计算目标。

<a id="marlbatch-terminal_flags"></a>

## MARLBatch.terminal_flags

[源码位置](../../marl/core/batch.py#L49) · [页内目录](#符号目录)

```python
def terminal_flags(self) -> tuple[Tensor, Tensor]
```

返回逐智能体终止标记，缺失时使用全 False。

<a id="marlbatch-validate"></a>

## MARLBatch.validate

[源码位置](../../marl/core/batch.py#L73) · [页内目录](#符号目录)

```python
def validate(self, num_agents: int, observation_dim: int) -> None
```

尽早检查公共维度，避免错误在损失计算深处才暴露。

<a id="marlbatch-to"></a>

## MARLBatch.to

[源码位置](../../marl/core/batch.py#L124) · [页内目录](#符号目录)

```python
def to(self, device: torch.device | str, *, non_blocking: bool=False) -> MARLBatch
```

返回移动到目标设备的新批次，不在原对象上做隐式修改。

<a id="marlbatch-to-move"></a>

## MARLBatch.to.move

[源码位置](../../marl/core/batch.py#L127) · [页内目录](#符号目录)

```python
def move(value: Tensor | None) -> Tensor | None
```

迁移可选 Tensor 到目标设备，None 保持 None；返回迁移结果。

<a id="marlbatch-pin_memory"></a>

## MARLBatch.pin_memory

[源码位置](../../marl/core/batch.py#L155) · [页内目录](#符号目录)

```python
def pin_memory(self) -> MARLBatch
```

返回 pinned-CPU 批次，供 CUDA non-blocking 整批传输。

<a id="marlbatch-pin_memory-pin"></a>

## MARLBatch.pin_memory.pin

[源码位置](../../marl/core/batch.py#L158) · [页内目录](#符号目录)

```python
def pin(value: Tensor | None) -> Tensor | None
```

对可选 CPU Tensor 执行锁页，返回 Tensor 或 None。
