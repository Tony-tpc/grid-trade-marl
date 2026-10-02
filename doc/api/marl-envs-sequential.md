# marl/envs/sequential.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/envs/sequential.py)

顺序博弈的环境能力：先承诺后响应，一个 settle 才算物理步。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [SequentialSpec](#sequentialspec)
- [SequentialSpec.__post_init__](#sequentialspec-__post_init__)
- [SequentialEnvironment](#sequentialenvironment)
- [SequentialEnvironment.spec](#sequentialenvironment-spec)
- [SequentialEnvironment.sequential_spec](#sequentialenvironment-sequential_spec)
- [SequentialEnvironment.checkpoint_context](#sequentialenvironment-checkpoint_context)
- [SequentialEnvironment.reset](#sequentialenvironment-reset)
- [SequentialEnvironment.commit_leader](#sequentialenvironment-commit_leader)
- [SequentialEnvironment.settle](#sequentialenvironment-settle)

<a id="sequentialspec"></a>

## SequentialSpec

[源码位置](../../marl/envs/sequential.py#L12) · [页内目录](#符号目录)

`class SequentialSpec()`

四个角色/联合 EnvironmentSpec 的不可变组合；环境拥有尺寸定义，算法只消费规格。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
joint: EnvironmentSpec
leader: EnvironmentSpec
coordinator: EnvironmentSpec
followers: EnvironmentSpec
```

<a id="sequentialspec-__post_init__"></a>

## SequentialSpec.__post_init__

[源码位置](../../marl/envs/sequential.py#L19) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

尽早校验角色布局、公共 state、事件长度和个体奖励，禁止隐式广播补齐。

<a id="sequentialenvironment"></a>

## SequentialEnvironment

[源码位置](../../marl/envs/sequential.py#L40) · [页内目录](#符号目录)

`class SequentialEnvironment(Protocol)`

联合观测第 0 行为 leader，其余行为 followers；计划信号另行广播。

<a id="sequentialenvironment-spec"></a>

## SequentialEnvironment.spec

[源码位置](../../marl/envs/sequential.py#L43) · [页内目录](#符号目录)

```python
def spec(self) -> EnvironmentSpec
```

返回静态 EnvironmentSpec；智能体数、观测/动作/state 尺寸和 horizon 由当前环境决定。

<a id="sequentialenvironment-sequential_spec"></a>

## SequentialEnvironment.sequential_spec

[源码位置](../../marl/envs/sequential.py#L48) · [页内目录](#符号目录)

```python
def sequential_spec(self) -> SequentialSpec
```

返回联合、UC、随机计划和条件消费者规格；消费者输入追加广播计划维度。

<a id="sequentialenvironment-checkpoint_context"></a>

## SequentialEnvironment.checkpoint_context

[源码位置](../../marl/envs/sequential.py#L53) · [页内目录](#符号目录)

```python
def checkpoint_context(self) -> Mapping[str, object]
```

返回数据 SHA、时间窗口、训练尺度及物理配置，用于拒绝不同数据/环境的 checkpoint 恢复。

<a id="sequentialenvironment-reset"></a>

## SequentialEnvironment.reset

[源码位置](../../marl/envs/sequential.py#L57) · [页内目录](#符号目录)

```python
def reset(self, seed: int | None=None) -> EnvironmentStep
```

按 seed 选择完整日或重置固定事件，清空债务/累计购电和上次动作；返回统一初始观测。

<a id="sequentialenvironment-commit_leader"></a>

## SequentialEnvironment.commit_leader

[源码位置](../../marl/envs/sequential.py#L61) · [页内目录](#符号目录)

```python
def commit_leader(self, action: np.ndarray) -> EnvironmentStep
```

提交 [4] UC 请求，返回含公开承诺的响应阶段观测；不推进物理时间。

<a id="sequentialenvironment-settle"></a>

## SequentialEnvironment.settle

[源码位置](../../marl/envs/sequential.py#L65) · [页内目录](#符号目录)

```python
def settle(self, actions: np.ndarray) -> EnvironmentStep
```

在已承诺 UC 动作下清算消费者 [N,4] 请求，推进一次物理步，返回逐个体奖励及账本。
