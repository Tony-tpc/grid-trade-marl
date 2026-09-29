# marl/envs/base.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/envs/base.py)

所有论文环境都要遵守的接口约定。

这里不引用任何特定环境的类。算法训练器只需要认识 EnvironmentSpec、
EnvironmentStep 和 MARLBatch，因此以后换论文环境时不用改算法源码。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [ActionKind](#actionkind)
- [RewardStructure](#rewardstructure)
- [EnvironmentSpec](#environmentspec)
- [EnvironmentSpec.__post_init__](#environmentspec-__post_init__)
- [EnvironmentStep](#environmentstep)
- [EnvironmentStep.done](#environmentstep-done)
- [EnvironmentAdapter](#environmentadapter)
- [EnvironmentAdapter.validate_step](#environmentadapter-validate_step)
- [EnvironmentAdapter.spec](#environmentadapter-spec)
- [EnvironmentAdapter.reset](#environmentadapter-reset)
- [EnvironmentAdapter.step](#environmentadapter-step)
- [Transition](#transition)
- [transitions_to_batch](#transitions_to_batch)
- [transitions_to_batch.stack](#transitions_to_batch-stack)

<a id="actionkind"></a>

## ActionKind

[源码位置](../../marl/envs/base.py#L22) · [页内目录](#符号目录)

`class ActionKind(str, Enum)`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

<a id="rewardstructure"></a>

## RewardStructure

[源码位置](../../marl/envs/base.py#L27) · [页内目录](#符号目录)

`class RewardStructure(str, Enum)`

环境给谁定义目标：个体收益博弈，或全体共享同一团队奖励。

<a id="environmentspec"></a>

## EnvironmentSpec

[源码位置](../../marl/envs/base.py#L35) · [页内目录](#符号目录)

`class EnvironmentSpec()`

从环境推导的模型尺寸，而非训练脚本里手填的常量。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
num_agents: int
observation_dim: int
action_dim: int
state_dim: int
action_kind: ActionKind
horizon: int
reward_structure: RewardStructure = RewardStructure.INDIVIDUAL
observation_history: HistoryLayout | None = None
state_layout: StateLayout | None = None
adjacency: tuple[tuple[float, ...], ...] | None = None
```

<a id="environmentspec-__post_init__"></a>

## EnvironmentSpec.__post_init__

[源码位置](../../marl/envs/base.py#L49) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 EnvironmentSpec 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="environmentstep"></a>

## EnvironmentStep

[源码位置](../../marl/envs/base.py#L72) · [页内目录](#符号目录)

`class EnvironmentStep()`

适配器返回的统一格式。观测 [N,O]，状态 [S]，奖励 [N]。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
observations: np.ndarray
state: np.ndarray
rewards: np.ndarray
terminated: bool
truncated: bool
action_mask: np.ndarray | None = None
info: dict[str, Any] = field(default_factory=dict)
```

<a id="environmentstep-done"></a>

## EnvironmentStep.done

[源码位置](../../marl/envs/base.py#L84) · [页内目录](#符号目录)

```python
def done(self) -> bool
```

返回 terminated or truncated；仅用于 episode 控制，不可代替 bootstrap 的 terminated。

<a id="environmentadapter"></a>

## EnvironmentAdapter

[源码位置](../../marl/envs/base.py#L88) · [页内目录](#符号目录)

`class EnvironmentAdapter(ABC)`

训练器依赖的唯一环境接口；具体论文环境继承并实现三个方法。

<a id="environmentadapter-validate_step"></a>

## EnvironmentAdapter.validate_step

[源码位置](../../marl/envs/base.py#L91) · [页内目录](#符号目录)

```python
def validate_step(self, step: EnvironmentStep) -> EnvironmentStep
```

在环境边界检查形状和数值，便于定位新论文适配器中的数据错误。

<a id="environmentadapter-spec"></a>

## EnvironmentAdapter.spec

[源码位置](../../marl/envs/base.py#L122) · [页内目录](#符号目录)

```python
def spec(self) -> EnvironmentSpec
```

返回智能体数、观测/动作/状态尺寸和动作类型。

<a id="environmentadapter-reset"></a>

## EnvironmentAdapter.reset

[源码位置](../../marl/envs/base.py#L126) · [页内目录](#符号目录)

```python
def reset(self, seed: int | None=None) -> EnvironmentStep
```

开启一个 episode。reset 的奖励应为全零。

<a id="environmentadapter-step"></a>

## EnvironmentAdapter.step

[源码位置](../../marl/envs/base.py#L130) · [页内目录](#符号目录)

```python
def step(self, actions: np.ndarray) -> EnvironmentStep
```

执行标准动作 [N] 或 [N,A]，并返回下一时刻数据。

<a id="transition"></a>

## Transition

[源码位置](../../marl/envs/base.py#L135) · [页内目录](#符号目录)

`class Transition()`

环境交互得到的一步经验，供 replay buffer 或 rollout 保存。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
current: EnvironmentStep
actions: np.ndarray
next: EnvironmentStep
```

<a id="transitions_to_batch"></a>

## transitions_to_batch

[源码位置](../../marl/envs/base.py#L143) · [页内目录](#符号目录)

```python
def transitions_to_batch(transitions: list[Transition], device: torch.device | str='cpu') -> MARLBatch
```

把多步经验转换成算法统一读取的 MARLBatch。

``terminated`` 会停止 Bellman bootstrap；``truncated`` 只表示时间限制边界。
二者作为 MARLBatch 的一等字段传递，算法不得重新合并成模糊的 done。

<a id="transitions_to_batch-stack"></a>

## transitions_to_batch.stack

[源码位置](../../marl/envs/base.py#L155) · [页内目录](#符号目录)

```python
def stack(field_name: str, *, next_step: bool=False) -> Tensor
```

按字段名堆叠 current 或 next 的 NumPy 数组，返回目标设备的 float32 Tensor。
