# marl/envs/memory_cue.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/envs/memory_cue.py)

延迟提示记忆任务：观测别名使无状态策略不能利用当前帧恢复提示。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [MemoryCueAdapter](#memorycueadapter)
- [MemoryCueAdapter.__init__](#memorycueadapter-__init__)
- [MemoryCueAdapter.spec](#memorycueadapter-spec)
- [MemoryCueAdapter._step_result](#memorycueadapter-_step_result)
- [MemoryCueAdapter.reset](#memorycueadapter-reset)
- [MemoryCueAdapter.step](#memorycueadapter-step)

<a id="memorycueadapter"></a>

## MemoryCueAdapter

[源码位置](../../marl/envs/memory_cue.py#L9) · [页内目录](#符号目录)

`class MemoryCueAdapter(EnvironmentAdapter)`

两个独立二值提示，最后一步按个体答案给 +/-1，中间动作被忽略。

观测 [N,2] 是 (可见提示, 查询标记)；state 仅拼接当前观测，不泄漏历史。
reset(seed=0..3) 平衡枚举四种联合提示；其他 seed 同样通过低两位确定提示。
这是记忆链验收任务，不把它冒充非合作博弈的收敛证明。

<a id="memorycueadapter-__init__"></a>

## MemoryCueAdapter.__init__

[源码位置](../../marl/envs/memory_cue.py#L17) · [页内目录](#符号目录)

```python
def __init__(self, horizon: int=10) -> None
```

构造 MemoryCueAdapter，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="memorycueadapter-spec"></a>

## MemoryCueAdapter.spec

[源码位置](../../marl/envs/memory_cue.py#L25) · [页内目录](#符号目录)

```python
def spec(self) -> EnvironmentSpec
```

无输入；返回静态 EnvironmentSpec，作为所有算法尺寸与动作/奖励语义的唯一来源。

<a id="memorycueadapter-_step_result"></a>

## MemoryCueAdapter._step_result

[源码位置](../../marl/envs/memory_cue.py#L28) · [页内目录](#符号目录)

```python
def _step_result(self, rewards: np.ndarray) -> EnvironmentStep
```

根据当前时间和历史提示构造观测/查询标记，附加输入奖励并返回统一环境结果。

<a id="memorycueadapter-reset"></a>

## MemoryCueAdapter.reset

[源码位置](../../marl/envs/memory_cue.py#L39) · [页内目录](#符号目录)

```python
def reset(self, seed: int | None=None) -> EnvironmentStep
```

按可选 seed 重置环境并返回初始结果；adapter 返回统一 EnvironmentStep，底层协议返回原生格式。

<a id="memorycueadapter-step"></a>

## MemoryCueAdapter.step

[源码位置](../../marl/envs/memory_cue.py#L45) · [页内目录](#符号目录)

```python
def step(self, actions: np.ndarray) -> EnvironmentStep
```

执行输入动作并推进环境；adapter 返回下一 EnvironmentStep，底层协议返回原生观测/奖励/终止信息。
