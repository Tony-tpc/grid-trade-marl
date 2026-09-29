# doc/examples/cue_environment.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../doc/examples/cue_environment.py)

可见二值提示教学环境；用于演示 adapter，不用于记忆能力验收。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [CueEnvironment](#cueenvironment)
- [CueEnvironment.__init__](#cueenvironment-__init__)
- [CueEnvironment.spec](#cueenvironment-spec)
- [CueEnvironment._snapshot](#cueenvironment-_snapshot)
- [CueEnvironment.reset](#cueenvironment-reset)
- [CueEnvironment.step](#cueenvironment-step)

<a id="cueenvironment"></a>

## CueEnvironment

[源码位置](../../doc/examples/cue_environment.py#L10) · [页内目录](#符号目录)

`class CueEnvironment(EnvironmentAdapter)`

两个玩家分别按可见提示作答，正确 +1、错误 -1，时间上限截断。

<a id="cueenvironment-__init__"></a>

## CueEnvironment.__init__

[源码位置](../../doc/examples/cue_environment.py#L13) · [页内目录](#符号目录)

```python
def __init__(self, horizon: int=8) -> None
```

输入正整数 horizon；输出 None，初始化固定尺寸规格和私有 RNG。

<a id="cueenvironment-spec"></a>

## CueEnvironment.spec

[源码位置](../../doc/examples/cue_environment.py#L23) · [页内目录](#符号目录)

```python
def spec(self) -> EnvironmentSpec
```

无输入；返回 N=2、O=2、A=2、S=4 的个体奖励规格。

<a id="cueenvironment-_snapshot"></a>

## CueEnvironment._snapshot

[源码位置](../../doc/examples/cue_environment.py#L27) · [页内目录](#符号目录)

```python
def _snapshot(self, rewards: np.ndarray) -> EnvironmentStep
```

输入本步 float32 奖励 [2]；输出独立数组组成且已校验的环境结果。

<a id="cueenvironment-reset"></a>

## CueEnvironment.reset

[源码位置](../../doc/examples/cue_environment.py#L43) · [页内目录](#符号目录)

```python
def reset(self, seed: int | None=None) -> EnvironmentStep
```

输入可选种子；输出初始观测和零奖励，重置时间与提示。

<a id="cueenvironment-step"></a>

## CueEnvironment.step

[源码位置](../../doc/examples/cue_environment.py#L51) · [页内目录](#符号目录)

```python
def step(self, actions: np.ndarray) -> EnvironmentStep
```

输入 int64 动作 [2]；按当前提示奖励后推进，输出下一观测与截断标记。
