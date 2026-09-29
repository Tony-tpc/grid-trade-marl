# marl/value_scaling.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/value_scaling.py)

逐智能体 target 统计；只缩放 critic 误差，不改变对外 V/Q 的单位。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [TargetScale](#targetscale)
- [TargetScale.__init__](#targetscale-__init__)
- [TargetScale.update](#targetscale-update)
- [TargetScale.forward](#targetscale-forward)
- [TargetScale.scale](#targetscale-scale)
- [agent_statistics](#agent_statistics)
- [value_diagnostics](#value_diagnostics)

<a id="targetscale"></a>

## TargetScale

[源码位置](../../marl/value_scaling.py#L9) · [页内目录](#符号目录)

`class TargetScale(nn.Module)`

输入 ``[..., N]``，沿所有样本维更新 Welford 统计。

开关关闭时不保存额外状态，兼容旧模型权重。开启时 count/mean/m2 随
state_dict 保存；统计只由算法 update 显式更新，诊断 loss 不修改状态。
最小标准差为 1，避免初始低方差把梯度放大。不是 PopArt，也不保证收敛。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
count: Tensor
mean: Tensor
m2: Tensor
```

<a id="targetscale-__init__"></a>

## TargetScale.__init__

[源码位置](../../marl/value_scaling.py#L21) · [页内目录](#符号目录)

```python
def __init__(self, num_agents: int, enabled: bool=False) -> None
```

构造 TargetScale，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="targetscale-update"></a>

## TargetScale.update

[源码位置](../../marl/value_scaling.py#L31) · [页内目录](#符号目录)

```python
def update(self, targets: Tensor) -> None
```

输入 [...,N] target，在启用时更新逐智能体 Welford 统计；无梯度、返回 None。

<a id="targetscale-forward"></a>

## TargetScale.forward

[源码位置](../../marl/value_scaling.py#L45) · [页内目录](#符号目录)

```python
def forward(self, values: Tensor) -> Tensor
```

输入 [...,N] 数值，启用时减逐智能体均值再除以尺度；返回同形 Tensor，禁用时原样返回。

<a id="targetscale-scale"></a>

## TargetScale.scale

[源码位置](../../marl/value_scaling.py#L50) · [页内目录](#符号目录)

```python
def scale(self) -> Tensor
```

无输入，返回 [N] 尺度；启用时为最小值 1 的标准差，禁用时为全 1。

<a id="agent_statistics"></a>

## agent_statistics

[源码位置](../../marl/value_scaling.py#L55) · [页内目录](#符号目录)

```python
def agent_statistics(name: str, values: Tensor) -> dict[str, Tensor]
```

对 [..., N] 原始尺度数据返回逐智能体均值、绝对峰值与非有限事件。

<a id="value_diagnostics"></a>

## value_diagnostics

[源码位置](../../marl/value_scaling.py#L69) · [页内目录](#符号目录)

```python
def value_diagnostics(predictions: Tensor, targets: Tensor) -> dict[str, Tensor]
```

原始单位的价值误差；解释方差只沿样本维计算，不混合角色。
