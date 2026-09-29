# marl/returns.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/returns.py)

与具体算法解耦的 return/advantage 估计。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [TD0Estimator](#td0estimator)
- [TD0Estimator.__post_init__](#td0estimator-__post_init__)
- [TD0Estimator.estimate](#td0estimator-estimate)
- [GAEEstimator](#gaeestimator)
- [GAEEstimator.__post_init__](#gaeestimator-__post_init__)
- [GAEEstimator.estimate](#gaeestimator-estimate)
- [AdvantageEstimator](#advantageestimator)
- [AdvantageEstimator.estimate](#advantageestimator-estimate)
- [ValueTargetEstimator](#valuetargetestimator)
- [ValueTargetEstimator.estimate](#valuetargetestimator-estimate)
- [GAEConfig](#gaeconfig)
- [GAEConfig.__post_init__](#gaeconfig-__post_init__)
- [GAEConfig.build](#gaeconfig-build)
- [TD0Config](#td0config)
- [TD0Config.__post_init__](#td0config-__post_init__)
- [TD0Config.build](#td0config-build)

<a id="td0estimator"></a>

## TD0Estimator

[源码位置](../../marl/returns.py#L13) · [页内目录](#符号目录)

`class TD0Estimator()`

逐智能体一步 Bellman target，只由 true termination 阻止 bootstrap。
y_t = r_t + gamma * (1-terminated_t) * V(s_{t+1})

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
gamma: float = 0.99
```

<a id="td0estimator-__post_init__"></a>

## TD0Estimator.__post_init__

[源码位置](../../marl/returns.py#L20) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 TD0Estimator 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="td0estimator-estimate"></a>

## TD0Estimator.estimate

[源码位置](../../marl/returns.py#L24) · [页内目录](#符号目录)

```python
def estimate(self, rewards: Tensor, next_values: Tensor, terminated: Tensor) -> Tensor
```

输入同形 reward/next_value/terminated，返回 r+gamma*(1-terminated)*next_value；不自行 detach。

<a id="gaeestimator"></a>

## GAEEstimator

[源码位置](../../marl/returns.py#L33) · [页内目录](#符号目录)

`class GAEEstimator()`

逐智能体 Generalized Advantage Estimation。

输入第一维必须是时间 ``T``，其余批维保持不变。真实 termination 会阻止
bootstrap；time-limit truncation 仍在当前 delta 中 bootstrap，但会停止 advantage
向下一个 episode 递推。

当前采集器在 rollout 内不 reset：values[t+1] 必须是该 transition 的实际后继
value，末步使用 next_value。若将多个 episode 拼接（尤其截断后 reset），
需先扩展为逐 transition bootstrap value，不能直接传入 reset 后的 value。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
gamma: float = 0.99
gae_lambda: float = 0.95
normalize: bool = True
normalization_scope: Literal['per_agent', 'global'] = 'per_agent'
normalization_epsilon: float = 1e-08
```

<a id="gaeestimator-__post_init__"></a>

## GAEEstimator.__post_init__

[源码位置](../../marl/returns.py#L51) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 GAEEstimator 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="gaeestimator-estimate"></a>

## GAEEstimator.estimate

[源码位置](../../marl/returns.py#L61) · [页内目录](#符号目录)

```python
def estimate(self, rewards: Tensor, values: Tensor, next_value: Tensor, terminated: Tensor, truncated: Tensor) -> tuple[Tensor, Tensor]
```

输入时间优先 [T,...,N] 奖励/value/终止及末步 value [...,N]，返回同形 advantage/return；只标准化 advantage。

<a id="advantageestimator"></a>

## AdvantageEstimator

[源码位置](../../marl/returns.py#L124) · [页内目录](#符号目录)

`class AdvantageEstimator(Protocol)`

输入 [T,...,N]；输出相同形状的 advantage 和 return。

<a id="advantageestimator-estimate"></a>

## AdvantageEstimator.estimate

[源码位置](../../marl/returns.py#L127) · [页内目录](#符号目录)

```python
def estimate(self, rewards: Tensor, values: Tensor, next_value: Tensor, terminated: Tensor, truncated: Tensor) -> tuple[Tensor, Tensor]
```

声明时间优先优势估计接口，输出与 rewards 同形的 (advantages,returns)。

<a id="valuetargetestimator"></a>

## ValueTargetEstimator

[源码位置](../../marl/returns.py#L133) · [页内目录](#符号目录)

`class ValueTargetEstimator(Protocol)`

逐智能体 Bellman target 能力，与 GAE 时序接口分开。

<a id="valuetargetestimator-estimate"></a>

## ValueTargetEstimator.estimate

[源码位置](../../marl/returns.py#L136) · [页内目录](#符号目录)

```python
def estimate(self, rewards: Tensor, next_values: Tensor, terminated: Tensor) -> Tensor
```

声明同形逐智能体 TD target 估计能力；不接收整个时间序列的 GAE 状态。

<a id="gaeconfig"></a>

## GAEConfig

[源码位置](../../marl/returns.py#L142) · [页内目录](#符号目录)

`class GAEConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
kind: Literal['gae'] = 'gae'
gamma: float = 0.99
gae_lambda: float = 0.95
normalize: bool = True
normalization_scope: Literal['per_agent', 'global'] = 'per_agent'
```

<a id="gaeconfig-__post_init__"></a>

## GAEConfig.__post_init__

[源码位置](../../marl/returns.py#L149) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 GAEConfig 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="gaeconfig-build"></a>

## GAEConfig.build

[源码位置](../../marl/returns.py#L155) · [页内目录](#符号目录)

```python
def build(self) -> GAEEstimator
```

从 GAEConfig 的静态参数构造目标对象；输入与具体返回类型见签名，不执行训练。

<a id="td0config"></a>

## TD0Config

[源码位置](../../marl/returns.py#L165) · [页内目录](#符号目录)

`class TD0Config()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
kind: Literal['td0'] = 'td0'
gamma: float = 0.99
```

<a id="td0config-__post_init__"></a>

## TD0Config.__post_init__

[源码位置](../../marl/returns.py#L169) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 TD0Config 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="td0config-build"></a>

## TD0Config.build

[源码位置](../../marl/returns.py#L173) · [页内目录](#符号目录)

```python
def build(self) -> TD0Estimator
```

从 TD0Config 的静态参数构造目标对象；输入与具体返回类型见签名，不执行训练。
