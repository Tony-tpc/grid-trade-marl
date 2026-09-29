# marl/modules/action_head.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/modules/action_head.py)

模块职责见类与函数说明。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [ActionHeadOutput](#actionheadoutput)
- [BaseActionHead](#baseactionhead)
- [BaseActionHead.forward](#baseactionhead-forward)
- [BaseActionHead.evaluate_actions](#baseactionhead-evaluate_actions)
- [DiscreteActionHead](#discreteactionhead)
- [DiscreteActionHead.__init__](#discreteactionhead-__init__)
- [DiscreteActionHead.logits](#discreteactionhead-logits)
- [DiscreteActionHead.forward](#discreteactionhead-forward)
- [DiscreteActionHead.evaluate_actions](#discreteactionhead-evaluate_actions)
- [DeterministicActionHead](#deterministicactionhead)
- [DeterministicActionHead.__init__](#deterministicactionhead-__init__)
- [DeterministicActionHead.forward](#deterministicactionhead-forward)
- [GaussianActionHead](#gaussianactionhead)
- [GaussianActionHead.__init__](#gaussianactionhead-__init__)
- [GaussianActionHead.forward](#gaussianactionhead-forward)

<a id="actionheadoutput"></a>

## ActionHeadOutput

[源码位置](../../marl/modules/action_head.py#L14) · [页内目录](#符号目录)

`class ActionHeadOutput()`

单 actor 结果：离散 actions [...]，连续 actions [...,A]。

log_prob/entropy 均为 [...]，连续动作已对 A 求和；此处不创建智能体维，
policy 拓扑负责将多个 actor 的结果堆叠成 MARLModelOutput。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
actions: Tensor
log_prob: Tensor
entropy: Tensor
distribution_params: dict[str, Tensor]
hidden_state: RecurrentState = None
```

<a id="baseactionhead"></a>

## BaseActionHead

[源码位置](../../marl/modules/action_head.py#L28) · [页内目录](#符号目录)

`class BaseActionHead(nn.Module, ABC)`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

<a id="baseactionhead-forward"></a>

## BaseActionHead.forward

[源码位置](../../marl/modules/action_head.py#L30) · [页内目录](#符号目录)

```python
def forward(self, features: Tensor, deterministic: bool=False, action_mask: Tensor | None=None) -> ActionHeadOutput
```

从特征构造动作分布并采样。

<a id="baseactionhead-evaluate_actions"></a>

## BaseActionHead.evaluate_actions

[源码位置](../../marl/modules/action_head.py#L35) · [页内目录](#符号目录)

```python
def evaluate_actions(self, features: Tensor, actions: Tensor, action_mask: Tensor | None=None) -> ActionHeadOutput
```

评估给定动作；不支持该能力的动作头应显式报错。

<a id="discreteactionhead"></a>

## DiscreteActionHead

[源码位置](../../marl/modules/action_head.py#L48) · [页内目录](#符号目录)

`class DiscreteActionHead(BaseActionHead)`

带非法动作 mask 的分类动作头。

<a id="discreteactionhead-__init__"></a>

## DiscreteActionHead.__init__

[源码位置](../../marl/modules/action_head.py#L51) · [页内目录](#符号目录)

```python
def __init__(self, feature_dim: int, action_dim: int) -> None
```

构造 DiscreteActionHead，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="discreteactionhead-logits"></a>

## DiscreteActionHead.logits

[源码位置](../../marl/modules/action_head.py#L55) · [页内目录](#符号目录)

```python
def logits(self, features: Tensor, action_mask: Tensor | None=None) -> Tensor
```

返回分类分布参数，供只需要 logits 的训练路径使用。

调用方须确保每行 action_mask 至少有一个合法动作。

<a id="discreteactionhead-forward"></a>

## DiscreteActionHead.forward

[源码位置](../../marl/modules/action_head.py#L67) · [页内目录](#符号目录)

```python
def forward(self, features: Tensor, deterministic: bool=False, action_mask: Tensor | None=None) -> ActionHeadOutput
```

输入 features、deterministic 与可选 mask，返回采样动作及其 logits/log_prob/entropy。

<a id="discreteactionhead-evaluate_actions"></a>

## DiscreteActionHead.evaluate_actions

[源码位置](../../marl/modules/action_head.py#L77) · [页内目录](#符号目录)

```python
def evaluate_actions(self, features: Tensor, actions: Tensor, action_mask: Tensor | None=None) -> ActionHeadOutput
```

返回给定离散动作的 log-prob 和 entropy，不重新采样。

<a id="deterministicactionhead"></a>

## DeterministicActionHead

[源码位置](../../marl/modules/action_head.py#L98) · [页内目录](#符号目录)

`class DeterministicActionHead(BaseActionHead)`

确定性连续动作头；探索噪声由 collector 添加，不混入网络定义。

返回的零 log_prob/entropy 仅为统一结果的占位值，不是真实概率密度，
不能用于 PPO 概率比或 SAC 熵项。evaluate_actions 因而保持显式不支持。

<a id="deterministicactionhead-__init__"></a>

## DeterministicActionHead.__init__

[源码位置](../../marl/modules/action_head.py#L105) · [页内目录](#符号目录)

```python
def __init__(self, feature_dim: int, action_dim: int) -> None
```

构造 DeterministicActionHead，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="deterministicactionhead-forward"></a>

## DeterministicActionHead.forward

[源码位置](../../marl/modules/action_head.py#L109) · [页内目录](#符号目录)

```python
def forward(self, features: Tensor, deterministic: bool=False, action_mask: Tensor | None=None) -> ActionHeadOutput
```

输入 features，返回 tanh 限幅连续动作的 ActionHeadOutput。

<a id="gaussianactionhead"></a>

## GaussianActionHead

[源码位置](../../marl/modules/action_head.py#L122) · [页内目录](#符号目录)

`class GaussianActionHead(BaseActionHead)`

对角高斯重参数化采样，再经 tanh 限制到 [-1,1]。

mean 依赖观测，log_std 是每个动作维共享于所有观测的可学习参数；不是
state-dependent std。当前只实现采样路径，不宣称支持连续 PPO 的固定动作评估。

<a id="gaussianactionhead-__init__"></a>

## GaussianActionHead.__init__

[源码位置](../../marl/modules/action_head.py#L129) · [页内目录](#符号目录)

```python
def __init__(self, feature_dim: int, action_dim: int, min_log_std: float=-20.0, max_log_std: float=2.0) -> None
```

构造 GaussianActionHead，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="gaussianactionhead-forward"></a>

## GaussianActionHead.forward

[源码位置](../../marl/modules/action_head.py#L141) · [页内目录](#符号目录)

```python
def forward(self, features: Tensor, deterministic: bool=False, action_mask: Tensor | None=None) -> ActionHeadOutput
```

输入 features 与 deterministic，返回 tanh-Gaussian 动作、修正 log-prob 和熵估计。
