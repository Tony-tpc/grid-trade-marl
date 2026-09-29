# marl/objectives.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/objectives.py)

可组合的训练目标与统一损失结果。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [_scalar](#_scalar)
- [ObjectiveResult](#objectiveresult)
- [ObjectiveResult.__post_init__](#objectiveresult-__post_init__)
- [LossBundle](#lossbundle)
- [LossBundle.__post_init__](#lossbundle-__post_init__)
- [LossBundle.combine](#lossbundle-combine)
- [PPOClipObjective](#ppoclipobjective)
- [PPOClipObjective.__post_init__](#ppoclipobjective-__post_init__)
- [PPOClipObjective.__call__](#ppoclipobjective-__call__)
- [ValueMSEObjective](#valuemseobjective)
- [ValueMSEObjective.__post_init__](#valuemseobjective-__post_init__)
- [ValueMSEObjective.__call__](#valuemseobjective-__call__)
- [ClippedValueObjective](#clippedvalueobjective)
- [ClippedValueObjective.__post_init__](#clippedvalueobjective-__post_init__)
- [ClippedValueObjective.__call__](#clippedvalueobjective-__call__)
- [EntropyObjective](#entropyobjective)
- [EntropyObjective.__post_init__](#entropyobjective-__post_init__)
- [EntropyObjective.__call__](#entropyobjective-__call__)
- [TDLossObjective](#tdlossobjective)
- [TDLossObjective.__post_init__](#tdlossobjective-__post_init__)
- [TDLossObjective.__call__](#tdlossobjective-__call__)
- [CounterfactualPolicyObjective](#counterfactualpolicyobjective)
- [CounterfactualPolicyObjective.__call__](#counterfactualpolicyobjective-__call__)
- [DeterministicPolicyObjective](#deterministicpolicyobjective)
- [DeterministicPolicyObjective.__call__](#deterministicpolicyobjective-__call__)
- [SACEntropyObjective](#sacentropyobjective)
- [SACEntropyObjective.__init__](#sacentropyobjective-__init__)
- [SACEntropyObjective.alpha](#sacentropyobjective-alpha)
- [SACEntropyObjective.actor](#sacentropyobjective-actor)
- [SACEntropyObjective.temperature](#sacentropyobjective-temperature)

<a id="_scalar"></a>

## _scalar

[源码位置](../../marl/objectives.py#L13) · [页内目录](#符号目录)

```python
def _scalar(value: Tensor, name: str) -> Tensor
```

检查 Tensor 只有一个元素并 reshape 为零维标量，保留计算图；否则抛 ValueError。

<a id="objectiveresult"></a>

## ObjectiveResult

[源码位置](../../marl/objectives.py#L20) · [页内目录](#符号目录)

`class ObjectiveResult()`

单个数学目标的 loss（带计算图）和标量监控指标（调用方应 detach）。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
loss: Tensor
metrics: Mapping[str, Tensor] = field(default_factory=dict)
```

<a id="objectiveresult-__post_init__"></a>

## ObjectiveResult.__post_init__

[源码位置](../../marl/objectives.py#L26) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 ObjectiveResult 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="lossbundle"></a>

## LossBundle

[源码位置](../../marl/objectives.py#L33) · [页内目录](#符号目录)

`class LossBundle()`

组合后的标量损失与命名指标；不拥有网络、optimizer 或执行顺序。

total 保留计算图供反向传播；terms 用于诊断，其中 loss 可能也带图。
日志出口由 OptimizerRuntime 统一 detach，避免长期保存训练图。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
total: Tensor
terms: Mapping[str, Tensor]
```

<a id="lossbundle-__post_init__"></a>

## LossBundle.__post_init__

[源码位置](../../marl/objectives.py#L43) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 LossBundle 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="lossbundle-combine"></a>

## LossBundle.combine

[源码位置](../../marl/objectives.py#L49) · [页内目录](#符号目录)

```python
def combine(cls, results: Sequence[ObjectiveResult]) -> LossBundle
```

输入非空 ObjectiveResult 序列，求和 loss 并合并唯一命名指标，返回 LossBundle。

<a id="ppoclipobjective"></a>

## PPOClipObjective

[源码位置](../../marl/objectives.py#L64) · [页内目录](#符号目录)

`class PPOClipObjective()`

标准 PPO clipped surrogate objective。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
clip_ratio: float = 0.2
```

<a id="ppoclipobjective-__post_init__"></a>

## PPOClipObjective.__post_init__

[源码位置](../../marl/objectives.py#L69) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 PPOClipObjective 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="ppoclipobjective-__call__"></a>

## PPOClipObjective.__call__

[源码位置](../../marl/objectives.py#L73) · [页内目录](#符号目录)

```python
def __call__(self, new_log_prob: Tensor, old_log_prob: Tensor, advantages: Tensor) -> ObjectiveResult
```

输入同形 [...,N] 新旧 log-prob/advantage，返回裁剪策略目标及 KL、clip_fraction。

<a id="valuemseobjective"></a>

## ValueMSEObjective

[源码位置](../../marl/objectives.py#L97) · [页内目录](#符号目录)

`class ValueMSEObjective()`

逐智能体 value MSE；coefficient 只缩放总损失贡献。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
coefficient: float = 0.5
```

<a id="valuemseobjective-__post_init__"></a>

## ValueMSEObjective.__post_init__

[源码位置](../../marl/objectives.py#L102) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 ValueMSEObjective 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="valuemseobjective-__call__"></a>

## ValueMSEObjective.__call__

[源码位置](../../marl/objectives.py#L106) · [页内目录](#符号目录)

```python
def __call__(self, values: Tensor, returns: Tensor) -> ObjectiveResult
```

输入同形 [...,N] value/return，返回系数加权 MSE 和未加权 value_loss 指标。

<a id="clippedvalueobjective"></a>

## ClippedValueObjective

[源码位置](../../marl/objectives.py#L117) · [页内目录](#符号目录)

`class ClippedValueObjective()`

PPO value clipping，使用 rollout 中保存的 old value 约束单次更新幅度。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
coefficient: float = 0.5
clip_ratio: float = 0.2
```

<a id="clippedvalueobjective-__post_init__"></a>

## ClippedValueObjective.__post_init__

[源码位置](../../marl/objectives.py#L123) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 ClippedValueObjective 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="clippedvalueobjective-__call__"></a>

## ClippedValueObjective.__call__

[源码位置](../../marl/objectives.py#L129) · [页内目录](#符号目录)

```python
def __call__(self, values: Tensor, old_values: Tensor, returns: Tensor, *, scale: Tensor | None=None) -> ObjectiveResult
```

输入 value/old_value/return 和可选尺度，取裁剪与未裁剪误差较大者，返回目标及裁剪率。

<a id="entropyobjective"></a>

## EntropyObjective

[源码位置](../../marl/objectives.py#L153) · [页内目录](#符号目录)

`class EntropyObjective()`

最大化策略熵，因此对最小化目标贡献负值。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
coefficient: float = 0.01
```

<a id="entropyobjective-__post_init__"></a>

## EntropyObjective.__post_init__

[源码位置](../../marl/objectives.py#L158) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 EntropyObjective 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="entropyobjective-__call__"></a>

## EntropyObjective.__call__

[源码位置](../../marl/objectives.py#L162) · [页内目录](#符号目录)

```python
def __call__(self, entropy: Tensor) -> ObjectiveResult
```

输入逐智能体熵，返回负系数乘平均熵的目标和平均 entropy 指标。

<a id="tdlossobjective"></a>

## TDLossObjective

[源码位置](../../marl/objectives.py#L171) · [页内目录](#符号目录)

`class TDLossObjective()`

一步 TD 回归使用的均方误差。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
coefficient: float = 1.0
```

<a id="tdlossobjective-__post_init__"></a>

## TDLossObjective.__post_init__

[源码位置](../../marl/objectives.py#L176) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 TDLossObjective 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="tdlossobjective-__call__"></a>

## TDLossObjective.__call__

[源码位置](../../marl/objectives.py#L180) · [页内目录](#符号目录)

```python
def __call__(self, prediction: Tensor, target: Tensor, *, name: str='critic_loss') -> ObjectiveResult
```

输入同形 prediction/target，返回加权 TD MSE 和 name 指定的原始 MSE 指标。

<a id="counterfactualpolicyobjective"></a>

## CounterfactualPolicyObjective

[源码位置](../../marl/objectives.py#L193) · [页内目录](#符号目录)

`class CounterfactualPolicyObjective()`

固定其他智能体动作，枚举当前智能体 A 个离散动作的反事实目标。

logits/q_values 为 [...,A]；baseline=sum_a pi(a)*Q(a)。概率权重和 Q 全部
detach，只对 log pi 求导，是 score-function 梯度，不对 baseline 反向求导。
baseline 抵消后标量 loss 可接近零，但梯度仍非零，不能据此判断停止学习。

<a id="counterfactualpolicyobjective-__call__"></a>

## CounterfactualPolicyObjective.__call__

[源码位置](../../marl/objectives.py#L201) · [页内目录](#符号目录)

```python
def __call__(self, logits: Tensor, q_values: Tensor) -> ObjectiveResult
```

输入 [...,A] logits/Q，计算 detached 反事实 advantage 的 score-function 目标，返回 ObjectiveResult。

<a id="deterministicpolicyobjective"></a>

## DeterministicPolicyObjective

[源码位置](../../marl/objectives.py#L216) · [页内目录](#符号目录)

`class DeterministicPolicyObjective()`

最大化逐智能体集中式 Q 的确定性策略目标。
input: Q值大小 (多个Batch，多个智能体)
output: 所有智能体的 actor loss (-Q.mean())

<a id="deterministicpolicyobjective-__call__"></a>

## DeterministicPolicyObjective.__call__

[源码位置](../../marl/objectives.py#L222) · [页内目录](#符号目录)

```python
def __call__(self, q_values: Tensor) -> ObjectiveResult
```

输入逐智能体 Q，返回 -mean(Q) 及 actor_loss 指标；梯度隔离由调用算法完成。

<a id="sacentropyobjective"></a>

## SACEntropyObjective

[源码位置](../../marl/objectives.py#L230) · [页内目录](#符号目录)

`class SACEntropyObjective(nn.Module)`

逐智能体 SAC actor/temperature 目标，log_alpha 参数形状 [N]。

actor 将 alpha 视为常量；temperature 将策略 log_prob 视为常量。二者只能由
各自 optimizer 更新，目标熵默认 -A 是连续密度的约定，不是离散最大熵 log(A)。

<a id="sacentropyobjective-__init__"></a>

## SACEntropyObjective.__init__

[源码位置](../../marl/objectives.py#L237) · [页内目录](#符号目录)

```python
def __init__(self, num_agents: int, action_dim: int, *, initial_alpha: float=0.2, target_entropy: float | None=None) -> None
```

构造 SACEntropyObjective，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="sacentropyobjective-alpha"></a>

## SACEntropyObjective.alpha

[源码位置](../../marl/objectives.py#L256) · [页内目录](#符号目录)

```python
def alpha(self) -> Tensor
```

返回 exp(log_alpha) 的正温度 Tensor [N]，该属性本身保留梯度。

<a id="sacentropyobjective-actor"></a>

## SACEntropyObjective.actor

[源码位置](../../marl/objectives.py#L259) · [页内目录](#符号目录)

```python
def actor(self, log_prob: Tensor, min_q: Tensor) -> ObjectiveResult
```

输入 [...,N] log_prob/min_q，返回 mean(detach(alpha)*log_prob-min_q) 策略目标。

<a id="sacentropyobjective-temperature"></a>

## SACEntropyObjective.temperature

[源码位置](../../marl/objectives.py#L268) · [页内目录](#符号目录)

```python
def temperature(self, log_prob: Tensor) -> ObjectiveResult
```

输入 [...,N] log_prob，返回只对 log_alpha 求导的熵目标与温度指标。
