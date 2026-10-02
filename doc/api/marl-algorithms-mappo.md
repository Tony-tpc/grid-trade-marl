# marl/algorithms/mappo.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/algorithms/mappo.py)

由通用策略、价值函数、目标和训练组件组装的 MAPPO。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [MAPPOLossConfig](#mappolossconfig)
- [MAPPOLossConfig.__post_init__](#mappolossconfig-__post_init__)
- [MAPPOConfig](#mappoconfig)
- [MAPPOConfig.validate](#mappoconfig-validate)
- [MAPPOConfig.build](#mappoconfig-build)
- [MAPPO](#mappo)
- [MAPPO.__init__](#mappo-__init__)
- [MAPPO._values_with_state](#mappo-_values_with_state)
- [MAPPO.values](#mappo-values)
- [MAPPO.sample](#mappo-sample)
- [MAPPO.act](#mappo-act)
- [MAPPO.compute_loss_bundle](#mappo-compute_loss_bundle)
- [MAPPO._validate_training_batch](#mappo-_validate_training_batch)
- [MAPPO.compute_policy_loss_bundle](#mappo-compute_policy_loss_bundle)
- [MAPPO.compute_value_loss_bundle](#mappo-compute_value_loss_bundle)
- [MAPPO.update](#mappo-update)

<a id="mappolossconfig"></a>

## MAPPOLossConfig

[源码位置](../../marl/algorithms/mappo.py#L36) · [页内目录](#符号目录)

`class MAPPOLossConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
clip_ratio: float = 0.2
value_coefficient: float = 0.5
entropy_coefficient: float = 0.01
normalize_targets: bool = False
```

<a id="mappolossconfig-__post_init__"></a>

## MAPPOLossConfig.__post_init__

[源码位置](../../marl/algorithms/mappo.py#L42) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

```text
检查 PPO clip_ratio 位于 (0,1)，value/entropy 系数为非负。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    None；非法配置在构造阶段抛 ValueError。
```

<a id="mappoconfig"></a>

## MAPPOConfig

[源码位置](../../marl/algorithms/mappo.py#L57) · [页内目录](#符号目录)

`class MAPPOConfig()`

MAPPO 配置；环境尺寸由 spec 注入。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
schema_version: Literal[2] = 2
algorithm: Literal['mappo'] = 'mappo'
policy: Buildable[StochasticPolicy] = IndependentDiscreteConfig()
critic: Buildable[ValueNetwork] = CentralizedValueConfig()
loss: MAPPOLossConfig = MAPPOLossConfig()
advantage: GAEConfig = GAEConfig()
rollout: RolloutConfig = RolloutConfig()
update: PPOUpdateConfig = PPOUpdateConfig()
```

<a id="mappoconfig-validate"></a>

## MAPPOConfig.validate

[源码位置](../../marl/algorithms/mappo.py#L69) · [页内目录](#符号目录)

```python
def validate(self, spec: EnvironmentSpec) -> None
```

```text
验证配置版本及内置策略与环境动作类型的匹配。

Args:
    spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。

Returns:
    None；不兼容时抛 ValueError。
```

<a id="mappoconfig-build"></a>

## MAPPOConfig.build

[源码位置](../../marl/algorithms/mappo.py#L89) · [页内目录](#符号目录)

```python
def build(self, spec: EnvironmentSpec) -> MAPPO
```

```text
以当前配置和环境尺寸直接构造 MAPPO。

Args:
    spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。

Returns:
    新算法实例；optimizer/trainer 由 build_experiment 另外装配。
```

<a id="mappo"></a>

## MAPPO

[源码位置](../../marl/algorithms/mappo.py#L101) · [页内目录](#符号目录)

`class MAPPO(BaseMARLAlgorithm)`

组合 PPO 目标并拥有 actor→critic 多 epoch 更新顺序。

trainer 只负责采集/GAE/checkpoint。act 只返回动作供环境执行，sample 额外
返回 old log-prob/value 供 rollout 保存；两者不是两套策略实现。

<a id="mappo-__init__"></a>

## MAPPO.__init__

[源码位置](../../marl/algorithms/mappo.py#L108) · [页内目录](#符号目录)

```python
def __init__(self, spec: EnvironmentSpec, config: MAPPOConfig) -> None
```

```text
校验配置并装配 MAPPO 的在线网络、数学目标及所需 target/统计模块。

Args:
    spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。
    config: 当前算法的冻结配置，提供网络、数学目标和更新超参数。

Returns:
    None；可训练参数和 buffers 注册到当前 nn.Module。
```

<a id="mappo-_values_with_state"></a>

## MAPPO._values_with_state

[源码位置](../../marl/algorithms/mappo.py#L141) · [页内目录](#符号目录)

```python
def _values_with_state(self, observations: Tensor, state: Tensor | None, value_state: RecurrentState) -> tuple[Tensor, RecurrentState]
```

```text
用全局 state（缺失时拼接联合观测）计算逐智能体 value，并返回更新后的 critic 状态。

Args:
    observations: 浮点观测 [...,N,O]；循环单步为 [B,N,O]，序列为 [B,T,N,O]。
    state: 可选全局 state [...,S]；None 时使用展平联合观测，尺寸必须与 critic 输入一致。
    value_state: 处理当前观测之前的 critic 状态 [K,B,H]；LSTM 为 (h,c)，MLP 为 None。

Returns:
    (values [...,N], 新 critic 状态)；无状态 critic 返回 None 状态。
```

<a id="mappo-values"></a>

## MAPPO.values

[源码位置](../../marl/algorithms/mappo.py#L165) · [页内目录](#符号目录)

```python
def values(self, observations: Tensor, state: Tensor | None=None, *, value_state: RecurrentState=None) -> Tensor
```

```text
仅估计 value；不调用 actor。函数本身不关闭梯度，bootstrap 调用方须使用 no_grad。

bootstrap 与 value loss 共用前向；不采样 actor，也不改变调用方状态。

Args:
    observations: 浮点观测 [...,N,O]；循环单步为 [B,N,O]，序列为 [B,T,N,O]。
    state: 可选全局 state [...,S]；None 时使用展平联合观测，尺寸必须与 critic 输入一致。
    value_state: 处理当前观测之前的 critic 状态 [K,B,H]；LSTM 为 (h,c)，MLP 为 None。

Returns:
    逐智能体 value Tensor [...,N]；不返回新隐藏状态。
```

<a id="mappo-sample"></a>

## MAPPO.sample

[源码位置](../../marl/algorithms/mappo.py#L186) · [页内目录](#符号目录)

```python
def sample(self, observations: Tensor, state: Tensor | None=None, *, deterministic: bool=False, action_mask: Tensor | None=None, policy_state: RecurrentState=None, value_state: RecurrentState=None) -> MARLModelOutput
```

```text
同时调用 actor 和 critic，为采集返回动作、log-prob、value 及下一时刻隐藏状态。

Args:
    observations: 浮点观测 [...,N,O]；循环单步为 [B,N,O]，序列为 [B,T,N,O]。
    state: 可选全局 state [...,S]；None 时使用展平联合观测，尺寸必须与 critic 输入一致。
    deterministic: 是否采用确定性动作；具体算法的例外见本函数说明。
    action_mask: 可选 bool [...,N,A]，True 表示合法；连续动作必须为 None。
    policy_state: 处理当前观测之前的 actor 状态 [K,B,N,H]；LSTM 为 (h,c)，MLP 为 None。
    value_state: 处理当前观测之前的 critic 状态 [K,B,H]；LSTM 为 (h,c)，MLP 为 None。

Returns:
    MARLModelOutput：actions [...,N]，log_prob/entropy/values [...,N]，以及新 h/c。
```

<a id="mappo-act"></a>

## MAPPO.act

[源码位置](../../marl/algorithms/mappo.py#L231) · [页内目录](#符号目录)

```python
def act(self, observations: Tensor, *, deterministic: bool=False, action_mask: Tensor | None=None, **kwargs: Tensor) -> Tensor
```

```text
将各智能体观测映射为联合动作。循环 actor 必须改用 sample 并持续传递状态。

Args:
    observations: 浮点观测 [...,N,O]；循环单步为 [B,N,O]，序列为 [B,T,N,O]。
    deterministic: 是否采用确定性动作；具体算法的例外见本函数说明。
    action_mask: 可选 bool [...,N,A]，True 表示合法；连续动作必须为 None。
    kwargs: 保留的接口扩展关键字；当前具体算法 act 不读取这些值。

Returns:
    动作 Tensor：离散 [...,N]，连续 [...,N,A]；此接口不返回 log-prob/value。
```

<a id="mappo-compute_loss_bundle"></a>

## MAPPO.compute_loss_bundle

[源码位置](../../marl/algorithms/mappo.py#L258) · [页内目录](#符号目录)

```python
def compute_loss_bundle(self, batch: MARLBatch) -> LossBundle
```

```text
汇总各子目标用于数学诊断；只构图，不执行 backward、step 或 target 统计更新。

Args:
    batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。

Returns:
    LossBundle；total 是标量总目标，terms 是各项日志；正常训练仍应调用 update。
```

<a id="mappo-_validate_training_batch"></a>

## MAPPO._validate_training_batch

[源码位置](../../marl/algorithms/mappo.py#L273) · [页内目录](#符号目录)

```python
def _validate_training_batch(self, batch: MARLBatch) -> None
```

```text
校验公共 batch 形状以及 MAPPO 所需的 actions 及 extras 中
    old_log_prob/advantages/returns；循环路径还校验状态和
    [B,T,N,O]。

Args:
    batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。

Returns:
    None；字段缺失或形状不符抛 ValueError。
```

<a id="mappo-compute_policy_loss_bundle"></a>

## MAPPO.compute_policy_loss_bundle

[源码位置](../../marl/algorithms/mappo.py#L305) · [页内目录](#符号目录)

```python
def compute_policy_loss_bundle(self, batch: MARLBatch) -> LossBundle
```

```text
按旧策略 log-prob 与 advantage 计算 PPO clipped surrogate 和熵正则，排除 padding。

只构建 policy+entropy 计算图，供 actor optimizer 独立更新。

Args:
    batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。

Returns:
    LossBundle：policy/entropy 标量目标和 KL、clip fraction、entropy 等诊断；不 step。
```

<a id="mappo-compute_value_loss_bundle"></a>

## MAPPO.compute_value_loss_bundle

[源码位置](../../marl/algorithms/mappo.py#L358) · [页内目录](#符号目录)

```python
def compute_value_loss_bundle(self, batch: MARLBatch) -> LossBundle
```

```text
回归逐智能体 return；可选围绕 old_values 做 value clipping，排除 padding。

只构建 critic 计算图；可选使用 rollout 保存的 old value 做 clipping。

Args:
    batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。

Returns:
    仅包含 critic 目标的 LossBundle；不更新 target 尺度统计或参数。
```

<a id="mappo-update"></a>

## MAPPO.update

[源码位置](../../marl/algorithms/mappo.py#L398) · [页内目录](#符号目录)

```python
def update(self, experience: PreparedRollout, runtime: OptimizerRuntime) -> dict[str, float]
```

```text
消费一次 fresh rollout，逐 epoch/minibatch 执行 actor
    → critic 更新并汇总有效样本指标。

在 fresh rollout 上按 policy→critic 执行 PPO 多轮 mini-batch 更新。

Args:
    experience: 尚未消费的 PreparedRollout；携带固定
        old log-prob/value、advantage、return。
    runtime: 已装配的命名 optimizer、裁剪上限、目标更新器、随机生成器及计数状态。

Returns:
    命名 float 指标；关闭 runtime 同步时返回 {}。副作用：参数、统计、计数与消费标记改变。
```
