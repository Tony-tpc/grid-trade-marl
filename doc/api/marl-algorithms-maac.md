# marl/algorithms/maac.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/algorithms/maac.py)

组合式离散动作 MAAC，保留逐智能体收益。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [MAACLossConfig](#maaclossconfig)
- [MAACLossConfig.__post_init__](#maaclossconfig-__post_init__)
- [MAACConfig](#maacconfig)
- [MAACConfig.validate](#maacconfig-validate)
- [MAACConfig.build](#maacconfig-build)
- [MAAC](#maac)
- [MAAC.__init__](#maac-__init__)
- [MAAC.act](#maac-act)
- [MAAC.compute_loss_bundle](#maac-compute_loss_bundle)
- [MAAC._validate_training_batch](#maac-_validate_training_batch)
- [MAAC.compute_critic_loss_bundle](#maac-compute_critic_loss_bundle)
- [MAAC.compute_actor_loss_bundle](#maac-compute_actor_loss_bundle)
- [MAAC.update](#maac-update)
- [MAAC.target_pairs](#maac-target_pairs)

<a id="maaclossconfig"></a>

## MAACLossConfig

[源码位置](../../marl/algorithms/maac.py#L41) · [页内目录](#符号目录)

`class MAACLossConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
td_coefficient: float = 1.0
normalize_targets: bool = False
entropy_coefficient: float = 0.01
```

<a id="maaclossconfig-__post_init__"></a>

## MAACLossConfig.__post_init__

[源码位置](../../marl/algorithms/maac.py#L46) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

```text
检查 TD 与 entropy 系数为非负；不在此处构造网络。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    None；非法配置在构造阶段抛 ValueError。
```

<a id="maacconfig"></a>

## MAACConfig

[源码位置](../../marl/algorithms/maac.py#L60) · [页内目录](#符号目录)

`class MAACConfig()`

MAAC 配置；环境尺寸由 spec 注入。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
schema_version: Literal[2] = 2
algorithm: Literal['maac'] = 'maac'
policy: Buildable[DiscretePolicy] = IndependentDiscreteConfig()
critic: Buildable[AttentionQNetwork] = AttentionQConfig()
loss: MAACLossConfig = MAACLossConfig()
value_target: TD0Config = TD0Config()
replay: ReplayConfig = ReplayConfig()
update: ActorCriticUpdateConfig = ActorCriticUpdateConfig()
target_update: SoftTargetConfig | HardTargetConfig = SoftTargetConfig()
```

<a id="maacconfig-validate"></a>

## MAACConfig.validate

[源码位置](../../marl/algorithms/maac.py#L74) · [页内目录](#符号目录)

```python
def validate(self, spec: EnvironmentSpec) -> None
```

```text
验证配置版本和算法标识，并要求环境使用离散动作。

Args:
    spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。

Returns:
    None；不兼容时抛 ValueError。
```

<a id="maacconfig-build"></a>

## MAACConfig.build

[源码位置](../../marl/algorithms/maac.py#L88) · [页内目录](#符号目录)

```python
def build(self, spec: EnvironmentSpec) -> MAAC
```

```text
以当前配置和环境尺寸直接构造 MAAC。

Args:
    spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。

Returns:
    新算法实例；optimizer/trainer 由 build_experiment 另外装配。
```

<a id="maac"></a>

## MAAC

[源码位置](../../marl/algorithms/maac.py#L100) · [页内目录](#符号目录)

`class MAAC(BaseMARLAlgorithm)`

离散反事实 actor-critic；每个玩家保留自己的策略和 Q head。

``compute_*_loss_bundle`` 只构图，``update`` 才决定 critic→各 actor→target。
policy loss 固定其他玩家动作，不通过 critic 回传；critic loss 使用 replay
中的真实联合动作。采集和 replay 生命周期仍由通用 trainer 管理。

<a id="maac-__init__"></a>

## MAAC.__init__

[源码位置](../../marl/algorithms/maac.py#L108) · [页内目录](#符号目录)

```python
def __init__(self, spec: EnvironmentSpec, config: MAACConfig) -> None
```

```text
校验配置并装配 MAAC 的在线网络、数学目标及所需 target/统计模块。

Args:
    spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。
    config: 当前算法的冻结配置，提供网络、数学目标和更新超参数。

Returns:
    None；可训练参数和 buffers 注册到当前 nn.Module。
```

<a id="maac-act"></a>

## MAAC.act

[源码位置](../../marl/algorithms/maac.py#L133) · [页内目录](#符号目录)

```python
def act(self, observations: Tensor, *, deterministic: bool=False, action_mask: Tensor | None=None, **kwargs: Tensor) -> Tensor
```

```text
将各智能体观测映射为联合动作。在合法离散动作上采样或选择最大 logit。

Args:
    observations: 浮点观测 [...,N,O]；循环单步为 [B,N,O]，序列为 [B,T,N,O]。
    deterministic: 是否采用确定性动作；具体算法的例外见本函数说明。
    action_mask: 可选 bool [...,N,A]，True 表示合法；连续动作必须为 None。
    kwargs: 保留的接口扩展关键字；当前具体算法 act 不读取这些值。

Returns:
    动作 Tensor：离散 [...,N]，连续 [...,N,A]；此接口不返回 log-prob/value。
```

<a id="maac-compute_loss_bundle"></a>

## MAAC.compute_loss_bundle

[源码位置](../../marl/algorithms/maac.py#L158) · [页内目录](#符号目录)

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

<a id="maac-_validate_training_batch"></a>

## MAAC._validate_training_batch

[源码位置](../../marl/algorithms/maac.py#L189) · [页内目录](#符号目录)

```python
def _validate_training_batch(self, batch: MARLBatch) -> None
```

```text
校验公共 batch 形状以及 MAAC 所需的 actions/rewards/next_observations。

Args:
    batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。

Returns:
    None；字段缺失或形状不符抛 ValueError。
```

<a id="maac-compute_critic_loss_bundle"></a>

## MAAC.compute_critic_loss_bundle

[源码位置](../../marl/algorithms/maac.py#L202) · [页内目录](#符号目录)

```python
def compute_critic_loss_bundle(self, batch: MARLBatch, *, update_statistics: bool=False) -> LossBundle
```

```text
使用 replay 联合动作拟合逐智能体 TD target；下一值来自target
    policy、attention Q 及熵正则。

使用 replay 联合动作回归逐智能体 TD target。

Args:
    batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。
    update_statistics: 是否更新逐智能体 target 尺度统计；诊断调用默认 False。

Returns:
    critic LossBundle；target 无梯度，只有 terminated 阻止 bootstrap；可选更新尺度统计。
```

<a id="maac-compute_actor_loss_bundle"></a>

## MAAC.compute_actor_loss_bundle

[源码位置](../../marl/algorithms/maac.py#L253) · [页内目录](#符号目录)

```python
def compute_actor_loss_bundle(self, batch: MARLBatch, agent_index: int, *, current_q_values: Tensor | None=None) -> LossBundle
```

```text
构建单智能体反事实策略目标；固定 Q 表和其他玩家动作。

只构建一个智能体的 policy loss；其他 actor 不接收梯度。

Args:
    batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。
    agent_index: 目标智能体的零起始索引，必须位于 [0,N)。
    current_q_values: 可复用的当前联合动作 Q 表 [...,N,A]；None 时内部无梯度采样并计算。

Returns:
    actor LossBundle；只构图，不执行 optimizer.step。
```

<a id="maac-update"></a>

## MAAC.update

[源码位置](../../marl/algorithms/maac.py#L304) · [页内目录](#符号目录)

```python
def update(self, batch: MARLBatch, runtime: OptimizerRuntime) -> dict[str, float]
```

```text
对一批 replay 经验按 critic → 逐智能体 actor → target
    顺序更新参数；复用基类 optimize。

按 critic → 每个智能体 policy → target 的 MAAC 顺序更新。

Args:
    batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。
    runtime: 已装配的命名 optimizer、裁剪上限、目标更新器、随机生成器及计数状态。

Returns:
    命名 float 指标；runtime.sync_metrics=False 时返回 {}，之后用 flush_metrics 获取。
```

<a id="maac-target_pairs"></a>

## MAAC.target_pairs

[源码位置](../../marl/algorithms/maac.py#L376) · [页内目录](#符号目录)

```python
def target_pairs(self) -> tuple[tuple[nn.Module, nn.Module], ...]
```

```text
声明目标网络与在线网络的配对，供 runtime.finish 统一同步。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    有序 (target, online) 对；基类默认空，不在本函数执行同步。
```
