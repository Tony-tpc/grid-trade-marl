# marl/algorithms/qmix.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/algorithms/qmix.py)

组合式离散合作任务 QMIX。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [QMIXLossConfig](#qmixlossconfig)
- [QMIXLossConfig.__post_init__](#qmixlossconfig-__post_init__)
- [QMIXConfig](#qmixconfig)
- [QMIXConfig.validate](#qmixconfig-validate)
- [QMIXConfig.build](#qmixconfig-build)
- [QMIX](#qmix)
- [QMIX.__init__](#qmix-__init__)
- [QMIX.act](#qmix-act)
- [QMIX.compute_loss_bundle](#qmix-compute_loss_bundle)
- [QMIX.update](#qmix-update)
- [QMIX.target_pairs](#qmix-target_pairs)

<a id="qmixlossconfig"></a>

## QMIXLossConfig

[源码位置](../../marl/algorithms/qmix.py#L28) · [页内目录](#符号目录)

`class QMIXLossConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
td_coefficient: float = 1.0
```

<a id="qmixlossconfig-__post_init__"></a>

## QMIXLossConfig.__post_init__

[源码位置](../../marl/algorithms/qmix.py#L31) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

```text
检查团队 TD 损失系数为非负；不在此处构造网络。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    None；非法配置在构造阶段抛 ValueError。
```

<a id="qmixconfig"></a>

## QMIXConfig

[源码位置](../../marl/algorithms/qmix.py#L44) · [页内目录](#符号目录)

`class QMIXConfig()`

QMIX 配置；环境尺寸由 spec 注入。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
schema_version: Literal[2] = 2
algorithm: Literal['qmix'] = 'qmix'
policy: Buildable[LocalQPolicy] = SharedDiscreteQConfig()
mixer: Buildable[MixingNetwork] = QMixerConfig()
loss: QMIXLossConfig = QMIXLossConfig()
value_target: TD0Config = TD0Config()
replay: ReplayConfig = ReplayConfig()
update: OffPolicyUpdateConfig = OffPolicyUpdateConfig()
target_update: SoftTargetConfig | HardTargetConfig = SoftTargetConfig()
```

<a id="qmixconfig-validate"></a>

## QMIXConfig.validate

[源码位置](../../marl/algorithms/qmix.py#L58) · [页内目录](#符号目录)

```python
def validate(self, spec: EnvironmentSpec) -> None
```

```text
验证配置版本和算法标识，并要求环境使用离散动作。QMIX 还要求共享奖励。

Args:
    spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。

Returns:
    None；不兼容时抛 ValueError。
```

<a id="qmixconfig-build"></a>

## QMIXConfig.build

[源码位置](../../marl/algorithms/qmix.py#L74) · [页内目录](#符号目录)

```python
def build(self, spec: EnvironmentSpec) -> QMIX
```

```text
以当前配置和环境尺寸直接构造 QMIX。

Args:
    spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。

Returns:
    新算法实例；optimizer/trainer 由 build_experiment 另外装配。
```

<a id="qmix"></a>

## QMIX

[源码位置](../../marl/algorithms/qmix.py#L86) · [页内目录](#符号目录)

`class QMIX(BaseMARLAlgorithm)`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

<a id="qmix-__init__"></a>

## QMIX.__init__

[源码位置](../../marl/algorithms/qmix.py#L87) · [页内目录](#符号目录)

```python
def __init__(self, spec: EnvironmentSpec, config: QMIXConfig) -> None
```

```text
校验配置并装配 QMIX 的在线网络、数学目标及所需 target/统计模块。

Args:
    spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。
    config: 当前算法的冻结配置，提供网络、数学目标和更新超参数。

Returns:
    None；可训练参数和 buffers 注册到当前 nn.Module。
```

<a id="qmix-act"></a>

## QMIX.act

[源码位置](../../marl/algorithms/qmix.py#L107) · [页内目录](#符号目录)

```python
def act(self, observations: Tensor, *, deterministic: bool=True, action_mask: Tensor | None=None, **kwargs: Tensor) -> Tensor
```

```text
将各智能体观测映射为联合动作。始终返回合法动作中局部 Q
    最大的动作，deterministic 参数不改变行为。

Args:
    observations: 浮点观测 [...,N,O]；循环单步为 [B,N,O]，序列为 [B,T,N,O]。
    deterministic: 是否采用确定性动作；具体算法的例外见本函数说明。
    action_mask: 可选 bool [...,N,A]，True 表示合法；连续动作必须为 None。
    kwargs: 保留的接口扩展关键字；当前具体算法 act 不读取这些值。

Returns:
    动作 Tensor：离散 [...,N]，连续 [...,N,A]；此接口不返回 log-prob/value。
```

<a id="qmix-compute_loss_bundle"></a>

## QMIX.compute_loss_bundle

[源码位置](../../marl/algorithms/qmix.py#L133) · [页内目录](#符号目录)

```python
def compute_loss_bundle(self, batch: MARLBatch) -> LossBundle
```

```text
验证共享奖励后，混合局部 Q 得到团队 Q；在线选下一动作、target 评估，构造 TD 目标。

Args:
    batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。

Returns:
    团队 TD LossBundle；要求 actions/rewards/next_observations/state/next_state。
```

<a id="qmix-update"></a>

## QMIX.update

[源码位置](../../marl/algorithms/qmix.py#L199) · [页内目录](#符号目录)

```python
def update(self, batch: MARLBatch, runtime: OptimizerRuntime) -> dict[str, float]
```

```text
对一批 replay 经验按 value → target 顺序更新参数；复用基类 optimize。

QMIX 仅执行一次 value optimizer，再更新 target。

Args:
    batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。
    runtime: 已装配的命名 optimizer、裁剪上限、目标更新器、随机生成器及计数状态。

Returns:
    命名 float 指标；runtime.sync_metrics=False 时返回 {}，之后用 flush_metrics 获取。
```

<a id="qmix-target_pairs"></a>

## QMIX.target_pairs

[源码位置](../../marl/algorithms/qmix.py#L228) · [页内目录](#符号目录)

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
