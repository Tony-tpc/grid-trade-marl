# marl/algorithms/masac.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/algorithms/masac.py)

组合式一般和博弈多智能体 SAC。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [MASACLossConfig](#masaclossconfig)
- [MASACLossConfig.__post_init__](#masaclossconfig-__post_init__)
- [MASACConfig](#masacconfig)
- [MASACConfig.validate](#masacconfig-validate)
- [MASACConfig.build](#masacconfig-build)
- [MASAC](#masac)
- [MASAC.__init__](#masac-__init__)
- [MASAC.act](#masac-act)
- [MASAC.compute_loss_bundle](#masac-compute_loss_bundle)
- [MASAC._validate_training_batch](#masac-_validate_training_batch)
- [MASAC.compute_critic_loss_bundle](#masac-compute_critic_loss_bundle)
- [MASAC.compute_actor_loss_bundle](#masac-compute_actor_loss_bundle)
- [MASAC._actor_and_temperature_results](#masac-_actor_and_temperature_results)
- [MASAC.compute_temperature_loss_bundle](#masac-compute_temperature_loss_bundle)
- [MASAC.update](#masac-update)
- [MASAC.target_pairs](#masac-target_pairs)

<a id="masaclossconfig"></a>

## MASACLossConfig

[源码位置](../../marl/algorithms/masac.py#L35) · [页内目录](#符号目录)

`class MASACLossConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
td_coefficient: float = 1.0
normalize_targets: bool = False
initial_alpha: float = 0.2
target_entropy: float | None = None
```

<a id="masaclossconfig-__post_init__"></a>

## MASACLossConfig.__post_init__

[源码位置](../../marl/algorithms/masac.py#L41) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

```text
检查 TD 损失系数为非负，初始熵温度 initial_alpha 必须为正。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    None；非法配置在构造阶段抛 ValueError。
```

<a id="masacconfig"></a>

## MASACConfig

[源码位置](../../marl/algorithms/masac.py#L56) · [页内目录](#符号目录)

`class MASACConfig()`

MASAC 配置；环境尺寸由 spec 注入。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
schema_version: Literal[2] = 2
algorithm: Literal['masac'] = 'masac'
policy: Buildable[PolicyTopology] = IndependentGaussianConfig()
critic: Buildable[TwinQEnsemble] = TwinQConfig()
loss: MASACLossConfig = MASACLossConfig()
value_target: TD0Config = TD0Config()
replay: ReplayConfig = ReplayConfig()
update: TemperatureActorCriticUpdateConfig = TemperatureActorCriticUpdateConfig()
target_update: SoftTargetConfig | HardTargetConfig = SoftTargetConfig()
```

<a id="masacconfig-validate"></a>

## MASACConfig.validate

[源码位置](../../marl/algorithms/masac.py#L70) · [页内目录](#符号目录)

```python
def validate(self, spec: EnvironmentSpec) -> None
```

```text
验证配置版本和算法标识，并要求环境使用连续动作。

Args:
    spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。

Returns:
    None；不兼容时抛 ValueError。
```

<a id="masacconfig-build"></a>

## MASACConfig.build

[源码位置](../../marl/algorithms/masac.py#L84) · [页内目录](#符号目录)

```python
def build(self, spec: EnvironmentSpec) -> MASAC
```

```text
以当前配置和环境尺寸直接构造 MASAC。

Args:
    spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。

Returns:
    新算法实例；optimizer/trainer 由 build_experiment 另外装配。
```

<a id="masac"></a>

## MASAC

[源码位置](../../marl/algorithms/masac.py#L96) · [页内目录](#符号目录)

`class MASAC(BaseMARLAlgorithm)`

逐智能体双 Q 与可学习温度的连续动作算法。

critic 拟合各自的 soft TD target；actor 通过冻结 critic 对自己的动作求导；
temperature 只优化 log_alpha。三种 loss 共用公式但独立构图/更新，避免
诊断用的总 loss 被误当作单 optimizer 训练入口。

<a id="masac-__init__"></a>

## MASAC.__init__

[源码位置](../../marl/algorithms/masac.py#L104) · [页内目录](#符号目录)

```python
def __init__(self, spec: EnvironmentSpec, config: MASACConfig) -> None
```

```text
校验配置并装配 MASAC 的在线网络、数学目标及所需 target/统计模块。

Args:
    spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。
    config: 当前算法的冻结配置，提供网络、数学目标和更新超参数。

Returns:
    None；可训练参数和 buffers 注册到当前 nn.Module。
```

<a id="masac-act"></a>

## MASAC.act

[源码位置](../../marl/algorithms/masac.py#L130) · [页内目录](#符号目录)

```python
def act(self, observations: Tensor, *, deterministic: bool=False, action_mask: Tensor | None=None, **kwargs: Tensor) -> Tensor
```

```text
将各智能体观测映射为联合动作。随机模式使用 tanh-Gaussian
    采样，确定性模式使用策略确定性输出。

Args:
    observations: 浮点观测 [...,N,O]；循环单步为 [B,N,O]，序列为 [B,T,N,O]。
    deterministic: 是否采用确定性动作；具体算法的例外见本函数说明。
    action_mask: 可选 bool [...,N,A]，True 表示合法；连续动作必须为 None。
    kwargs: 保留的接口扩展关键字；当前具体算法 act 不读取这些值。

Returns:
    动作 Tensor：离散 [...,N]，连续 [...,N,A]；此接口不返回 log-prob/value。
```

<a id="masac-compute_loss_bundle"></a>

## MASAC.compute_loss_bundle

[源码位置](../../marl/algorithms/masac.py#L156) · [页内目录](#符号目录)

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

<a id="masac-_validate_training_batch"></a>

## MASAC._validate_training_batch

[源码位置](../../marl/algorithms/masac.py#L173) · [页内目录](#符号目录)

```python
def _validate_training_batch(self, batch: MARLBatch) -> None
```

```text
校验公共 batch 形状以及 MASAC 所需的 actions/rewards/next_observations。

Args:
    batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。

Returns:
    None；字段缺失或形状不符抛 ValueError。
```

<a id="masac-compute_critic_loss_bundle"></a>

## MASAC.compute_critic_loss_bundle

[源码位置](../../marl/algorithms/masac.py#L186) · [页内目录](#符号目录)

```python
def compute_critic_loss_bundle(self, batch: MARLBatch, *, update_statistics: bool=False) -> LossBundle
```

```text
使用 replay 联合动作拟合逐智能体 TD
    target；下一值来自当前策略、target twin-Q 最小值及熵温度。

使用 replay 动作更新 twin centralized critics。

Args:
    batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。
    update_statistics: 是否更新逐智能体 target 尺度统计；诊断调用默认 False。

Returns:
    critic LossBundle；target 无梯度，只有 terminated 阻止 bootstrap；可选更新尺度统计。
```

<a id="masac-compute_actor_loss_bundle"></a>

## MASAC.compute_actor_loss_bundle

[源码位置](../../marl/algorithms/masac.py#L237) · [页内目录](#符号目录)

```python
def compute_actor_loss_bundle(self, batch: MARLBatch) -> LossBundle
```

```text
冻结 critic 参数并隔离其他 actor 动作，构建每个智能体自己的策略目标。

固定 twin critics 与其他 actor 动作，更新当前策略。

Args:
    batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。

Returns:
    actor LossBundle；只构图，不执行 optimizer.step。
```

<a id="masac-_actor_and_temperature_results"></a>

## MASAC._actor_and_temperature_results

[源码位置](../../marl/algorithms/masac.py#L252) · [页内目录](#符号目录)

```python
def _actor_and_temperature_results(self, batch: MARLBatch) -> tuple[ObjectiveResult, ObjectiveResult]
```

```text
共享一次当前策略采样，计算隔离智能体动作梯度的 actor 目标与独立温度目标。

Args:
    batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。

Returns:
    (actor_result, temperature_result)；两者为 ObjectiveResult，不执行 optimizer。
```

<a id="masac-compute_temperature_loss_bundle"></a>

## MASAC.compute_temperature_loss_bundle

[源码位置](../../marl/algorithms/masac.py#L290) · [页内目录](#符号目录)

```python
def compute_temperature_loss_bundle(self, batch: MARLBatch) -> LossBundle
```

```text
无梯度重新采样当前策略，仅构建 log_alpha 的温度调节目标。

只更新 log alpha；策略 log-prob 在目标内部显式 detach。

Args:
    batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。

Returns:
    温度 LossBundle；梯度只流向熵温度参数。
```

<a id="masac-update"></a>

## MASAC.update

[源码位置](../../marl/algorithms/masac.py#L313) · [页内目录](#符号目录)

```python
def update(self, batch: MARLBatch, runtime: OptimizerRuntime) -> dict[str, float]
```

```text
对一批 replay 经验按 critic → actor → temperature →
    target 顺序更新参数；复用基类 optimize。

按 twin critic → actor → temperature → target 的 MASAC 顺序更新。

Args:
    batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。
    runtime: 已装配的命名 optimizer、裁剪上限、目标更新器、随机生成器及计数状态。

Returns:
    命名 float 指标；runtime.sync_metrics=False 时返回 {}，之后用 flush_metrics 获取。
```

<a id="masac-target_pairs"></a>

## MASAC.target_pairs

[源码位置](../../marl/algorithms/masac.py#L370) · [页内目录](#符号目录)

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
