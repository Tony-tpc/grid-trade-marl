# marl/algorithms/maddpg.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/algorithms/maddpg.py)

一般和博弈版 MADDPG：每个智能体优化自己的累计奖励。
replay 保存观测、联合动作、逐体奖励、下一观测及独立的终止/截断标记。
下一动作由 target actor 生成；只有真正终止阻止 TD bootstrap。
一个 Agent 对应一个 Actor 、一个 Critic
Critic 是集中式的，读取所有智能体的观测和动作。
Actor 是独立的，只读取各自智能体的观测。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [MADDPGLossConfig](#maddpglossconfig)
- [MADDPGLossConfig.__post_init__](#maddpglossconfig-__post_init__)
- [MADDPGConfig](#maddpgconfig)
- [MADDPGConfig.validate](#maddpgconfig-validate)
- [MADDPGConfig.build](#maddpgconfig-build)
- [MADDPG](#maddpg)
- [MADDPG.__init__](#maddpg-__init__)
- [MADDPG.act](#maddpg-act)
- [MADDPG.compute_loss_bundle](#maddpg-compute_loss_bundle)
- [MADDPG._validate_training_batch](#maddpg-_validate_training_batch)
- [MADDPG.compute_critic_loss_bundle](#maddpg-compute_critic_loss_bundle)
- [MADDPG.compute_actor_loss_bundle](#maddpg-compute_actor_loss_bundle)
- [MADDPG.update](#maddpg-update)
- [MADDPG.target_pairs](#maddpg-target_pairs)

<a id="maddpglossconfig"></a>

## MADDPGLossConfig

[源码位置](../../marl/algorithms/maddpg.py#L43) · [页内目录](#符号目录)

`class MADDPGLossConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
td_coefficient: float = 1.0
normalize_targets: bool = False
```

<a id="maddpglossconfig-__post_init__"></a>

## MADDPGLossConfig.__post_init__

[源码位置](../../marl/algorithms/maddpg.py#L47) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

```text
检查 TD 损失系数为非负；不在此处构造网络。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    None；非法配置在构造阶段抛 ValueError。
```

<a id="maddpgconfig"></a>

## MADDPGConfig

[源码位置](../../marl/algorithms/maddpg.py#L60) · [页内目录](#符号目录)

`class MADDPGConfig()`

MADDPG 配置；环境尺寸由 spec 注入。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
schema_version: Literal[2] = 2
algorithm: Literal['maddpg'] = 'maddpg'
policy: Buildable[PolicyTopology] = IndependentDeterministicConfig()
critic: Buildable[QEnsemble] = IndependentQConfig()
loss: MADDPGLossConfig = MADDPGLossConfig()
value_target: TD0Config = TD0Config()
replay: ReplayConfig = ReplayConfig()
update: ActorCriticUpdateConfig = ActorCriticUpdateConfig()
target_update: SoftTargetConfig | HardTargetConfig = SoftTargetConfig()
```

<a id="maddpgconfig-validate"></a>

## MADDPGConfig.validate

[源码位置](../../marl/algorithms/maddpg.py#L74) · [页内目录](#符号目录)

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

<a id="maddpgconfig-build"></a>

## MADDPGConfig.build

[源码位置](../../marl/algorithms/maddpg.py#L88) · [页内目录](#符号目录)

```python
def build(self, spec: EnvironmentSpec) -> MADDPG
```

```text
以当前配置和环境尺寸直接构造 MADDPG。

Args:
    spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。

Returns:
    新算法实例；optimizer/trainer 由 build_experiment 另外装配。
```

<a id="maddpg"></a>

## MADDPG

[源码位置](../../marl/algorithms/maddpg.py#L100) · [页内目录](#符号目录)

`class MADDPG(BaseMARLAlgorithm)`

每个智能体有独立 Actor 和集中式 Critic 的 MADDPG。

Actor_i 只读取 o_i，输出自己的连续动作 a_i。Critic_i 在训练时读取联合
(o_1:N, a_1:N)，预测的是智能体 i 的 Q_i。其 Bellman 标签必须使用 r_i，
不能把不同家庭/玩家的奖励平均。所有 Actor 的输入输出维度目前要求相同，
但网络参数独立，可学习不同的博弈策略。

<a id="maddpg-__init__"></a>

## MADDPG.__init__

[源码位置](../../marl/algorithms/maddpg.py#L108) · [页内目录](#符号目录)

```python
def __init__(self, spec: EnvironmentSpec, config: MADDPGConfig) -> None
```

```text
校验配置并装配 MADDPG 的在线网络、数学目标及所需 target/统计模块。

Args:
    spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。
    config: 当前算法的冻结配置，提供网络、数学目标和更新超参数。

Returns:
    None；可训练参数和 buffers 注册到当前 nn.Module。
```

<a id="maddpg-act"></a>

## MADDPG.act

[源码位置](../../marl/algorithms/maddpg.py#L130) · [页内目录](#符号目录)

```python
def act(self, observations: Tensor, *, deterministic: bool=False, action_mask: Tensor | None=None, **kwargs: Tensor) -> Tensor
```

```text
将各智能体观测映射为联合动作。始终返回确定性动作，探索噪声应由采集器加入。

独立 Actor 并行执行；DDPG 探索噪声由采样器加在返回动作上。

Args:
    observations: 浮点观测 [...,N,O]；循环单步为 [B,N,O]，序列为 [B,T,N,O]。
    deterministic: 是否采用确定性动作；具体算法的例外见本函数说明。
    action_mask: 可选 bool [...,N,A]，True 表示合法；连续动作必须为 None。
    kwargs: 保留的接口扩展关键字；当前具体算法 act 不读取这些值。

Returns:
    动作 Tensor：离散 [...,N]，连续 [...,N,A]；此接口不返回 log-prob/value。
```

<a id="maddpg-compute_loss_bundle"></a>

## MADDPG.compute_loss_bundle

[源码位置](../../marl/algorithms/maddpg.py#L153) · [页内目录](#符号目录)

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

<a id="maddpg-_validate_training_batch"></a>

## MADDPG._validate_training_batch

[源码位置](../../marl/algorithms/maddpg.py#L168) · [页内目录](#符号目录)

```python
def _validate_training_batch(self, batch: MARLBatch) -> None
```

```text
校验公共 batch 形状以及 MADDPG 所需的 actions/rewards/next_observations。

Args:
    batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。

Returns:
    None；字段缺失或形状不符抛 ValueError。
```

<a id="maddpg-compute_critic_loss_bundle"></a>

## MADDPG.compute_critic_loss_bundle

[源码位置](../../marl/algorithms/maddpg.py#L181) · [页内目录](#符号目录)

```python
def compute_critic_loss_bundle(self, batch: MARLBatch, *, update_statistics: bool=False) -> LossBundle
```

```text
使用 replay 联合动作拟合逐智能体 TD target；下一值来自target actor 与独立集中式 Q。

使用 replay 联合动作更新全部独立 centralized critics。

Args:
    batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。
    update_statistics: 是否更新逐智能体 target 尺度统计；诊断调用默认 False。

Returns:
    critic LossBundle；target 无梯度，只有 terminated 阻止 bootstrap；可选更新尺度统计。
```

<a id="maddpg-compute_actor_loss_bundle"></a>

## MADDPG.compute_actor_loss_bundle

[源码位置](../../marl/algorithms/maddpg.py#L220) · [页内目录](#符号目录)

```python
def compute_actor_loss_bundle(self, batch: MARLBatch) -> LossBundle
```

```text
冻结 critic 参数并隔离其他 actor 动作，构建每个智能体自己的策略目标。

固定每个 Q_i 及其他 actor 动作，只更新对应 Actor_i。

Args:
    batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。

Returns:
    actor LossBundle；只构图，不执行 optimizer.step。
```

<a id="maddpg-update"></a>

## MADDPG.update

[源码位置](../../marl/algorithms/maddpg.py#L252) · [页内目录](#符号目录)

```python
def update(self, batch: MARLBatch, runtime: OptimizerRuntime) -> dict[str, float]
```

```text
对一批 replay 经验按 critic → actor → target 顺序更新参数；复用基类 optimize。

按 critic → actor → target 的 MADDPG 顺序更新。

Args:
    batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。
    runtime: 已装配的命名 optimizer、裁剪上限、目标更新器、随机生成器及计数状态。

Returns:
    命名 float 指标；runtime.sync_metrics=False 时返回 {}，之后用 flush_metrics 获取。
```

<a id="maddpg-target_pairs"></a>

## MADDPG.target_pairs

[源码位置](../../marl/algorithms/maddpg.py#L294) · [页内目录](#符号目录)

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
