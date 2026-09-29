# marl/training/off_policy.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/training/off_policy.py)

通用 replay 与离策略 checkpoint 训练链；不包含具体算法更新顺序。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [OffPolicyActor](#offpolicyactor)
- [OffPolicyActor.act](#offpolicyactor-act)
- [OffPolicyAlgorithm](#offpolicyalgorithm)
- [OffPolicyAlgorithm.update](#offpolicyalgorithm-update)
- [OffPolicyAlgorithm.state_dict](#offpolicyalgorithm-state_dict)
- [OffPolicyAlgorithm.load_state_dict](#offpolicyalgorithm-load_state_dict)
- [ReplayConfig](#replayconfig)
- [ReplayConfig.__post_init__](#replayconfig-__post_init__)
- [OffPolicyUpdateConfig](#offpolicyupdateconfig)
- [OffPolicyUpdateConfig.__post_init__](#offpolicyupdateconfig-__post_init__)
- [ActorCriticUpdateConfig](#actorcriticupdateconfig)
- [ActorCriticUpdateConfig.__post_init__](#actorcriticupdateconfig-__post_init__)
- [ActorCriticUpdateConfig.resolved_actor_learning_rate](#actorcriticupdateconfig-resolved_actor_learning_rate)
- [ActorCriticUpdateConfig.resolved_critic_learning_rate](#actorcriticupdateconfig-resolved_critic_learning_rate)
- [ActorCriticUpdateConfig.resolved_actor_max_grad_norm](#actorcriticupdateconfig-resolved_actor_max_grad_norm)
- [ActorCriticUpdateConfig.resolved_critic_max_grad_norm](#actorcriticupdateconfig-resolved_critic_max_grad_norm)
- [TemperatureActorCriticUpdateConfig](#temperatureactorcriticupdateconfig)
- [TemperatureActorCriticUpdateConfig.__post_init__](#temperatureactorcriticupdateconfig-__post_init__)
- [TemperatureActorCriticUpdateConfig.resolved_temperature_learning_rate](#temperatureactorcriticupdateconfig-resolved_temperature_learning_rate)
- [TemperatureActorCriticUpdateConfig.resolved_temperature_max_grad_norm](#temperatureactorcriticupdateconfig-resolved_temperature_max_grad_norm)
- [OffPolicyCollector](#offpolicycollector)
- [OffPolicyCollector.__init__](#offpolicycollector-__init__)
- [OffPolicyCollector.rollout](#offpolicycollector-rollout)
- [OffPolicyTrainer](#offpolicytrainer)
- [OffPolicyTrainer.__init__](#offpolicytrainer-__init__)
- [OffPolicyTrainer.ready](#offpolicytrainer-ready)
- [OffPolicyTrainer.record](#offpolicytrainer-record)
- [OffPolicyTrainer.record_many](#offpolicytrainer-record_many)
- [OffPolicyTrainer.update](#offpolicytrainer-update)
- [OffPolicyTrainer.update_batch](#offpolicytrainer-update_batch)
- [OffPolicyTrainer.state_dict](#offpolicytrainer-state_dict)
- [OffPolicyTrainer.load_state_dict](#offpolicytrainer-load_state_dict)
- [OffPolicyTrainer.save_checkpoint](#offpolicytrainer-save_checkpoint)
- [OffPolicyTrainer.load_checkpoint](#offpolicytrainer-load_checkpoint)

<a id="offpolicyactor"></a>

## OffPolicyActor

[源码位置](../../marl/training/off_policy.py#L32) · [页内目录](#符号目录)

`class OffPolicyActor(Protocol)`

批量 collector 所需的最小动作能力。

<a id="offpolicyactor-act"></a>

## OffPolicyActor.act

[源码位置](../../marl/training/off_policy.py#L35) · [页内目录](#符号目录)

```python
def act(self, observations: Tensor, *, deterministic: bool=False, action_mask: Tensor | None=None) -> Tensor
```

```text
离策略采集器需要的批量动作接口；协议本身不实现网络。

Args:
    observations: 浮点观测 [...,N,O]；循环单步为 [B,N,O]，序列为 [B,T,N,O]。
    deterministic: 是否采用确定性动作；具体算法的例外见本函数说明。
    action_mask: 可选 bool [...,N,A]，True 表示合法；连续动作必须为 None。

Returns:
    离散动作 [B,N] 或连续动作 [B,N,A] Tensor。
```

<a id="offpolicyalgorithm"></a>

## OffPolicyAlgorithm

[源码位置](../../marl/training/off_policy.py#L55) · [页内目录](#符号目录)

`class OffPolicyAlgorithm(Protocol)`

trainer 实际调用的最小能力，不是供算法继承的第二层基类。

collector 单独依赖 OffPolicyActor；本协议不要求 act/parameters/target_pairs，
因为 trainer 不负责动作选择、公式或 optimizer 分配。

<a id="offpolicyalgorithm-update"></a>

## OffPolicyAlgorithm.update

[源码位置](../../marl/training/off_policy.py#L62) · [页内目录](#符号目录)

```python
def update(self, batch: MARLBatch, runtime: OptimizerRuntime) -> dict[str, float]
```

```text
声明离策略算法的一批经验更新能力；由具体算法实现 optimizer 顺序。

Args:
    batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。
    runtime: 已装配的命名 optimizer、裁剪上限、目标更新器、随机生成器及计数状态。

Returns:
    命名 float 指标字典；延迟同步模式允许返回 {}。
```

<a id="offpolicyalgorithm-state_dict"></a>

## OffPolicyAlgorithm.state_dict

[源码位置](../../marl/training/off_policy.py#L74) · [页内目录](#符号目录)

```python
def state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]
```

```text
声明与 nn.Module 兼容的模型状态导出接口。

Args:
    args: 传递给 nn.Module.state_dict 的位置参数，通常省略。
    kwargs: nn.Module.state_dict 的可选关键字，如 destination/prefix/keep_vars。

Returns:
    参数和 buffers 的 Mapping；不包含 optimizer。
```

<a id="offpolicyalgorithm-load_state_dict"></a>

## OffPolicyAlgorithm.load_state_dict

[源码位置](../../marl/training/off_policy.py#L86) · [页内目录](#符号目录)

```python
def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool=True) -> Any
```

```text
声明与 nn.Module 兼容的权重恢复能力。

Args:
    state_dict: 待恢复的模型权重与 buffers，遵循 nn.Module.load_state_dict 协议。
    strict: 是否要求模型 state_dict 的参数名称严格匹配。

Returns:
    沿用 nn.Module 的加载结果；协议不自行实现恢复。
```

<a id="replayconfig"></a>

## ReplayConfig

[源码位置](../../marl/training/off_policy.py#L100) · [页内目录](#符号目录)

`class ReplayConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
capacity: int = 100000
batch_size: int = 256
```

<a id="replayconfig-__post_init__"></a>

## ReplayConfig.__post_init__

[源码位置](../../marl/training/off_policy.py#L104) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

```text
检查 capacity/batch_size 为正且 batch_size 不超过容量。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    None；非法配置在构造阶段抛 ValueError。
```

<a id="offpolicyupdateconfig"></a>

## OffPolicyUpdateConfig

[源码位置](../../marl/training/off_policy.py#L120) · [页内目录](#符号目录)

`class OffPolicyUpdateConfig()`

单 optimizer 的静态超参数；不包含更新循环或网络实例。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
learning_rate: float = 0.0003
max_grad_norm: float | None = 10.0
amp_dtype: torch.dtype | None = None
```

<a id="offpolicyupdateconfig-__post_init__"></a>

## OffPolicyUpdateConfig.__post_init__

[源码位置](../../marl/training/off_policy.py#L127) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

```text
检查学习率、裁剪上限和可选 BF16 精度。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    None；非法配置在构造阶段抛 ValueError。
```

<a id="actorcriticupdateconfig"></a>

## ActorCriticUpdateConfig

[源码位置](../../marl/training/off_policy.py#L145) · [页内目录](#符号目录)

`class ActorCriticUpdateConfig(OffPolicyUpdateConfig)`

actor-critic 公共 optimizer 字段；专用值为空时回退兼容默认值。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
actor_learning_rate: float | None = None
critic_learning_rate: float | None = None
actor_max_grad_norm: float | None = None
critic_max_grad_norm: float | None = None
```

<a id="actorcriticupdateconfig-__post_init__"></a>

## ActorCriticUpdateConfig.__post_init__

[源码位置](../../marl/training/off_policy.py#L153) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

```text
先校验公共参数，再检查 actor/critic 专用覆盖值。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    None；非法配置在构造阶段抛 ValueError。
```

<a id="actorcriticupdateconfig-resolved_actor_learning_rate"></a>

## ActorCriticUpdateConfig.resolved_actor_learning_rate

[源码位置](../../marl/training/off_policy.py#L173) · [页内目录](#符号目录)

```python
def resolved_actor_learning_rate(self) -> float
```

```text
解析 actor 的 learning_rate；专用值为 None 时回退公共 learning_rate。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    有效学习率 float。
```

<a id="actorcriticupdateconfig-resolved_critic_learning_rate"></a>

## ActorCriticUpdateConfig.resolved_critic_learning_rate

[源码位置](../../marl/training/off_policy.py#L185) · [页内目录](#符号目录)

```python
def resolved_critic_learning_rate(self) -> float
```

```text
解析 critic 的 learning_rate；专用值为 None 时回退公共 learning_rate。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    有效学习率 float。
```

<a id="actorcriticupdateconfig-resolved_actor_max_grad_norm"></a>

## ActorCriticUpdateConfig.resolved_actor_max_grad_norm

[源码位置](../../marl/training/off_policy.py#L197) · [页内目录](#符号目录)

```python
def resolved_actor_max_grad_norm(self) -> float | None
```

```text
解析 actor 的 max_grad_norm；专用值为 None 时回退公共 max_grad_norm。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    有效裁剪上限 float 或 None；不会修改冻结配置。
```

<a id="actorcriticupdateconfig-resolved_critic_max_grad_norm"></a>

## ActorCriticUpdateConfig.resolved_critic_max_grad_norm

[源码位置](../../marl/training/off_policy.py#L211) · [页内目录](#符号目录)

```python
def resolved_critic_max_grad_norm(self) -> float | None
```

```text
解析 critic 的 max_grad_norm；专用值为 None 时回退公共 max_grad_norm。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    有效裁剪上限 float 或 None；不会修改冻结配置。
```

<a id="temperatureactorcriticupdateconfig"></a>

## TemperatureActorCriticUpdateConfig

[源码位置](../../marl/training/off_policy.py#L228) · [页内目录](#符号目录)

`class TemperatureActorCriticUpdateConfig(ActorCriticUpdateConfig)`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
temperature_learning_rate: float | None = None
temperature_max_grad_norm: float | None = None
```

<a id="temperatureactorcriticupdateconfig-__post_init__"></a>

## TemperatureActorCriticUpdateConfig.__post_init__

[源码位置](../../marl/training/off_policy.py#L232) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

```text
先校验 actor/critic 配置，再检查 temperature 覆盖值。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    None；非法配置在构造阶段抛 ValueError。
```

<a id="temperatureactorcriticupdateconfig-resolved_temperature_learning_rate"></a>

## TemperatureActorCriticUpdateConfig.resolved_temperature_learning_rate

[源码位置](../../marl/training/off_policy.py#L250) · [页内目录](#符号目录)

```python
def resolved_temperature_learning_rate(self) -> float
```

```text
解析 temperature 的 learning_rate；专用值为 None 时回退公共 learning_rate。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    有效学习率 float。
```

<a id="temperatureactorcriticupdateconfig-resolved_temperature_max_grad_norm"></a>

## TemperatureActorCriticUpdateConfig.resolved_temperature_max_grad_norm

[源码位置](../../marl/training/off_policy.py#L262) · [页内目录](#符号目录)

```python
def resolved_temperature_max_grad_norm(self) -> float | None
```

```text
解析 temperature 的 max_grad_norm；专用值为 None 时回退公共 max_grad_norm。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    有效裁剪上限 float 或 None；不会修改冻结配置。
```

<a id="offpolicycollector"></a>

## OffPolicyCollector

[源码位置](../../marl/training/off_policy.py#L278) · [页内目录](#符号目录)

`class OffPolicyCollector()`

同步向量环境 collector：每个环境时刻只执行一次批量 actor 前向。

<a id="offpolicycollector-__init__"></a>

## OffPolicyCollector.__init__

[源码位置](../../marl/training/off_policy.py#L281) · [页内目录](#符号目录)

```python
def __init__(self, environment: SyncVectorEnv, algorithm: OffPolicyActor, *, device: torch.device | str='cpu') -> None
```

```text
保存向量环境、动作算法和推理设备；不创建 replay 或 optimizer。

Args:
    environment: 已构造的同步向量环境；每个 adapter 具有相同 EnvironmentSpec。
    algorithm: 满足本训练器/采集器能力协议的算法实例；复用实验中已构造的网络。
    device: 模型前向或张量迁移使用的设备，例如 cpu 或 cuda。

Returns:
    None；初始化采集器。
```

<a id="offpolicycollector-rollout"></a>

## OffPolicyCollector.rollout

[源码位置](../../marl/training/off_policy.py#L302) · [页内目录](#符号目录)

```python
def rollout(self, seeds: Sequence[int], *, deterministic: bool=False, action_transform: Callable[[np.ndarray], np.ndarray] | None=None) -> Iterator[tuple[Transition, ...]]
```

```text
reset 全部环境后批量推理，逐时刻产出真实经验；可在环境执行前添加探索。

Args:
    seeds: 各并行环境的 reset 种子，数量必须等于环境数量。
    deterministic: 是否采用确定性动作；具体算法的例外见本函数说明。
    action_transform: 可选动作后处理：接收并返回 NumPy
        联合动作批次；用于探索并保持形状/边界。

Returns:
    迭代器，每次 yield 长度为 B 的 Transition 元组；不写入 replay，不更新参数。
```

<a id="offpolicytrainer"></a>

## OffPolicyTrainer

[源码位置](../../marl/training/off_policy.py#L349) · [页内目录](#符号目录)

`class OffPolicyTrainer()`

拥有 replay 和 optimizer 状态的通用离策略 trainer。

<a id="offpolicytrainer-__init__"></a>

## OffPolicyTrainer.__init__

[源码位置](../../marl/training/off_policy.py#L352) · [页内目录](#符号目录)

```python
def __init__(self, spec: EnvironmentSpec, algorithm: OffPolicyAlgorithm, replay: TensorReplayBuffer, replay_config: ReplayConfig, optimization: OptimizerRuntime, *, device: torch.device | str='cpu', config_data: Mapping[str, object] | None=None, rng: np.random.Generator | None=None) -> None
```

```text
绑定算法、replay 与优化器状态，并复制实验配置快照。

Args:
    spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。
    algorithm: 满足本训练器/采集器能力协议的算法实例；复用实验中已构造的网络。
    replay: 保存真实 transition 的 TensorReplayBuffer 实例。
    replay_config: 容量和采样 batch_size 的静态配置。
    optimization: 算法共享的 OptimizerRuntime，负责状态管理而不定义算法更新顺序。
    device: 模型前向或张量迁移使用的设备，例如 cpu 或 cuda。
    config_data: 实验构造时生成的规范化配置快照；None 只与同为 None 的快照匹配。
    rng: replay 抽样用 NumPy Generator；None 时创建独立生成器。

Returns:
    None；初始化训练器和 replay 抽样 RNG。
```

<a id="offpolicytrainer-ready"></a>

## OffPolicyTrainer.ready

[源码位置](../../marl/training/off_policy.py#L389) · [页内目录](#符号目录)

```python
def ready(self) -> bool
```

```text
检查 replay 是否足够采样一批；不要求缓冲区已满。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    bool：len(replay) >= batch_size。
```

<a id="offpolicytrainer-record"></a>

## OffPolicyTrainer.record

[源码位置](../../marl/training/off_policy.py#L400) · [页内目录](#符号目录)

```python
def record(self, transition: Transition) -> None
```

```text
把单条真实经验加入 replay；满容量后的覆盖由 replay 管理。

Args:
    transition: 一步经验：当前观测、实际执行动作和下一环境结果；奖励取自 next。

Returns:
    None；修改 replay。
```

<a id="offpolicytrainer-record_many"></a>

## OffPolicyTrainer.record_many

[源码位置](../../marl/training/off_policy.py#L411) · [页内目录](#符号目录)

```python
def record_many(self, transitions: Sequence[Transition]) -> None
```

```text
批量加入向量环境经验，复用 replay 的批量写入路径。

Args:
    transitions: 一组真实 Transition，通常来自同一时刻的多个并行环境。

Returns:
    None；修改 replay。
```

<a id="offpolicytrainer-update"></a>

## OffPolicyTrainer.update

[源码位置](../../marl/training/off_policy.py#L422) · [页内目录](#符号目录)

```python
def update(self) -> dict[str, float]
```

```text
检查样本量并从 replay 抽样，委派具体算法 update。

从 replay 抽取一批旧经验；只委派算法 update，不重复实现优化循环。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    命名 float 指标；样本不足抛 RuntimeError，延迟同步时返回 {}。
```

<a id="offpolicytrainer-update_batch"></a>

## OffPolicyTrainer.update_batch

[源码位置](../../marl/training/off_policy.py#L441) · [页内目录](#符号目录)

```python
def update_batch(self, batch: MARLBatch) -> dict[str, float]
```

```text
接收外部 MARLBatch 并迁移设备，跳过 replay 抽样直接委派更新。

外部经验来源的入口：跳过 replay 抽样，但复用完全相同的算法 update。

Args:
    batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。

Returns:
    命名 float 指标；修改网络/optimizer，不把外部 batch 写入 replay。
```

<a id="offpolicytrainer-state_dict"></a>

## OffPolicyTrainer.state_dict

[源码位置](../../marl/training/off_policy.py#L455) · [页内目录](#符号目录)

```python
def state_dict(self) -> dict[str, Any]
```

```text
创建完整离策略训练快照，包含模型、优化器、replay、NumPy/Torch RNG。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    schema 5 状态字典；保存网络/输入布局，不包含外部采样器和环境当前位置。
```

<a id="offpolicytrainer-load_state_dict"></a>

## OffPolicyTrainer.load_state_dict

[源码位置](../../marl/training/off_policy.py#L474) · [页内目录](#符号目录)

```python
def load_state_dict(self, state: Mapping[str, Any]) -> None
```

```text
检查 schema/config，恢复网络、优化器、replay 及 NumPy/Torch RNG。

Args:
    state: 待验证/恢复的 checkpoint 状态 Mapping；须来自兼容配置和 schema。

Returns:
    None；直接修改训练状态，不承诺损坏载荷的完整事务回滚。
```

<a id="offpolicytrainer-save_checkpoint"></a>

## OffPolicyTrainer.save_checkpoint

[源码位置](../../marl/training/off_policy.py#L495) · [页内目录](#符号目录)

```python
def save_checkpoint(self, path: str | Path) -> None
```

```text
导出当前 trainer 状态到本地文件；在完整算法更新边界调用。

Args:
    path: 本地 checkpoint 文件路径；保存时自动创建父目录。

Returns:
    None；父目录自动创建，同名文件会覆盖。
```

<a id="offpolicytrainer-load_checkpoint"></a>

## OffPolicyTrainer.load_checkpoint

[源码位置](../../marl/training/off_policy.py#L506) · [页内目录](#符号目录)

```python
def load_checkpoint(self, path: str | Path, *, map_location: str | torch.device='cpu') -> None
```

```text
读取可信本地 checkpoint 并委派 load_state_dict 恢复训练状态。

Args:
    path: 本地 checkpoint 文件路径；保存时自动创建父目录。
    map_location: torch.load 的张量映射设备；加载后的运行设备由已构造实验决定。

Returns:
    None；当前实验需使用相同配置和兼容网络，外部环境/seed 进度由调用方管理。
```
