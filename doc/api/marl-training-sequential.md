# marl/training/sequential.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/training/sequential.py)

顺序角色 rollout 采集与完整周期恢复；不包含算法名称分支或市场规则。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [SequentialRollout](#sequentialrollout)
- [SequentialLearner](#sequentiallearner)
- [SequentialLearner.leader](#sequentiallearner-leader)
- [SequentialLearner.coordinator](#sequentiallearner-coordinator)
- [SequentialLearner.followers](#sequentiallearner-followers)
- [SequentialLearner.update](#sequentiallearner-update)
- [SequentialLearner.state_dict](#sequentiallearner-state_dict)
- [SequentialLearner.load_state_dict](#sequentiallearner-load_state_dict)
- [SequentialTrainer](#sequentialtrainer)
- [SequentialTrainer.__init__](#sequentialtrainer-__init__)
- [SequentialTrainer.collect](#sequentialtrainer-collect)
- [SequentialTrainer._collect_episode](#sequentialtrainer-_collect_episode)
- [SequentialTrainer.train_cycle](#sequentialtrainer-train_cycle)
- [SequentialTrainer.state_dict](#sequentialtrainer-state_dict)
- [SequentialTrainer.load_state_dict](#sequentialtrainer-load_state_dict)
- [SequentialTrainer.load_state_dict.restore](#sequentialtrainer-load_state_dict-restore)
- [SequentialTrainer.save_checkpoint](#sequentialtrainer-save_checkpoint)
- [SequentialTrainer.load_checkpoint](#sequentialtrainer-load_checkpoint)

<a id="sequentialrollout"></a>

## SequentialRollout

[源码位置](../../marl/training/sequential.py#L30) · [页内目录](#符号目录)

`class SequentialRollout()`

三个角色的 fresh 数据、生成时策略版本、阶段和真实物理步数。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
leader: PreparedRollout
coordinator: PreparedRollout
followers: PreparedRollout
phase: Phase
versions: tuple[int, int, int]
physical_steps: int
diagnostics: dict[str, float] = field(default_factory=dict)
```

<a id="sequentiallearner"></a>

## SequentialLearner

[源码位置](../../marl/training/sequential.py#L41) · [页内目录](#符号目录)

`class SequentialLearner(Protocol)`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
sequential_spec: SequentialSpec
versions: tuple[int, int, int]
next_phase: Phase
```

<a id="sequentiallearner-leader"></a>

## SequentialLearner.leader

[源码位置](../../marl/training/sequential.py#L43) · [页内目录](#符号目录)

```python
def leader(self) -> OnPolicyActorCritic
```

```text
负责先行动作采样及 value 的角色模型。

Inputs:
    无显式参数；读取当前实例的配置或训练状态。

Returns:
    负责先行动作采样及 value 的角色模型。
```

<a id="sequentiallearner-coordinator"></a>

## SequentialLearner.coordinator

[源码位置](../../marl/training/sequential.py#L55) · [页内目录](#符号目录)

```python
def coordinator(self) -> OnPolicyActorCritic
```

```text
负责共享随机广播计划及聚合消费者 value 的角色模型。

Inputs:
    无显式参数；读取当前实例的配置或训练状态。

Returns:
    负责共享随机广播计划及聚合消费者 value 的角色模型。
```

<a id="sequentiallearner-followers"></a>

## SequentialLearner.followers

[源码位置](../../marl/training/sequential.py#L67) · [页内目录](#符号目录)

```python
def followers(self) -> OnPolicyActorCritic
```

```text
具有独立 decoder 和逐消费者 value 输出的角色模型。

Inputs:
    无显式参数；读取当前实例的配置或训练状态。

Returns:
    具有独立 decoder 和逐消费者 value 输出的角色模型。
```

<a id="sequentiallearner-update"></a>

## SequentialLearner.update

[源码位置](../../marl/training/sequential.py#L81) · [页内目录](#符号目录)

```python
def update(self, rollout: SequentialRollout, runtimes: tuple[OptimizerRuntime, OptimizerRuntime, OptimizerRuntime]) -> dict[str, float]
```

```text
命名 float 指标。

Args:
    rollout: 包含三个角色 fresh PreparedRollout、阶段和策略版本的 SequentialRollout。
    runtimes: leader/coordinator/followers 的三个 OptimizerRuntime，参数集合互不重叠。

Returns:
    命名 float 指标；成功消费新鲜阶段数据并推进策略版本/下一阶段。
```

<a id="sequentiallearner-state_dict"></a>

## SequentialLearner.state_dict

[源码位置](../../marl/training/sequential.py#L95) · [页内目录](#符号目录)

```python
def state_dict(self) -> Mapping[str, Any]
```

```text
声明网络参数导出能力；完整训练快照由 trainer 另行组装。

Inputs:
    无显式参数；读取当前实例的配置或训练状态。

Returns:
    模型参数和 buffers 的映射，遵循 nn.Module.state_dict 契约。
```

<a id="sequentiallearner-load_state_dict"></a>

## SequentialLearner.load_state_dict

[源码位置](../../marl/training/sequential.py#L106) · [页内目录](#符号目录)

```python
def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool=True) -> Any
```

```text
None/模块加载结果。

Args:
    state_dict: 与当前网络键名和形状一致的模型参数映射。
    strict: 是否要求完整模型键名匹配。

Returns:
    None/模块加载结果；失败时 trainer 回滚模型、optimizer 和 RNG。
```

<a id="sequentialtrainer"></a>

## SequentialTrainer

[源码位置](../../marl/training/sequential.py#L119) · [页内目录](#符号目录)

`class SequentialTrainer()`

每周期 DS→更新→丢弃→DN→更新；采样和 GAE 复用 RolloutBuffer。

<a id="sequentialtrainer-__init__"></a>

## SequentialTrainer.__init__

[源码位置](../../marl/training/sequential.py#L121) · [页内目录](#符号目录)

```python
def __init__(self, environment: SequentialEnvironment, algorithm: SequentialLearner, estimator: AdvantageEstimator, runtimes: tuple[OptimizerRuntime, OptimizerRuntime, OptimizerRuntime], *, device: str | torch.device='cpu', leader_trajectories: int=1, config_data: Mapping[str, object] | None=None, seed: int=0)
```

```text
绑定并验证本实例所负责的组件与状态。

Args:
    environment: 满足 SequentialEnvironment 的环境；commit 不推进时间，settle 推进一次。
    algorithm: 满足 SequentialLearner 的角色模型与阶段更新接口。
    estimator: 共享的逐 agent GAE 估计器；三个角色的 GAE 配置必须一致。
    runtimes: leader/coordinator/followers 的三个 OptimizerRuntime，参数集合互不重叠。
    device: 采样/更新设备；rollout 原始物理数据保存于 CPU。
    leader_trajectories: DS 在相同外部 seed 下采集的独立动作轨迹数，正整数。
    config_data: 配置数据快照，恢复时必须严格相等；不会动态导入代码。
    seed: NumPy 日窗口/阶段采样随机种子；不改变已构造模型参数。

Returns:
    None；绑定组件/状态，不开始环境交互或参数更新。
```

<a id="sequentialtrainer-collect"></a>

## SequentialTrainer.collect

[源码位置](../../marl/training/sequential.py#L156) · [页内目录](#符号目录)

```python
def collect(self, phase: Phase, seed: int) -> SequentialRollout
```

```text
冻结策略采集 DS 多条或 DN 单条 fresh 轨迹，沿 B 合并，不拼接时间。

Args:
    phase: 当前待更新的 DS/leader 或 DN/followers 阶段。
    seed: 本批外部场景 seed；各轨迹复用场景，策略 RNG 连续前进。
        环境 reset 不应重置全局策略 RNG。各条轨迹独立重置 h/c。

Returns:
    带统一策略版本和单次消费权的 SequentialRollout；记录全部实际采样步数。
```

<a id="sequentialtrainer-_collect_episode"></a>

## SequentialTrainer._collect_episode

[源码位置](../../marl/training/sequential.py#L180) · [页内目录](#符号目录)

```python
def _collect_episode(self, phase: Phase, seed: int) -> SequentialRollout
```

```text
带版本和单次消费权的 SequentialRollout。

Args:
    phase: leader 对应 DS，followers 对应 DN；必须匹配当前待更新阶段。
    seed: NumPy 日窗口/阶段采样随机种子；不改变已构造模型参数。

Returns:
    带版本和单次消费权的 SequentialRollout；物理步数增加，参数保持不变。
```

<a id="sequentialtrainer-train_cycle"></a>

## SequentialTrainer.train_cycle

[源码位置](../../marl/training/sequential.py#L253) · [页内目录](#符号目录)

```python
def train_cycle(self) -> dict[str, float]
```

```text
DS 与 DN 的独立诊断。

Inputs:
    无显式参数；读取当前实例的配置或训练状态。

Returns:
    DS 与 DN 的独立诊断；周期数增加一次，所有采样步计入总预算。
```

<a id="sequentialtrainer-state_dict"></a>

## SequentialTrainer.state_dict

[源码位置](../../marl/training/sequential.py#L274) · [页内目录](#符号目录)

```python
def state_dict(self) -> dict[str, Any]
```

```text
完整周期 schema 5 快照。

Inputs:
    无显式参数；读取当前实例的配置或训练状态。

Returns:
    完整周期 schema 5 快照；阶段中间调用会拒绝，返回张量已复制。
```

<a id="sequentialtrainer-load_state_dict"></a>

## SequentialTrainer.load_state_dict

[源码位置](../../marl/training/sequential.py#L295) · [页内目录](#符号目录)

```python
def load_state_dict(self, state: Mapping[str, Any]) -> None
```

```text
None/模块加载结果。

Args:
    state: 完整周期 checkpoint 映射；含模型、三个优化器、策略版本、计数和 RNG。

Returns:
    None/模块加载结果；失败时 trainer 回滚模型、optimizer 和 RNG。
```

<a id="sequentialtrainer-load_state_dict-restore"></a>

## SequentialTrainer.load_state_dict.restore

[源码位置](../../marl/training/sequential.py#L318) · [页内目录](#符号目录)

```python
def restore(saved: Mapping[str, Any]) -> None
```

```text
绑定并验证本实例所负责的组件与状态。

Args:
    saved: 通过预检的 checkpoint，或异常恢复用的事务前快照。

Returns:
    None；原地恢复模型、三套 optimizer、NumPy/Torch RNG 及计数。
```

<a id="sequentialtrainer-save_checkpoint"></a>

## SequentialTrainer.save_checkpoint

[源码位置](../../marl/training/sequential.py#L341) · [页内目录](#符号目录)

```python
def save_checkpoint(self, path: str | Path) -> None
```

```text
绑定并验证本实例所负责的组件与状态。

Args:
    path: 可信本地 checkpoint 路径；保存时创建父目录。

Returns:
    None；写入包含模型、optimizer、RNG 的可信本地 checkpoint。
```

<a id="sequentialtrainer-load_checkpoint"></a>

## SequentialTrainer.load_checkpoint

[源码位置](../../marl/training/sequential.py#L352) · [页内目录](#符号目录)

```python
def load_checkpoint(self, path: str | Path) -> None
```

```text
绑定并验证本实例所负责的组件与状态。

Args:
    path: 可信本地 checkpoint 路径；保存时创建父目录。

Returns:
    None；预检配置/数据上下文后恢复完整周期状态。
```
