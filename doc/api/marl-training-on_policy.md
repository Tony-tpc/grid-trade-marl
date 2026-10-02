# marl/training/on_policy.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/training/on_policy.py)

通用 on-policy rollout 采集与 checkpoint；更新顺序由具体算法定义。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [OnPolicyActorCritic](#onpolicyactorcritic)
- [OnPolicyActorCritic.sample](#onpolicyactorcritic-sample)
- [OnPolicyActorCritic.values](#onpolicyactorcritic-values)
- [OnPolicyActorCritic.update](#onpolicyactorcritic-update)
- [OnPolicyActorCritic.state_dict](#onpolicyactorcritic-state_dict)
- [OnPolicyActorCritic.load_state_dict](#onpolicyactorcritic-load_state_dict)
- [_allocate](#_allocate)
- [_index_optional](#_index_optional)
- [_index_batch](#_index_batch)
- [PreparedRollout](#preparedrollout)
- [PreparedRollout.__init__](#preparedrollout-__init__)
- [PreparedRollout.__init__.check](#preparedrollout-__init__-check)
- [PreparedRollout.size](#preparedrollout-size)
- [PreparedRollout.consumed](#preparedrollout-consumed)
- [PreparedRollout.to](#preparedrollout-to)
- [PreparedRollout.to.move](#preparedrollout-to-move)
- [PreparedRollout.validate_minibatches](#preparedrollout-validate_minibatches)
- [PreparedRollout._sequences](#preparedrollout-_sequences)
- [PreparedRollout._sequences.chunks_of](#preparedrollout-_sequences-chunks_of)
- [PreparedRollout._sequences.initial](#preparedrollout-_sequences-initial)
- [PreparedRollout.minibatches](#preparedrollout-minibatches)
- [PreparedRollout.consume](#preparedrollout-consume)
- [PreparedRollout.concatenate](#preparedrollout-concatenate)
- [PreparedRollout.concatenate.cat](#preparedrollout-concatenate-cat)
- [RolloutBuffer](#rolloutbuffer)
- [RolloutBuffer.__init__](#rolloutbuffer-__init__)
- [RolloutBuffer._record_state](#rolloutbuffer-_record_state)
- [RolloutBuffer._copy](#rolloutbuffer-_copy)
- [RolloutBuffer.add](#rolloutbuffer-add)
- [RolloutBuffer.finish](#rolloutbuffer-finish)
- [RolloutBuffer.finish.layout](#rolloutbuffer-finish-layout)
- [PPOUpdateConfig](#ppoupdateconfig)
- [PPOUpdateConfig.__post_init__](#ppoupdateconfig-__post_init__)
- [PPOUpdateConfig.resolved_actor_learning_rate](#ppoupdateconfig-resolved_actor_learning_rate)
- [PPOUpdateConfig.resolved_critic_learning_rate](#ppoupdateconfig-resolved_critic_learning_rate)
- [PPOUpdateConfig.resolved_actor_max_grad_norm](#ppoupdateconfig-resolved_actor_max_grad_norm)
- [PPOUpdateConfig.resolved_critic_max_grad_norm](#ppoupdateconfig-resolved_critic_max_grad_norm)
- [OnPolicyTrainer](#onpolicytrainer)
- [OnPolicyTrainer.__init__](#onpolicytrainer-__init__)
- [OnPolicyTrainer._tensors](#onpolicytrainer-_tensors)
- [OnPolicyTrainer.collect](#onpolicytrainer-collect)
- [OnPolicyTrainer.train_rollout](#onpolicytrainer-train_rollout)
- [OnPolicyTrainer.state_dict](#onpolicytrainer-state_dict)
- [OnPolicyTrainer.load_state_dict](#onpolicytrainer-load_state_dict)
- [OnPolicyTrainer.save_checkpoint](#onpolicytrainer-save_checkpoint)
- [OnPolicyTrainer.load_checkpoint](#onpolicytrainer-load_checkpoint)
- [RolloutConfig](#rolloutconfig)
- [RolloutConfig.__post_init__](#rolloutconfig-__post_init__)

<a id="onpolicyactorcritic"></a>

## OnPolicyActorCritic

[源码位置](../../marl/training/on_policy.py#L30) · [页内目录](#符号目录)

`class OnPolicyActorCritic(Protocol)`

采集和更新所需的结构化能力；不实现公式，也不要求算法继承此协议。

<a id="onpolicyactorcritic-sample"></a>

## OnPolicyActorCritic.sample

[源码位置](../../marl/training/on_policy.py#L33) · [页内目录](#符号目录)

```python
def sample(self, observations: Tensor, state: Tensor | None=None, *, deterministic: bool=False, action_mask: Tensor | None=None, policy_state: RecurrentState=None, value_state: RecurrentState=None) -> MARLModelOutput
```

```text
声明采集动作、old log-prob/value 和循环状态的能力。

Args:
    observations: 浮点观测 [...,N,O]；循环单步为 [B,N,O]，序列为 [B,T,N,O]。
    state: 可选全局 state [...,S]；None 时使用展平联合观测，尺寸必须与 critic 输入一致。
    deterministic: 是否采用确定性动作；具体算法的例外见本函数说明。
    action_mask: 可选 bool [...,N,A]，True 表示合法；连续动作必须为 None。
    policy_state: 处理当前观测之前的 actor 状态 [K,B,N,H]；LSTM 为 (h,c)，MLP 为 None。
    value_state: 处理当前观测之前的 critic 状态 [K,B,H]；LSTM 为 (h,c)，MLP 为 None。

Returns:
    MARLModelOutput；actions/log_prob/values 与新状态由具体算法提供。
```

<a id="onpolicyactorcritic-values"></a>

## OnPolicyActorCritic.values

[源码位置](../../marl/training/on_policy.py#L58) · [页内目录](#符号目录)

```python
def values(self, observations: Tensor, state: Tensor | None=None, *, value_state: RecurrentState=None) -> Tensor
```

```text
声明只计算 bootstrap value 的能力；采集器在 inference_mode 内调用。

Args:
    observations: 浮点观测 [...,N,O]；循环单步为 [B,N,O]，序列为 [B,T,N,O]。
    state: 可选全局 state [...,S]；None 时使用展平联合观测，尺寸必须与 critic 输入一致。
    value_state: 处理当前观测之前的 critic 状态 [K,B,H]；LSTM 为 (h,c)，MLP 为 None。

Returns:
    逐智能体 value Tensor [...,N]。
```

<a id="onpolicyactorcritic-update"></a>

## OnPolicyActorCritic.update

[源码位置](../../marl/training/on_policy.py#L77) · [页内目录](#符号目录)

```python
def update(self, experience: PreparedRollout, runtime: OptimizerRuntime) -> dict[str, float]
```

```text
声明消费 fresh rollout 并执行算法专用更新的能力。

Args:
    experience: 尚未消费的 PreparedRollout；携带固定
        old log-prob/value、advantage、return。
    runtime: 已装配的命名 optimizer、裁剪上限、目标更新器、随机生成器及计数状态。

Returns:
    命名 float 指标字典；延迟同步模式允许返回 {}。
```

<a id="onpolicyactorcritic-state_dict"></a>

## OnPolicyActorCritic.state_dict

[源码位置](../../marl/training/on_policy.py#L90) · [页内目录](#符号目录)

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

<a id="onpolicyactorcritic-load_state_dict"></a>

## OnPolicyActorCritic.load_state_dict

[源码位置](../../marl/training/on_policy.py#L102) · [页内目录](#符号目录)

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

<a id="_allocate"></a>

## _allocate

[源码位置](../../marl/training/on_policy.py#L115) · [页内目录](#符号目录)

```python
def _allocate(horizon: int, num_envs: int, *shape: int, dtype: torch.dtype, pin_memory: bool=False) -> Tensor
```

```text
在 CPU 预分配按时间排列的 rollout 张量；内容尚未初始化。

Args:
    horizon: 预分配/采集的时间步数 T，必须为正。
    num_envs: 并行环境数量 B，必须为正。
    dtype: 输出张量的数据类型；动作、mask 和浮点数据分别选择相应类型。
    pin_memory: 是否使用锁页 CPU 内存以支持向 CUDA 异步搬运。
    shape: 时间和环境维之后的尾部尺寸，例如 (N,O)。

Returns:
    Tensor [T,B,*shape]；读取前必须写满。
```

<a id="_index_optional"></a>

## _index_optional

[源码位置](../../marl/training/on_policy.py#L137) · [页内目录](#符号目录)

```python
def _index_optional(value: Tensor | None, indices: Tensor) -> Tensor | None
```

```text
对可选张量沿第 0 维取样，缺失字段保持缺失。

Args:
    value: 待索引、分块或复制的 Tensor；可选类型为 None 时保留缺失字段。
    indices: 沿样本/片段维选择的 long 索引，设备须与待索引张量兼容。

Returns:
    选中行组成的 Tensor，或 None。
```

<a id="_index_batch"></a>

## _index_batch

[源码位置](../../marl/training/on_policy.py#L150) · [页内目录](#符号目录)

```python
def _index_batch(batch: MARLBatch, indices: Tensor, valid_count: int | None=None) -> MARLBatch
```

```text
同步索引 MARLBatch 的数据、extras 和循环初态，避免字段错位。

Args:
    batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。
    indices: 沿样本/片段维选择的 long 索引，设备须与待索引张量兼容。
    valid_count: 本次选择中的有效时间位置数，不包含 padding，也不乘智能体数。

Returns:
    新 MARLBatch；数据沿维 0、隐藏状态沿环境/序列维 1 索引。
```

<a id="preparedrollout"></a>

## PreparedRollout

[源码位置](../../marl/training/on_policy.py#L184) · [页内目录](#符号目录)

`class PreparedRollout()`

一次性 fresh rollout：MLP 为 [T*B,N,O]；循环为 [B,T,N,O]。

完整隐藏历史为 [T,K,B,(N),H]，只用于选取片段起始状态。初态来自采样策略且
detach；片段内部按时间反传，不实施 burn-in。迁移设备会转交消费权。

<a id="preparedrollout-__init__"></a>

## PreparedRollout.__init__

[源码位置](../../marl/training/on_policy.py#L191) · [页内目录](#符号目录)

```python
def __init__(self, batch: MARLBatch, *, policy_history: RecurrentState=None, value_history: RecurrentState=None) -> None
```

```text
验证展平/序列布局与隐藏历史，授予一次性更新消费权。

Args:
    batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。
    policy_history: actor 输入状态历史 [T,K,B,N,H]；LSTM 为同形 h/c 元组。
    value_history: critic 输入状态历史 [T,K,B,H]；LSTM 为同形 h/c 元组。

Returns:
    None；保留传入张量引用，不自动复制或 detach 任意外部数据。
```

<a id="preparedrollout-__init__-check"></a>

## PreparedRollout.__init__.check

[源码位置](../../marl/training/on_policy.py#L225) · [页内目录](#符号目录)

```python
def check(tensor: Tensor, name: str=name, dimensions: int=dimensions) -> Tensor
```

```text
验证单个 h/c 历史张量的维数及 T/B/N 与 rollout 一致。

Args:
    tensor: 待校验或转换布局/设备的单个 Tensor，形状规则见函数说明。
    name: 状态所属角色 policy/value，决定是否检查智能体维。
    dimensions: 该角色期望的历史张量维数；actor 为 5，critic 为 4。

Returns:
    原 Tensor；形状不符抛 ValueError。
```

<a id="preparedrollout-size"></a>

## PreparedRollout.size

[源码位置](../../marl/training/on_policy.py#L251) · [页内目录](#符号目录)

```python
def size(self) -> int
```

```text
计算真实环境时间位置数；智能体维不算独立环境样本。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    int：展平路径为 T*B，循环路径也为 B*T。
```

<a id="preparedrollout-consumed"></a>

## PreparedRollout.consumed

[源码位置](../../marl/training/on_policy.py#L264) · [页内目录](#符号目录)

```python
def consumed(self) -> bool
```

```text
查询是否已开始 minibatch 消费或已转交设备迁移消费权。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    bool；只读属性。
```

<a id="preparedrollout-to"></a>

## PreparedRollout.to

[源码位置](../../marl/training/on_policy.py#L275) · [页内目录](#符号目录)

```python
def to(self, device: torch.device | str, *, non_blocking: bool=False) -> PreparedRollout
```

```text
迁移 batch 和全部 h/c 历史；成功后使旧对象失去消费权。

移动失败不消费；隐藏历史也一起移动，不能只搬 observations。

Args:
    device: 模型前向或张量迁移使用的设备，例如 cpu 或 cuda。
    non_blocking: 是否请求异步设备复制；实际异步效果取决于设备和 pinned memory。

Returns:
    具有消费权的新 PreparedRollout；重复迁移旧对象抛 RuntimeError。
```

<a id="preparedrollout-to-move"></a>

## PreparedRollout.to.move

[源码位置](../../marl/training/on_policy.py#L290) · [页内目录](#符号目录)

```python
def move(t: Tensor) -> Tensor
```

```text
迁移一个循环状态张量，保留其形状和 dtype。

Args:
    t: 待校验或转换布局/设备的单个 Tensor，形状规则见函数说明。

Returns:
    目标设备上的 Tensor。
```

<a id="preparedrollout-validate_minibatches"></a>

## PreparedRollout.validate_minibatches

[源码位置](../../marl/training/on_policy.py#L309) · [页内目录](#符号目录)

```python
def validate_minibatches(self, mini_batch_size: int, sequence_length: int | None) -> int
```

```text
在任何参数/统计变化前验证 batch 预算与片段长度。

在 target 统计/optimizer 变化前校验采样参数。

Args:
    mini_batch_size: 每批时间位置预算；循环训练按整片段打包，不乘 N。
    sequence_length: 循环片段长度；None 使用整个 rollout，纯 MLP 必须为 None。

Returns:
    有效片段长度 int；纯 MLP 返回 1。
```

<a id="preparedrollout-_sequences"></a>

## PreparedRollout._sequences

[源码位置](../../marl/training/on_policy.py#L336) · [页内目录](#符号目录)

```python
def _sequences(self, length: int) -> tuple[MARLBatch, list[int]]
```

```text
将每个环境轨迹切成连续片段，右补齐尾段并选取对应采样初态。

一次性右补齐和分块；各 epoch 只索引整条片段，不再次搬运原数据。

Args:
    length: 连续片段长度 L；调用前应完成 minibatch 参数校验。

Returns:
    (片段 MARLBatch, 每段有效长度列表)；观测 [B*C,L,N,O]，mask [B*C,L]。
```

<a id="preparedrollout-_sequences-chunks_of"></a>

## PreparedRollout._sequences.chunks_of

[源码位置](../../marl/training/on_policy.py#L352) · [页内目录](#符号目录)

```python
def chunks_of(value: Tensor | None, fill: int=0) -> Tensor | None
```

```text
对一个时间张量按相同顺序补齐和分块，保持所有字段对齐。

Args:
    value: 待索引、分块或复制的 Tensor；可选类型为 None 时保留缺失字段。
    fill: 右侧补齐值；动作 mask 用 1，其他数据通常用 0。

Returns:
    Tensor [B*C,L,...]，输入 None 则返回 None。
```

<a id="preparedrollout-_sequences-initial"></a>

## PreparedRollout._sequences.initial

[源码位置](../../marl/training/on_policy.py#L368) · [页内目录](#符号目录)

```python
def initial(history: Tensor) -> Tensor
```

```text
从状态历史每隔 L 步取初态，再按环境优先排列各片段。

Args:
    history: 循环状态历史；单 Tensor 为 [T,K,B,(N),H]，LSTM 使用同形 (h,c)。

Returns:
    detach 且连续的状态 [K,B*C,(N),H]。
```

<a id="preparedrollout-minibatches"></a>

## PreparedRollout.minibatches

[源码位置](../../marl/training/on_policy.py#L412) · [页内目录](#符号目录)

```python
def minibatches(self, *, epochs: int, mini_batch_size: int, generator: torch.Generator | None=None, sequence_length: int | None=None) -> Iterator[MARLBatch]
```

```text
单次消费内遍历多个 epoch；MLP 打乱样本，循环网络仅打乱完整片段。

Args:
    epochs: 同一 fresh rollout 内的 PPO 遍历次数，必须为正。
    mini_batch_size: 每批时间位置预算；循环训练按整片段打包，不乘 N。
    generator: 打乱 mini-batch 顺序的 Torch Generator；设备须与 randperm 一致。
    sequence_length: 循环片段长度；None 使用整个 rollout，纯 MLP 必须为 None。

Returns:
    MARLBatch 迭代器；首次推进即标记 consumed，禁止另开一次旧数据更新。
```

<a id="preparedrollout-consume"></a>

## PreparedRollout.consume

[源码位置](../../marl/training/on_policy.py#L454) · [页内目录](#符号目录)

```python
def consume(self) -> MARLBatch
```

```text
领取完整 fresh batch 的单次更新权，供全轨迹高阶目标使用。

Inputs:
    无显式参数；读取当前 rollout 的消费状态。

Returns:
    原始 MARLBatch；调用后 consumed=True，后续更新/迁移/领取会拒绝。
```

<a id="preparedrollout-concatenate"></a>

## PreparedRollout.concatenate

[源码位置](../../marl/training/on_policy.py#L469) · [页内目录](#符号目录)

```python
def concatenate(cls, rollouts: Sequence[PreparedRollout]) -> PreparedRollout
```

```text
合并同一策略采集的 fresh rollout，成功后转交所有输入的消费权。

Args:
    rollouts: 非空且不重复的同构 rollout；循环路径时间长度必须相等。
        调用方负责验证策略版本。GAE 已各自完成，不跨 episode 递推。

Returns:
    新 PreparedRollout；数据沿维 0 拼接，h/c 初态沿 B=1、历史沿 B=2。
    验证失败不会消费输入，old log-prob/value 和采样状态保持不变。
```

<a id="preparedrollout-concatenate-cat"></a>

## PreparedRollout.concatenate.cat

[源码位置](../../marl/training/on_policy.py#L490) · [页内目录](#符号目录)

```python
def cat(values: Sequence[Tensor | None]) -> Tensor | None
```

```text
同构可选数据张量沿第 0 维拼接。

Args:
    values: 同构 Tensor 或全部 None；混合缺失字段报错。

Returns:
    沿第 0 维拼接的 Tensor，或 None；不修改输入。
```

<a id="rolloutbuffer"></a>

## RolloutBuffer

[源码位置](../../marl/training/on_policy.py#L525) · [页内目录](#符号目录)

`class RolloutBuffer()`

预分配的固定长度 on-policy rollout；每个实例只能 finish 一次。

<a id="rolloutbuffer-__init__"></a>

## RolloutBuffer.__init__

[源码位置](../../marl/training/on_policy.py#L528) · [页内目录](#符号目录)

```python
def __init__(self, spec: EnvironmentSpec, *, horizon: int, num_envs: int, pin_memory: bool=False) -> None
```

```text
按 spec 预分配 T×B 的 CPU 存储，并将写入位置设为 0。

Args:
    spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。
    horizon: 预分配/采集的时间步数 T，必须为正。
    num_envs: 并行环境数量 B，必须为正。
    pin_memory: 是否使用锁页 CPU 内存以支持向 CUDA 异步搬运。

Returns:
    None；隐藏状态历史在首次 add 时按实际网络尺寸分配。
```

<a id="rolloutbuffer-_record_state"></a>

## RolloutBuffer._record_state

[源码位置](../../marl/training/on_policy.py#L594) · [页内目录](#符号目录)

```python
def _record_state(self, history: RecurrentState, state: RecurrentState) -> RecurrentState
```

```text
首次分配循环状态历史，随后保存处理当前观测前的 detached 状态。

Args:
    history: 循环状态历史；单 Tensor 为 [T,K,B,(N),H]，LSTM 使用同形 (h,c)。
    state: 当前输入隐藏状态；GRU 为 Tensor，LSTM 为 (h,c)，无状态网络为 None。

Returns:
    更新后的完整历史；无状态网络返回 None。
```

<a id="rolloutbuffer-_copy"></a>

## RolloutBuffer._copy

[源码位置](../../marl/training/on_policy.py#L628) · [页内目录](#符号目录)

```python
def _copy(self, target: Tensor, value: Tensor, name: str) -> None
```

```text
校验当前时间片形状并复制 detached 数据，避免采样计算图进入缓存。

Args:
    target: 预分配存储张量；本次写入 target[position]。
    value: 待索引、分块或复制的 Tensor；可选类型为 None 时保留缺失字段。
    name: 字段或 optimizer 角色名称；用于定位对应状态以及报告错误。

Returns:
    None；原地写入存储的当前时间位置。
```

<a id="rolloutbuffer-add"></a>

## RolloutBuffer.add

[源码位置](../../marl/training/on_policy.py#L644) · [页内目录](#符号目录)

```python
def add(self, *, observations: Tensor, states: Tensor, actions: Tensor, rewards: Tensor, old_log_prob: Tensor, old_values: Tensor, terminated: Tensor, truncated: Tensor, action_masks: Tensor | None=None, raw_actions: Tensor | None=None, policy_state: RecurrentState=None, value_state: RecurrentState=None) -> None
```

```text
保存一个时刻全部 B 个环境的采样数据，再推进写入位置。

Args:
    observations: 浮点观测 [...,N,O]；循环单步为 [B,N,O]，序列为 [B,T,N,O]。
    states: 集中式 critic 的全局状态浮点 [B,S]。
    actions: 联合动作；离散 long [...,N]，连续浮点 [...,N,A]。
    rewards: 本次动作产生的逐智能体奖励浮点 [B,N]。
    old_log_prob: 采样策略对实际动作的 log-prob [B,N]；保存时 detach。
    old_values: 采样时逐智能体 value [B,N]；保存后在 PPO 各 epoch 保持固定。
    terminated: bool [B,N]，真实 MDP 终止，禁止下一状态 bootstrap。
    truncated: bool [B,N]，时间截断；允许当前 bootstrap，但不跨 episode 递推 GAE。
    action_masks: 采样时的 bool [B,N,A]；离散动作必填，连续动作必须为 None。
    raw_actions: 可选 pre-tanh 浮点 [B,N,A]，连续 PPO 需要从首步一致保存。
    policy_state: 处理当前观测之前的 actor 状态 [K,B,N,H]；LSTM 为 (h,c)，MLP 为 None。
    value_state: 处理当前观测之前的 critic 状态 [K,B,H]；LSTM 为 (h,c)，MLP 为 None。

Returns:
    None；缓存已满、已 finish、mask 或字段形状不符时报错。
```

<a id="rolloutbuffer-finish"></a>

## RolloutBuffer.finish

[源码位置](../../marl/training/on_policy.py#L710) · [页内目录](#符号目录)

```python
def finish(self, next_value: Tensor, estimator: AdvantageEstimator) -> PreparedRollout
```

```text
在写满后用最终 bootstrap value 计算 GAE/return，形成一次性可训练 rollout。

Args:
    next_value: 最后一步之后、reset 之前的 value [B,N]；终止处理交给估计器。
    estimator: AdvantageEstimator，按奖励、old value 与终止标记计算 GAE/return。

Returns:
    PreparedRollout；MLP 展平为 [T*B,N,O]，循环转为 [B,T,N,O]。
```

<a id="rolloutbuffer-finish-layout"></a>

## RolloutBuffer.finish.layout

[源码位置](../../marl/training/on_policy.py#L737) · [页内目录](#符号目录)

```python
def layout(tensor: Tensor) -> Tensor
```

```text
把时间优先存储转换成模型训练所需的展平或 batch 优先布局。

Args:
    tensor: 待校验或转换布局/设备的单个 Tensor，形状规则见函数说明。

Returns:
    MLP 为 [T*B,...]，循环为 [B,T,...] 的 Tensor。
```

<a id="ppoupdateconfig"></a>

## PPOUpdateConfig

[源码位置](../../marl/training/on_policy.py#L785) · [页内目录](#符号目录)

`class PPOUpdateConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
learning_rate: float = 0.0003
actor_learning_rate: float | None = None
critic_learning_rate: float | None = None
epochs: int = 4
mini_batch_size: int = 256
max_grad_norm: float | None = 10.0
actor_max_grad_norm: float | None = None
critic_max_grad_norm: float | None = None
value_clip_ratio: float | None = None
sequence_length: int | None = None
```

<a id="ppoupdateconfig-__post_init__"></a>

## PPOUpdateConfig.__post_init__

[源码位置](../../marl/training/on_policy.py#L797) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

```text
检查学习率、epoch、batch、裁剪阈值和 sequence_length 范围。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    None；非法配置在构造阶段抛 ValueError。
```

<a id="ppoupdateconfig-resolved_actor_learning_rate"></a>

## PPOUpdateConfig.resolved_actor_learning_rate

[源码位置](../../marl/training/on_policy.py#L831) · [页内目录](#符号目录)

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

<a id="ppoupdateconfig-resolved_critic_learning_rate"></a>

## PPOUpdateConfig.resolved_critic_learning_rate

[源码位置](../../marl/training/on_policy.py#L843) · [页内目录](#符号目录)

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

<a id="ppoupdateconfig-resolved_actor_max_grad_norm"></a>

## PPOUpdateConfig.resolved_actor_max_grad_norm

[源码位置](../../marl/training/on_policy.py#L855) · [页内目录](#符号目录)

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

<a id="ppoupdateconfig-resolved_critic_max_grad_norm"></a>

## PPOUpdateConfig.resolved_critic_max_grad_norm

[源码位置](../../marl/training/on_policy.py#L869) · [页内目录](#符号目录)

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

<a id="onpolicytrainer"></a>

## OnPolicyTrainer

[源码位置](../../marl/training/on_policy.py#L885) · [页内目录](#符号目录)

`class OnPolicyTrainer()`

采集 fresh rollout，并调用具体 on-policy 算法的 update。

<a id="onpolicytrainer-__init__"></a>

## OnPolicyTrainer.__init__

[源码位置](../../marl/training/on_policy.py#L888) · [页内目录](#符号目录)

```python
def __init__(self, environment: SyncVectorEnv, algorithm: OnPolicyActorCritic, estimator: AdvantageEstimator, rollout_horizon: int, optimization: OptimizerRuntime, *, device: torch.device | str='cpu', config_data: Mapping[str, object] | None=None) -> None
```

```text
绑定环境、算法、优势估计器与优化状态，并验证 rollout horizon。

Args:
    environment: 已构造的同步向量环境；每个 adapter 具有相同 EnvironmentSpec。
    algorithm: 满足本训练器/采集器能力协议的算法实例；复用实验中已构造的网络。
    estimator: AdvantageEstimator，按奖励、old value 与终止标记计算 GAE/return。
    rollout_horizon: 每次采集步数，必须位于 [1, environment.spec.horizon]。
    optimization: 算法共享的 OptimizerRuntime，负责状态管理而不定义算法更新顺序。
    device: 模型前向或张量迁移使用的设备，例如 cpu 或 cuda。
    config_data: 实验构造时生成的规范化配置快照；None 只与同为 None 的快照匹配。

Returns:
    None；不开始环境交互。
```

<a id="onpolicytrainer-_tensors"></a>

## OnPolicyTrainer._tensors

[源码位置](../../marl/training/on_policy.py#L923) · [页内目录](#符号目录)

```python
def _tensors(self, steps: Sequence[EnvironmentStep]) -> tuple[Tensor, Tensor, Tensor | None]
```

```text
将同步环境结果打包到训练设备供策略或 bootstrap 前向。

Args:
    steps: 每个并行环境当前的 EnvironmentStep 序列。

Returns:
    (observations [B,N,O], state [B,S], action_mask [B,N,A] 或 None)。
```

<a id="onpolicytrainer-collect"></a>

## OnPolicyTrainer.collect

[源码位置](../../marl/training/on_policy.py#L938) · [页内目录](#符号目录)

```python
def collect(self, seeds: list[int]) -> PreparedRollout
```

```text
重置环境和 h/c，采集固定 T 步，并在最终观测上计算 bootstrap 后生成 rollout。

Args:
    seeds: 各并行环境的 reset 种子，数量必须等于环境数量。

Returns:
    未消费 PreparedRollout；中途任一环境结束会报错，不自动部分 reset。
```

<a id="onpolicytrainer-train_rollout"></a>

## OnPolicyTrainer.train_rollout

[源码位置](../../marl/training/on_policy.py#L1032) · [页内目录](#符号目录)

```python
def train_rollout(self, seeds: list[int]) -> dict[str, float]
```

```text
采集一份 fresh rollout，迁移到训练设备并立即委派算法更新。

Args:
    seeds: 各并行环境的 reset 种子，数量必须等于环境数量。

Returns:
    命名 float 指标；完成后旧 rollout 不可再次训练。
```

<a id="onpolicytrainer-state_dict"></a>

## OnPolicyTrainer.state_dict

[源码位置](../../marl/training/on_policy.py#L1047) · [页内目录](#符号目录)

```python
def state_dict(self) -> dict[str, Any]
```

```text
创建完整 rollout 更新边界上的训练快照。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    schema 5 状态字典；保存输入布局，不保存活跃环境、半段 rollout 或外部 seed 调度。
```

<a id="onpolicytrainer-load_state_dict"></a>

## OnPolicyTrainer.load_state_dict

[源码位置](../../marl/training/on_policy.py#L1063) · [页内目录](#符号目录)

```python
def load_state_dict(self, state: Mapping[str, Any]) -> None
```

```text
预检配置与网络结构后恢复训练；optimizer/RNG 恢复异常时回滚原状态。

Args:
    state: 待验证/恢复的 checkpoint 状态 Mapping；须来自兼容配置和 schema。

Returns:
    None；仅支持完整 rollout 更新后的 checkpoint 恢复。
```

<a id="onpolicytrainer-save_checkpoint"></a>

## OnPolicyTrainer.save_checkpoint

[源码位置](../../marl/training/on_policy.py#L1091) · [页内目录](#符号目录)

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

<a id="onpolicytrainer-load_checkpoint"></a>

## OnPolicyTrainer.load_checkpoint

[源码位置](../../marl/training/on_policy.py#L1102) · [页内目录](#符号目录)

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

<a id="rolloutconfig"></a>

## RolloutConfig

[源码位置](../../marl/training/on_policy.py#L1118) · [页内目录](#符号目录)

`class RolloutConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
horizon: int | None = None
```

<a id="rolloutconfig-__post_init__"></a>

## RolloutConfig.__post_init__

[源码位置](../../marl/training/on_policy.py#L1121) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

```text
检查 horizon 为 None 或正数。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    None；非法配置在构造阶段抛 ValueError。
```
