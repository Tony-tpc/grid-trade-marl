# marl/runtime.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/runtime.py)

通用执行设施：设备选择、同步环境批次和 CPU 环形 replay。

这里只处理数据存放/搬运，不构造 loss、不执行 optimizer，也不按算法名分支。
环境仍在 CPU 上运行；把策略放到 CUDA 不会自动把环境动力学也迁移到 GPU。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [resolve_device](#resolve_device)
- [environment_steps_to_tensors](#environment_steps_to_tensors)
- [SyncVectorEnv](#syncvectorenv)
- [SyncVectorEnv.__init__](#syncvectorenv-__init__)
- [SyncVectorEnv.reset](#syncvectorenv-reset)
- [SyncVectorEnv.step](#syncvectorenv-step)
- [TensorReplayBuffer](#tensorreplaybuffer)
- [TensorReplayBuffer.__init__](#tensorreplaybuffer-__init__)
- [TensorReplayBuffer.__init__.allocate](#tensorreplaybuffer-__init__-allocate)
- [TensorReplayBuffer.__len__](#tensorreplaybuffer-__len__)
- [TensorReplayBuffer.add](#tensorreplaybuffer-add)
- [TensorReplayBuffer.record_many](#tensorreplaybuffer-record_many)
- [TensorReplayBuffer.sample](#tensorreplaybuffer-sample)
- [TensorReplayBuffer.sample.take](#tensorreplaybuffer-sample-take)
- [TensorReplayBuffer.state_dict](#tensorreplaybuffer-state_dict)
- [TensorReplayBuffer.load_state_dict](#tensorreplaybuffer-load_state_dict)

<a id="resolve_device"></a>

## resolve_device

[源码位置](../../marl/runtime.py#L25) · [页内目录](#符号目录)

```python
def resolve_device(name: str='auto') -> torch.device
```

auto 只检查 CUDA 可用性，不承诺 GPU 比 CPU 快；小网络需端到端计时。

<a id="environment_steps_to_tensors"></a>

## environment_steps_to_tensors

[源码位置](../../marl/runtime.py#L38) · [页内目录](#符号目录)

```python
def environment_steps_to_tensors(steps: Sequence[EnvironmentStep], spec: EnvironmentSpec, *, device: torch.device | str) -> tuple[Tensor, Tensor, Tensor | None]
```

批量转换环境输出，供 on/off-policy collector 共用一次 actor 前向。

<a id="syncvectorenv"></a>

## SyncVectorEnv

[源码位置](../../marl/runtime.py#L72) · [页内目录](#符号目录)

`class SyncVectorEnv()`

多个 adapter 组成一次策略推理批次，step 仍在当前进程依次执行。

这不是多进程环境池，也不负责部分 reset。独立算法/seed 的多进程并发由基准
脚本的 spawn worker 管理；不要把 num_envs 误当作 CPU 核心数。

<a id="syncvectorenv-__init__"></a>

## SyncVectorEnv.__init__

[源码位置](../../marl/runtime.py#L79) · [页内目录](#符号目录)

```python
def __init__(self, adapters: Sequence[EnvironmentAdapter]) -> None
```

构造 SyncVectorEnv，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="syncvectorenv-reset"></a>

## SyncVectorEnv.reset

[源码位置](../../marl/runtime.py#L87) · [页内目录](#符号目录)

```python
def reset(self, seeds: Sequence[int]) -> list[EnvironmentStep]
```

输入每个 adapter 对应的 seed，顺序 reset，返回 EnvironmentStep 元组。

<a id="syncvectorenv-step"></a>

## SyncVectorEnv.step

[源码位置](../../marl/runtime.py#L92) · [页内目录](#符号目录)

```python
def step(self, actions: np.ndarray) -> list[EnvironmentStep]
```

输入 NumPy [B,N] 或 [B,N,A]，依次推进环境并返回结果元组；不是多进程。

<a id="tensorreplaybuffer"></a>

## TensorReplayBuffer

[源码位置](../../marl/runtime.py#L101) · [页内目录](#符号目录)

`class TensorReplayBuffer()`

固定形状 CPU 环形存储；一次 transition 是所有智能体的联合经验。

size 表示已填充数量，position 指向下次写入槽；满容量后覆盖最旧数据。
同一个抽样索引用于所有字段，保持 (o,a,r,next_o) 和智能体维对齐。
pinned memory 只是加速 CPU→CUDA 传输，不会把 replay 常驻显存。

<a id="tensorreplaybuffer-__init__"></a>

## TensorReplayBuffer.__init__

[源码位置](../../marl/runtime.py#L109) · [页内目录](#符号目录)

```python
def __init__(self, spec: EnvironmentSpec, capacity: int, *, pin_memory: bool=False) -> None
```

构造 TensorReplayBuffer，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="tensorreplaybuffer-__init__-allocate"></a>

## TensorReplayBuffer.__init__.allocate

[源码位置](../../marl/runtime.py#L121) · [页内目录](#符号目录)

```python
def allocate(*shape: int, dtype: torch.dtype) -> Tensor
```

按容量及尾部 shape 分配指定 dtype 的 CPU replay 存储，返回 Tensor。

<a id="tensorreplaybuffer-__len__"></a>

## TensorReplayBuffer.__len__

[源码位置](../../marl/runtime.py#L142) · [页内目录](#符号目录)

```python
def __len__(self) -> int
```

无输入，返回当前有效 transition 数，而非配置容量。

<a id="tensorreplaybuffer-add"></a>

## TensorReplayBuffer.add

[源码位置](../../marl/runtime.py#L145) · [页内目录](#符号目录)

```python
def add(self, transition: Transition) -> None
```

输入 Transition，写入环形存储并更新位置/大小；返回 None。

<a id="tensorreplaybuffer-record_many"></a>

## TensorReplayBuffer.record_many

[源码位置](../../marl/runtime.py#L148) · [页内目录](#符号目录)

```python
def record_many(self, transitions: Sequence[Transition]) -> None
```

按环形顺序批量写入 transition，等价于依次调用 ``add``。

<a id="tensorreplaybuffer-sample"></a>

## TensorReplayBuffer.sample

[源码位置](../../marl/runtime.py#L217) · [页内目录](#符号目录)

```python
def sample(self, batch_size: int, rng: np.random.Generator, *, device: torch.device | str='cpu') -> MARLBatch
```

输入 batch_size、NumPy RNG 和目标 device，随机抽样并返回 MARLBatch，不消费存储。

<a id="tensorreplaybuffer-sample-take"></a>

## TensorReplayBuffer.sample.take

[源码位置](../../marl/runtime.py#L234) · [页内目录](#符号目录)

```python
def take(values: Tensor) -> Tensor
```

选择已抽中的 replay 索引并迁移到目标设备，返回一个字段的批 Tensor。

<a id="tensorreplaybuffer-state_dict"></a>

## TensorReplayBuffer.state_dict

[源码位置](../../marl/runtime.py#L267) · [页内目录](#符号目录)

```python
def state_dict(self) -> dict[str, object]
```

保存完整环形缓冲区，以便 checkpoint 后精确继续离策略训练。

<a id="tensorreplaybuffer-load_state_dict"></a>

## TensorReplayBuffer.load_state_dict

[源码位置](../../marl/runtime.py#L293) · [页内目录](#符号目录)

```python
def load_state_dict(self, state: dict[str, object]) -> None
```

输入 replay 状态快照，校验 spec/capacity 与存储数据后恢复位置、大小和张量；返回 None。
