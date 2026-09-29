# marl/training/checkpoint.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/training/checkpoint.py)

trainer 共享的 checkpoint 文件与 PyTorch RNG 状态工具。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [build_trainer_checkpoint_state](#build_trainer_checkpoint_state)
- [validate_trainer_checkpoint_state](#validate_trainer_checkpoint_state)
- [capture_torch_rng_state](#capture_torch_rng_state)
- [restore_torch_rng_state](#restore_torch_rng_state)
- [save_checkpoint_state](#save_checkpoint_state)
- [load_checkpoint_state](#load_checkpoint_state)

<a id="build_trainer_checkpoint_state"></a>

## build_trainer_checkpoint_state

[源码位置](../../marl/training/checkpoint.py#L15) · [页内目录](#符号目录)

```python
def build_trainer_checkpoint_state(algorithm_state: Mapping[str, Any], config_data: Mapping[str, object] | None, optimization_state: Mapping[str, Any], *, input_spec: Mapping[str, object] | None=None) -> dict[str, Any]
```

```text
复制训练器共有的模型、配置、优化器与 Torch RNG 状态。

构造 on/off-policy trainer 共同拥有的 checkpoint 字段。

Args:
    algorithm_state: 模型 state_dict；校验函数中为可选的当前模型结构参考。
    config_data: 实验构造时生成的规范化配置快照；None 只与同为 None 的快照匹配。
    optimization_state: OptimizerRuntime.state_dict()，包含优化器和更新计数等状态。
    input_spec: 环境尺寸、历史索引、state 节点和固定邻接的数据快照。

Returns:
    schema 5 的字典；不包含环境进度，replay/NumPy RNG 由离策略训练器补充。
```

<a id="validate_trainer_checkpoint_state"></a>

## validate_trainer_checkpoint_state

[源码位置](../../marl/training/checkpoint.py#L46) · [页内目录](#符号目录)

```python
def validate_trainer_checkpoint_state(state: Mapping[str, Any], config_data: Mapping[str, object] | None, *, algorithm_state: Mapping[str, Any] | None=None, input_spec: Mapping[str, object] | None=None) -> None
```

```text
验证 schema/config；提供 algorithm_state 时额外校验参数名称、shape、dtype。

统一检查 checkpoint schema 与实验配置，禁止静默部分恢复。

Args:
    state: 待验证/恢复的 checkpoint 状态 Mapping；须来自兼容配置和 schema。
    config_data: 实验构造时生成的规范化配置快照；None 只与同为 None 的快照匹配。
    algorithm_state: 模型 state_dict；校验函数中为可选的当前模型结构参考。
    input_spec: 当前环境布局快照；必须与 checkpoint 完全相同。

Returns:
    None；不匹配抛 ValueError，不修改任何训练状态。
```

<a id="capture_torch_rng_state"></a>

## capture_torch_rng_state

[源码位置](../../marl/training/checkpoint.py#L103) · [页内目录](#符号目录)

```python
def capture_torch_rng_state() -> dict[str, object]
```

```text
复制当前 Torch CPU RNG 和可用的全部 CUDA RNG 状态。

复制 CPU/CUDA RNG 状态，避免 checkpoint 被后续采样原地影响。

Inputs:
    无显式参数；读取当前 Torch RNG。

Returns:
    包含 torch_rng、cuda_rng 的字典；无 CUDA 时 cuda_rng 为 None。
```

<a id="restore_torch_rng_state"></a>

## restore_torch_rng_state

[源码位置](../../marl/training/checkpoint.py#L125) · [页内目录](#符号目录)

```python
def restore_torch_rng_state(state: Mapping[str, Any]) -> None
```

```text
恢复保存的 Torch 随机状态，供下一次采样/更新继续使用。

恢复由 :func:`capture_torch_rng_state` 保存的 RNG 状态。

Args:
    state: 待验证/恢复的 checkpoint 状态 Mapping；须来自兼容配置和 schema。

Returns:
    None；修改进程级 Torch CPU/CUDA RNG。
```

<a id="save_checkpoint_state"></a>

## save_checkpoint_state

[源码位置](../../marl/training/checkpoint.py#L150) · [页内目录](#符号目录)

```python
def save_checkpoint_state(state: Mapping[str, Any], path: str | Path) -> None
```

```text
创建父目录并使用 torch.save 序列化完整状态。

创建父目录并保存 mapping checkpoint。

Args:
    state: 待验证/恢复的 checkpoint 状态 Mapping；须来自兼容配置和 schema。
    path: 本地 checkpoint 文件路径；保存时自动创建父目录。

Returns:
    None；写入或覆盖 path 指定文件。
```

<a id="load_checkpoint_state"></a>

## load_checkpoint_state

[源码位置](../../marl/training/checkpoint.py#L168) · [页内目录](#符号目录)

```python
def load_checkpoint_state(path: str | Path, *, map_location: str | torch.device='cpu') -> Mapping[str, Any]
```

```text
反序列化可信 checkpoint 并检查顶层为 Mapping。

加载可信的本地 checkpoint，并在 trainer 读取字段前验证顶层协议。

包含 optimizer/NumPy RNG 的完整状态需要 weights_only=False；pickle 可执行
代码，因此不得用此函数加载不可信下载文件。加载模型权重不等于完整续训。

Args:
    path: 本地 checkpoint 文件路径；保存时自动创建父目录。
    map_location: torch.load 的张量映射设备；加载后的运行设备由已构造实验决定。

Returns:
    状态 Mapping；只读文件，实际模型恢复由 trainer 完成。
```
