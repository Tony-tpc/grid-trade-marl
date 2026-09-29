# marl/training/optimization.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/training/optimization.py)

不含算法分支的 optimizer、AMP、计数器和 checkpoint 状态。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [metrics_to_float](#metrics_to_float)
- [OptimizerRuntime](#optimizerruntime)
- [OptimizerRuntime.__init__](#optimizerruntime-__init__)
- [OptimizerRuntime.optimizer](#optimizerruntime-optimizer)
- [OptimizerRuntime.max_grad_norm](#optimizerruntime-max_grad_norm)
- [OptimizerRuntime.parameters](#optimizerruntime-parameters)
- [OptimizerRuntime.autocast](#optimizerruntime-autocast)
- [OptimizerRuntime.record_optimizer_step](#optimizerruntime-record_optimizer_step)
- [OptimizerRuntime.accumulate_metrics](#optimizerruntime-accumulate_metrics)
- [OptimizerRuntime.mean_metrics](#optimizerruntime-mean_metrics)
- [OptimizerRuntime.export_metrics](#optimizerruntime-export_metrics)
- [OptimizerRuntime.flush_metrics](#optimizerruntime-flush_metrics)
- [OptimizerRuntime.finish](#optimizerruntime-finish)
- [OptimizerRuntime.state_dict](#optimizerruntime-state_dict)
- [OptimizerRuntime.load_state_dict](#optimizerruntime-load_state_dict)

<a id="metrics_to_float"></a>

## metrics_to_float

[源码位置](../../marl/training/optimization.py#L16) · [页内目录](#符号目录)

```python
def metrics_to_float(metrics: Mapping[str, Tensor]) -> dict[str, float]
```

```text
将标量指标打包后一次性同步到 CPU，减少逐项 GPU 同步。

把 detached 标量一次性搬回 CPU；日志边界之外避免逐项 .item() 同步。

Args:
    metrics: 名称到标量 Tensor 的映射；导出前会 detach。

Returns:
    dict[str,float]；输入为空则返回 {}。
```

<a id="optimizerruntime"></a>

## OptimizerRuntime

[源码位置](../../marl/training/optimization.py#L37) · [页内目录](#符号目录)

`class OptimizerRuntime()`

保存 optimizer、设备端统计及恢复状态，不构图、不 backward、不编排更新。

optimizer 的参数集合在构造后必须保持不变；一个参数只能属于一个 optimizer。
``update_count`` 计算法更新轮次，``optimizer_step_count`` 计真实 step 次数，
例如 PPO 一次 rollout 会包含多次 actor/critic step，二者不能互换。

<a id="optimizerruntime-__init__"></a>

## OptimizerRuntime.__init__

[源码位置](../../marl/training/optimization.py#L45) · [页内目录](#符号目录)

```python
def __init__(self, optimizers: Mapping[str, torch.optim.Optimizer], max_grad_norms: Mapping[str, float | None], *, amp_dtype: torch.dtype | None=None, target_update: TargetUpdate | None=None, generator: torch.Generator | None=None, sync_metrics: bool=True) -> None
```

```text
验证 optimizer 名称和参数归属，缓存参数元组并初始化计数/日志状态。

Args:
    optimizers: 名称到 optimizer 的非空映射；各 optimizer 不得拥有重复参数。
    max_grad_norms: 与 optimizers 名称完全一致的裁剪上限映射；None 表示不裁剪。
    amp_dtype: None 使用普通精度；torch.bfloat16 要求支持 BF16 的 CUDA。
    target_update: 可选 hard/soft target 更新器；finish 按算法更新次数调度。
    generator: 打乱 mini-batch 顺序的 Torch Generator；设备须与 randperm 一致。
    sync_metrics: True 立即导出 CPU float；False 在设备上累计，之后用 flush_metrics 读取。

Returns:
    None；不执行 backward 或 step。
```

<a id="optimizerruntime-optimizer"></a>

## OptimizerRuntime.optimizer

[源码位置](../../marl/training/optimization.py#L104) · [页内目录](#符号目录)

```python
def optimizer(self, name: str) -> torch.optim.Optimizer
```

```text
按角色名称取回 optimizer，未知名称立即报错。

Args:
    name: 字段或 optimizer 角色名称；用于定位对应状态以及报告错误。

Returns:
    torch.optim.Optimizer；返回现有实例。
```

<a id="optimizerruntime-max_grad_norm"></a>

## OptimizerRuntime.max_grad_norm

[源码位置](../../marl/training/optimization.py#L118) · [页内目录](#符号目录)

```python
def max_grad_norm(self, name: str) -> float | None
```

```text
读取该 optimizer 的有效裁剪阈值。

Args:
    name: 字段或 optimizer 角色名称；用于定位对应状态以及报告错误。

Returns:
    float 或 None；未知名称抛 KeyError。
```

<a id="optimizerruntime-parameters"></a>

## OptimizerRuntime.parameters

[源码位置](../../marl/training/optimization.py#L131) · [页内目录](#符号目录)

```python
def parameters(self, name: str) -> tuple[nn.Parameter, ...]
```

```text
读取初始化时缓存的参数，供 optimize 限定裁剪和梯度统计范围。

返回初始化时缓存的 optimizer 参数，避免每次更新重新遍历 param_groups。

Args:
    name: 字段或 optimizer 角色名称；用于定位对应状态以及报告错误。

Returns:
    nn.Parameter 元组；不会复制参数。
```

<a id="optimizerruntime-autocast"></a>

## OptimizerRuntime.autocast

[源码位置](../../marl/training/optimization.py#L148) · [页内目录](#符号目录)

```python
def autocast(self, device: torch.device) -> AbstractContextManager[None]
```

```text
创建自动混合精度上下文，启用 BF16 前校验 CUDA 支持。

Args:
    device: 模型前向或张量迁移使用的设备，例如 cpu 或 cuda。

Returns:
    上下文管理器；退出恢复原精度上下文。
```

<a id="optimizerruntime-record_optimizer_step"></a>

## OptimizerRuntime.record_optimizer_step

[源码位置](../../marl/training/optimization.py#L167) · [页内目录](#符号目录)

```python
def record_optimizer_step(self, name: str | None=None, norm: Tensor | None=None) -> None
```

```text
记录一次真实 optimizer step，可选累加裁剪、范数峰值和非有限事件。

Args:
    name: 字段或 optimizer 角色名称；用于定位对应状态以及报告错误。
    norm: 可选的裁剪前标量梯度范数，用于记录裁剪次数与非有限事件。

Returns:
    None；总 step 计数增加，不执行优化动作。
```

<a id="optimizerruntime-accumulate_metrics"></a>

## OptimizerRuntime.accumulate_metrics

[源码位置](../../marl/training/optimization.py#L193) · [页内目录](#符号目录)

```python
def accumulate_metrics(destination: dict[str, Tensor], values: Mapping[str, Tensor], *, weight: int=1) -> None
```

```text
累积 detached 日志：普通指标加权求和、事件计数求和、峰值取最大。

累加 detached 指标：普通均值按 weight 加权，事件数求和，峰值取最大。

PPO 的普通指标以 mini-batch 样本数为权重；梯度范数以 optimizer step 为
单位，需单独用默认 weight=1 累加。后缀 _count/_abs_max/_nonfinite 的
字段从不按样本数缩放，避免把一次裁剪误计成多次事件。

Args:
    destination: 原地累加的命名张量指标字典。
    values: 本次待累加的命名标量张量指标。
    weight: 普通指标的正整数权重；事件计数与极值不乘此权重。

Returns:
    None；原地修改 destination。
```

<a id="optimizerruntime-mean_metrics"></a>

## OptimizerRuntime.mean_metrics

[源码位置](../../marl/training/optimization.py#L255) · [页内目录](#符号目录)

```python
def mean_metrics(totals: Mapping[str, Tensor], count: int) -> dict[str, Tensor]
```

```text
将普通累计值除以权重，保留计数和极值，并计算实际梯度裁剪率。

除以累计权重；计数和极值保留原义，裁剪率由真实 step 事件计算。

Args:
    totals: 累计指标；普通字段存加权和，事件字段存计数或极值。
    count: 普通指标的累计权重或次数，必须为正。

Returns:
    新的 dict[str,Tensor]；不改变 totals。
```

<a id="optimizerruntime-export_metrics"></a>

## OptimizerRuntime.export_metrics

[源码位置](../../marl/training/optimization.py#L282) · [页内目录](#符号目录)

```python
def export_metrics(self, metrics: Mapping[str, Tensor]) -> dict[str, float]
```

```text
合并本轮 step 诊断；立即同步或将结果累计到设备端。

默认立即返回 CPU 数值；关闭同步时在当前设备累计 detached 指标。

Args:
    metrics: 名称到标量 Tensor 的映射；导出前会 detach。

Returns:
    同步模式返回 dict[str,float]，延迟模式返回 {}；本轮 step 缓存被清空。
```

<a id="optimizerruntime-flush_metrics"></a>

## OptimizerRuntime.flush_metrics

[源码位置](../../marl/training/optimization.py#L303) · [页内目录](#符号目录)

```python
def flush_metrics(self) -> dict[str, float]
```

```text
导出并清空延迟累计的训练指标。

同步并清空设备端累计指标；没有待处理指标时返回空字典。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    累计轮次平均的 dict[str,float]；没有累计值时返回 {}。
```

<a id="optimizerruntime-finish"></a>

## OptimizerRuntime.finish

[源码位置](../../marl/training/optimization.py#L322) · [页内目录](#符号目录)

```python
def finish(self, target_pairs: Iterable[tuple[nn.Module, nn.Module]]=()) -> None
```

```text
结束一轮算法 update，并按 hard/soft 规则同步 target。

Args:
    target_pairs: 按 (target, online) 排列的网络对；无目标网络时为空。

Returns:
    None；update_count 加一，与 optimizer_step_count 区别开。
```

<a id="optimizerruntime-state_dict"></a>

## OptimizerRuntime.state_dict

[源码位置](../../marl/training/optimization.py#L340) · [页内目录](#符号目录)

```python
def state_dict(self) -> dict[str, Any]
```

```text
复制 optimizer、mini-batch RNG、更新计数及尚未 flush 的日志。

Inputs:
    无显式参数；读取当前实例字段。

Returns:
    schema 3 状态字典；不包含网络与环境。
```

<a id="optimizerruntime-load_state_dict"></a>

## OptimizerRuntime.load_state_dict

[源码位置](../../marl/training/optimization.py#L367) · [页内目录](#符号目录)

```python
def load_state_dict(self, state: Mapping[str, Any]) -> None
```

```text
检查 schema 3 与 optimizer 名称后恢复 optimizer、RNG、计数和日志。

Args:
    state: 待验证/恢复的 checkpoint 状态 Mapping；须来自兼容配置和 schema。

Returns:
    None；直接修改 runtime，不独立提供事务回滚。
```
