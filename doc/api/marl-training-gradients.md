# marl/training/gradients.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/training/gradients.py)

训练目标共享的梯度边界工具。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [isolate_agent_action](#isolate_agent_action)
- [frozen_parameters](#frozen_parameters)

<a id="isolate_agent_action"></a>

## isolate_agent_action

[源码位置](../../marl/training/gradients.py#L12) · [页内目录](#符号目录)

```python
def isolate_agent_action(actions: Tensor, agent_index: int) -> Tensor
```

```text
固定其他智能体动作，只保留目标智能体动作到 actor 的梯度。

只保留 ``agent_index`` 动作的梯度，其他智能体动作视作常量。

Args:
    actions: 连续联合动作 Tensor [...,N,A]，至少包含智能体维和动作维。
    agent_index: 目标智能体的零起始索引，必须位于 [0,N)。

Returns:
    数值与输入相同的 Tensor [...,N,A]，只有选中智能体切片保留梯度。
```

<a id="frozen_parameters"></a>

## frozen_parameters

[源码位置](../../marl/training/gradients.py#L42) · [页内目录](#符号目录)

```python
def frozen_parameters(module: nn.Module) -> Iterator[None]
```

```text
临时关闭模块参数梯度，保留输出对输入（例如 actor 动作）的可导性。

临时冻结模块参数，同时保留输出对输入的梯度。

Args:
    module: 暂时冻结参数的 nn.Module；退出上下文时逐参数恢复 requires_grad。

Returns:
    上下文管理器，yield None；正常或异常退出均恢复原 requires_grad。
```
