# marl/core/output.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/core/output.py)

模块职责见类与函数说明。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [MARLModelOutput](#marlmodeloutput)

<a id="marlmodeloutput"></a>

## MARLModelOutput

[源码位置](../../marl/core/output.py#L12) · [页内目录](#符号目录)

`class MARLModelOutput()`

不同算法统一返回的模型结果。

``extras`` 用来容纳算法特有内容，例如 Q 值、attention 权重等
调试信息。循环状态使用 policy_state/value_state，不藏入无类型 extras。

actions 为 [...,N] 或 [...,N,A]；log_prob/entropy/values 为 [...,N]。
与单智能体 ActionHeadOutput 的区别是本类已组装智能体维，且 value/logits
并非所有算法都具备。它是前向结果，不是可直接重放的 MARLBatch。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
actions: Tensor
logits: Tensor | None = None
log_prob: Tensor | None = None
entropy: Tensor | None = None
values: Tensor | None = None
policy_state: RecurrentState = None
value_state: RecurrentState = None
extras: dict[str, Any] = field(default_factory=dict)
```
