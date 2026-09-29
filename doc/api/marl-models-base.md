# marl/models/base.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/models/base.py)

模块职责见类与函数说明。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [BackboneOutput](#backboneoutput)
- [BaseBackbone](#basebackbone)
- [BaseBackbone.__init__](#basebackbone-__init__)
- [BaseBackbone._validate_input](#basebackbone-_validate_input)
- [BaseBackbone.forward](#basebackbone-forward)
- [BaseBackbone.initial_state](#basebackbone-initial_state)

<a id="backboneoutput"></a>

## BackboneOutput

[源码位置](../../marl/models/base.py#L13) · [页内目录](#符号目录)

`class BackboneOutput()`

骨干网络的统一输出；循环状态和调试信息均为可选。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
features: Tensor
hidden_state: RecurrentState = None
extras: dict[str, Tensor] | None = None
```

<a id="basebackbone"></a>

## BaseBackbone

[源码位置](../../marl/models/base.py#L21) · [页内目录](#符号目录)

`class BaseBackbone(nn.Module, ABC)`

MLP/GRU/LSTM/GNN/Transformer 的共同接口。

Backbone 只做表示学习，不输出动作、Q 值或 loss，因此算法层可以独立替换网络。
这里统一的是特征输出，不保证不同 backbone 的输入布局可直接互换：GRU/LSTM 需要
时间维，GNN 需要图结构，Transformer 的序列语义也必须由使用者显式指定。

<a id="basebackbone-__init__"></a>

## BaseBackbone.__init__

[源码位置](../../marl/models/base.py#L31) · [页内目录](#符号目录)

```python
def __init__(self, input_dim: int, output_dim: int) -> None
```

构造 BaseBackbone，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="basebackbone-_validate_input"></a>

## BaseBackbone._validate_input

[源码位置](../../marl/models/base.py#L37) · [页内目录](#符号目录)

```python
def _validate_input(self, inputs: Tensor) -> None
```

检查输入张量尺寸、mask 或循环布局与实例规格一致；返回 None，错误早报。

<a id="basebackbone-forward"></a>

## BaseBackbone.forward

[源码位置](../../marl/models/base.py#L42) · [页内目录](#符号目录)

```python
def forward(self, inputs: Tensor, hidden_state: RecurrentState=None, **kwargs: Tensor) -> BackboneOutput
```

把输入映射到 ``output_dim`` 维特征。

<a id="basebackbone-initial_state"></a>

## BaseBackbone.initial_state

[源码位置](../../marl/models/base.py#L47) · [页内目录](#符号目录)

```python
def initial_state(self, *batch_shape: int, device: torch.device | str | None=None, dtype: torch.dtype | None=None) -> RecurrentState
```

无状态网络无需 hidden；循环网络默认继承参数的设备和 dtype。
