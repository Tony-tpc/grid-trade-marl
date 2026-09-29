# marl/models/encoder.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/models/encoder.py)

无跨样本状态的输入编码器；窗口维由显式布局声明。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [InputEncoder](#inputencoder)
- [InputEncoder.__init__](#inputencoder-__init__)
- [InputEncoder.validate](#inputencoder-validate)
- [InputEncoder.forward](#inputencoder-forward)
- [IdentityEncoder](#identityencoder)
- [IdentityEncoder.__init__](#identityencoder-__init__)
- [IdentityEncoder.forward](#identityencoder-forward)
- [HistoryEncoder](#historyencoder)
- [HistoryEncoder.__init__](#historyencoder-__init__)
- [HistoryEncoder.forward](#historyencoder-forward)
- [IdentityEncoderConfig](#identityencoderconfig)
- [IdentityEncoderConfig.build](#identityencoderconfig-build)
- [HistoryEncoderConfig](#historyencoderconfig)
- [HistoryEncoderConfig.__post_init__](#historyencoderconfig-__post_init__)
- [HistoryEncoderConfig.build](#historyencoderconfig-build)

<a id="inputencoder"></a>

## InputEncoder

[源码位置](../../marl/models/encoder.py#L15) · [页内目录](#符号目录)

`class InputEncoder(nn.Module)`

接收 [*B,O]、返回 [*B,E]；不保存样本间 hidden state。

<a id="inputencoder-__init__"></a>

## InputEncoder.__init__

[源码位置](../../marl/models/encoder.py#L18) · [页内目录](#符号目录)

```python
def __init__(self, input_dim: int, output_dim: int) -> None
```

记录 input_dim/output_dim，初始化 nn.Module；不分配学习参数，不返回特征。

<a id="inputencoder-validate"></a>

## InputEncoder.validate

[源码位置](../../marl/models/encoder.py#L22) · [页内目录](#符号目录)

```python
def validate(self, inputs: Tensor) -> None
```

在网络入口验证最后一维，避免广播掩盖布局错误。

<a id="inputencoder-forward"></a>

## InputEncoder.forward

[源码位置](../../marl/models/encoder.py#L27) · [页内目录](#符号目录)

```python
def forward(self, inputs: Tensor) -> Tensor
```

子类实现 [*B,input_dim] -> [*B,output_dim]；基类不执行计算。

<a id="identityencoder"></a>

## IdentityEncoder

[源码位置](../../marl/models/encoder.py#L32) · [页内目录](#符号目录)

`class IdentityEncoder(InputEncoder)`

原样返回输入，不新增参数、不改变初始化随机数序列。

<a id="identityencoder-__init__"></a>

## IdentityEncoder.__init__

[源码位置](../../marl/models/encoder.py#L35) · [页内目录](#符号目录)

```python
def __init__(self, input_dim: int) -> None
```

输入 input_dim，建立输入输出同宽的无参数编码器。

<a id="identityencoder-forward"></a>

## IdentityEncoder.forward

[源码位置](../../marl/models/encoder.py#L38) · [页内目录](#符号目录)

```python
def forward(self, inputs: Tensor) -> Tensor
```

检查 [*B,O] 后返回同一 Tensor，无复制、参数或状态变化。

<a id="historyencoder"></a>

## HistoryEncoder

[源码位置](../../marl/models/encoder.py#L47) · [页内目录](#符号目录)

`class HistoryEncoder(InputEncoder)`

[*B,O] -> 重排 [*B,W,C] -> 最后时刻隐藏输出 -> 拼接当前特征。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
history_indices: Tensor
current_indices: Tensor
```

<a id="historyencoder-__init__"></a>

## HistoryEncoder.__init__

[源码位置](../../marl/models/encoder.py#L53) · [页内目录](#符号目录)

```python
def __init__(self, input_dim: int, layout: HistoryLayout, temporal: TemporalConfig) -> None
```

输入 input_dim、HistoryLayout 和 temporal Config；验证索引完整覆盖后创建时序网络，注册历史/当前索引 buffer，输出宽度为时序隐藏宽度加当前特征数。返回 None。

<a id="historyencoder-forward"></a>

## HistoryEncoder.forward

[源码位置](../../marl/models/encoder.py#L63) · [页内目录](#符号目录)

```python
def forward(self, inputs: Tensor) -> Tensor
```

窗口从零状态计算；梯度贯穿 W 步，批次与 replay 排列互不影响。

<a id="identityencoderconfig"></a>

## IdentityEncoderConfig

[源码位置](../../marl/models/encoder.py#L72) · [页内目录](#符号目录)

`class IdentityEncoderConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
kind: Literal['identity'] = 'identity'
```

<a id="identityencoderconfig-build"></a>

## IdentityEncoderConfig.build

[源码位置](../../marl/models/encoder.py#L75) · [页内目录](#符号目录)

```python
def build(self, input_dim: int, layout: HistoryLayout | None=None) -> IdentityEncoder
```

输入 input_dim 与可选 layout，返回新 IdentityEncoder；不使用布局，不创建参数。

<a id="historyencoderconfig"></a>

## HistoryEncoderConfig

[源码位置](../../marl/models/encoder.py#L80) · [页内目录](#符号目录)

`class HistoryEncoderConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
kind: Literal['history'] = 'history'
temporal: TemporalConfig = LSTMBackboneConfig()
```

<a id="historyencoderconfig-__post_init__"></a>

## HistoryEncoderConfig.__post_init__

[源码位置](../../marl/models/encoder.py#L84) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

验证 temporal 是 GRU/LSTM/Transformer Config；非法类型抛 TypeError，返回 None。

<a id="historyencoderconfig-build"></a>

## HistoryEncoderConfig.build

[源码位置](../../marl/models/encoder.py#L90) · [页内目录](#符号目录)

```python
def build(self, input_dim: int, layout: HistoryLayout | None=None) -> HistoryEncoder
```

输入 input_dim 和 HistoryLayout，返回新 HistoryEncoder；layout=None 报错。创建独立参数，每次 forward 从零状态处理完整窗口。
