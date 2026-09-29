# marl/models/lstm.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/models/lstm.py)

原生 LSTM 的形状适配；循环门与反向传播交给 PyTorch。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [LSTMBackbone](#lstmbackbone)
- [LSTMBackbone.__init__](#lstmbackbone-__init__)
- [LSTMBackbone.forward](#lstmbackbone-forward)
- [LSTMBackbone.initial_state](#lstmbackbone-initial_state)
- [LSTMBackboneConfig](#lstmbackboneconfig)
- [LSTMBackboneConfig.__post_init__](#lstmbackboneconfig-__post_init__)
- [LSTMBackboneConfig.build](#lstmbackboneconfig-build)

<a id="lstmbackbone"></a>

## LSTMBackbone

[源码位置](../../marl/models/lstm.py#L16) · [页内目录](#符号目录)

`class LSTMBackbone(BaseBackbone)`

输入 [...,T,O] → [...,T,H]；h/c 均为 [K,prod(...),H]。

batch_first 不改变 h/c 顺序。单向、无 projection/dropout，防止未来信息泄漏，
并使给定观测序列和初态的 old log-prob 可以被 PPO 确定性重算。

<a id="lstmbackbone-__init__"></a>

## LSTMBackbone.__init__

[源码位置](../../marl/models/lstm.py#L25) · [页内目录](#符号目录)

```python
def __init__(self, input_dim: int, hidden_dim: int=128, num_layers: int=1) -> None
```

构造 LSTMBackbone，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="lstmbackbone-forward"></a>

## LSTMBackbone.forward

[源码位置](../../marl/models/lstm.py#L32) · [页内目录](#符号目录)

```python
def forward(self, inputs: Tensor, hidden_state: RecurrentState=None, **kwargs: Tensor) -> BackboneOutput
```

处理 [B,T,O] 序列（单步由 Actor 补时间维），返回 features 与新 (h,c)，两者 [K,B,H]。

<a id="lstmbackbone-initial_state"></a>

## LSTMBackbone.initial_state

[源码位置](../../marl/models/lstm.py#L46) · [页内目录](#符号目录)

```python
def initial_state(self, *batch_shape: int, device=None, dtype=None) -> tuple[Tensor, Tensor]
```

根据 batch 尺寸创建零循环状态，继承参数设备/dtype；MLP 返回 None，GRU 返回 h，LSTM 返回 (h,c)。

<a id="lstmbackboneconfig"></a>

## LSTMBackboneConfig

[源码位置](../../marl/models/lstm.py#L54) · [页内目录](#符号目录)

`class LSTMBackboneConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
kind: Literal['lstm'] = 'lstm'
hidden_dim: int = 128
num_layers: int = 1
```

<a id="lstmbackboneconfig-__post_init__"></a>

## LSTMBackboneConfig.__post_init__

[源码位置](../../marl/models/lstm.py#L59) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 LSTMBackboneConfig 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="lstmbackboneconfig-build"></a>

## LSTMBackboneConfig.build

[源码位置](../../marl/models/lstm.py#L63) · [页内目录](#符号目录)

```python
def build(self, input_dim: int) -> LSTMBackbone
```

从 LSTMBackboneConfig 的静态参数构造目标对象；输入与具体返回类型见签名，不执行训练。
