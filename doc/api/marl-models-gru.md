# marl/models/gru.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/models/gru.py)

模块职责见类与函数说明。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [GRUBackbone](#grubackbone)
- [GRUBackbone.__init__](#grubackbone-__init__)
- [GRUBackbone.forward](#grubackbone-forward)
- [GRUBackbone.initial_state](#grubackbone-initial_state)
- [GRUBackboneConfig](#grubackboneconfig)
- [GRUBackboneConfig.__post_init__](#grubackboneconfig-__post_init__)
- [GRUBackboneConfig.build](#grubackboneconfig-build)

<a id="grubackbone"></a>

## GRUBackbone

[源码位置](../../marl/models/gru.py#L14) · [页内目录](#符号目录)

`class GRUBackbone(BaseBackbone)`

输入 [...,T,input_dim]，输出 [...,T,hidden_dim] 的循环特征。

PyTorch hidden 使用 [num_layers,prod(前置批维),hidden_dim]，不是 [...,T,H]。
调用者负责 reset 时清空 hidden；整条序列交给原生 GRU，不按时间步展开。

<a id="grubackbone-__init__"></a>

## GRUBackbone.__init__

[源码位置](../../marl/models/gru.py#L23) · [页内目录](#符号目录)

```python
def __init__(self, input_dim: int, hidden_dim: int=128, num_layers: int=1) -> None
```

构造 GRUBackbone，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="grubackbone-forward"></a>

## GRUBackbone.forward

[源码位置](../../marl/models/gru.py#L30) · [页内目录](#符号目录)

```python
def forward(self, inputs: Tensor, hidden_state: RecurrentState=None, **kwargs: Tensor) -> BackboneOutput
```

处理 [B,T,O] 序列（单步由 Actor 补时间维），返回 features 与 [K,B,H] 新隐藏状态。

<a id="grubackbone-initial_state"></a>

## GRUBackbone.initial_state

[源码位置](../../marl/models/gru.py#L45) · [页内目录](#符号目录)

```python
def initial_state(self, *batch_shape: int, device=None, dtype=None) -> Tensor
```

根据 batch 尺寸创建零循环状态，继承参数设备/dtype；MLP 返回 None，GRU 返回 h，LSTM 返回 (h,c)。

<a id="grubackboneconfig"></a>

## GRUBackboneConfig

[源码位置](../../marl/models/gru.py#L52) · [页内目录](#符号目录)

`class GRUBackboneConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
kind: Literal['gru'] = 'gru'
hidden_dim: int = 128
num_layers: int = 1
```

<a id="grubackboneconfig-__post_init__"></a>

## GRUBackboneConfig.__post_init__

[源码位置](../../marl/models/gru.py#L57) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 GRUBackboneConfig 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="grubackboneconfig-build"></a>

## GRUBackboneConfig.build

[源码位置](../../marl/models/gru.py#L61) · [页内目录](#符号目录)

```python
def build(self, input_dim: int) -> GRUBackbone
```

从 GRUBackboneConfig 的静态参数构造目标对象；输入与具体返回类型见签名，不执行训练。
