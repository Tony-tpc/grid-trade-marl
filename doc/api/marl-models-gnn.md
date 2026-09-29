# marl/models/gnn.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/models/gnn.py)

静态智能体图上的消息聚合，不处理动作或动态通信图。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [GNNBackbone](#gnnbackbone)
- [GNNBackbone.__init__](#gnnbackbone-__init__)
- [GNNBackbone.forward](#gnnbackbone-forward)
- [GNNBackboneConfig](#gnnbackboneconfig)
- [GNNBackboneConfig.__post_init__](#gnnbackboneconfig-__post_init__)
- [GNNBackboneConfig.build](#gnnbackboneconfig-build)

<a id="gnnbackbone"></a>

## GNNBackbone

[源码位置](../../marl/models/gnn.py#L13) · [页内目录](#符号目录)

`class GNNBackbone(BaseBackbone)`

输入 [*B,N,F]，逐层拼接自身特征与行归一化邻居均值，输出 [*B,N,H]。

<a id="gnnbackbone-__init__"></a>

## GNNBackbone.__init__

[源码位置](../../marl/models/gnn.py#L16) · [页内目录](#符号目录)

```python
def __init__(self, input_dim: int, hidden_dim: int=128, num_layers: int=2) -> None
```

构造 GNNBackbone，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="gnnbackbone-forward"></a>

## GNNBackbone.forward

[源码位置](../../marl/models/gnn.py#L26) · [页内目录](#符号目录)

```python
def forward(self, inputs: Tensor, hidden_state: RecurrentState=None, **kwargs: Tensor) -> BackboneOutput
```

adjacency 为固定 [N,N]；添加单位自环 A+I 后按行归一化，不修改输入。

<a id="gnnbackboneconfig"></a>

## GNNBackboneConfig

[源码位置](../../marl/models/gnn.py#L55) · [页内目录](#符号目录)

`class GNNBackboneConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
kind: Literal['gnn'] = 'gnn'
hidden_dim: int = 128
num_layers: int = 2
```

<a id="gnnbackboneconfig-__post_init__"></a>

## GNNBackboneConfig.__post_init__

[源码位置](../../marl/models/gnn.py#L60) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 GNNBackboneConfig 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="gnnbackboneconfig-build"></a>

## GNNBackboneConfig.build

[源码位置](../../marl/models/gnn.py#L64) · [页内目录](#符号目录)

```python
def build(self, input_dim: int) -> GNNBackbone
```

从 GNNBackboneConfig 的静态参数构造目标对象；输入与具体返回类型见签名，不执行训练。
