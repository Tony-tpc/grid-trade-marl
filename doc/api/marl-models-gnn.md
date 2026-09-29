# marl/models/gnn.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/models/gnn.py)

模块职责见类与函数说明。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [GNNBackbone](#gnnbackbone)
- [GNNBackbone.__init__](#gnnbackbone-__init__)
- [GNNBackbone.forward](#gnnbackbone-forward)

<a id="gnnbackbone"></a>

## GNNBackbone

[源码位置](../../marl/models/gnn.py#L10) · [页内目录](#符号目录)

`class GNNBackbone(BaseBackbone)`

零额外依赖的消息传递网络。

输入 ``[..., N, input_dim]``；关键字参数 adjacency 为 ``[..., N, N]``。
图连接属于输入数据，不在此处假定通信规则。加入自环后按行归一化，用 PyTorch
matmul 聚合邻居，再拼接自身特征；这不是独立的多智能体训练算法。

<a id="gnnbackbone-__init__"></a>

## GNNBackbone.__init__

[源码位置](../../marl/models/gnn.py#L18) · [页内目录](#符号目录)

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

输入节点特征和图邻接信息，执行消息聚合，返回节点 BackboneOutput；调用方管理图语义。
