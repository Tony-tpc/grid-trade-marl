# marl/models/transformer.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/models/transformer.py)

模块职责见类与函数说明。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [TransformerBackbone](#transformerbackbone)
- [TransformerBackbone.__init__](#transformerbackbone-__init__)
- [TransformerBackbone.forward](#transformerbackbone-forward)

<a id="transformerbackbone"></a>

## TransformerBackbone

[源码位置](../../marl/models/transformer.py#L9) · [页内目录](#符号目录)

`class TransformerBackbone(BaseBackbone)`

输入 [B,L,O]（或无批次 [L,O]），输出同布局的 model_dim 维特征。

当前是双向 encoder，无位置编码/因果 mask；padding_mask=True 表示忽略位置，
与 action_mask=True 表示合法不同。适用于集合或已显式编码的序列，不能直接
把未来时间步送入它来实现因果策略，否则会泄漏未来信息。

<a id="transformerbackbone-__init__"></a>

## TransformerBackbone.__init__

[源码位置](../../marl/models/transformer.py#L17) · [页内目录](#符号目录)

```python
def __init__(self, input_dim: int, model_dim: int=128, num_heads: int=4, num_layers: int=2, dropout: float=0.0) -> None
```

构造 TransformerBackbone，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="transformerbackbone-forward"></a>

## TransformerBackbone.forward

[源码位置](../../marl/models/transformer.py#L39) · [页内目录](#符号目录)

```python
def forward(self, inputs: Tensor, hidden_state: RecurrentState=None, **kwargs: Tensor) -> BackboneOutput
```

编码输入序列为 BackboneOutput；不能据存在该类推断已支持因果时序训练。
