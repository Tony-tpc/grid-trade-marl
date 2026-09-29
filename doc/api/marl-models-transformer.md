# marl/models/transformer.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/models/transformer.py)

有位置编码、因果屏蔽的完整窗口 Transformer，固定 dropout=0。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [TransformerBackbone](#transformerbackbone)
- [TransformerBackbone.__init__](#transformerbackbone-__init__)
- [TransformerBackbone.forward](#transformerbackbone-forward)
- [TransformerBackboneConfig](#transformerbackboneconfig)
- [TransformerBackboneConfig.__post_init__](#transformerbackboneconfig-__post_init__)
- [TransformerBackboneConfig.build](#transformerbackboneconfig-build)

<a id="transformerbackbone"></a>

## TransformerBackbone

[源码位置](../../marl/models/transformer.py#L13) · [页内目录](#符号目录)

`class TransformerBackbone(BaseBackbone)`

[*B,W,C] -> [*B,W,D]，时间位置不与批次或智能体维混合。

<a id="transformerbackbone-__init__"></a>

## TransformerBackbone.__init__

[源码位置](../../marl/models/transformer.py#L16) · [页内目录](#符号目录)

```python
def __init__(self, input_dim: int, model_dim: int=128, num_heads: int=4, num_layers: int=2, dropout: float=0.0) -> None
```

构造 TransformerBackbone，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="transformerbackbone-forward"></a>

## TransformerBackbone.forward

[源码位置](../../marl/models/transformer.py#L35) · [页内目录](#符号目录)

```python
def forward(self, inputs: Tensor, hidden_state: RecurrentState=None, **kwargs: Tensor) -> BackboneOutput
```

可选 padding_mask [*B,W]，True 为屏蔽；不接受外部循环状态。

<a id="transformerbackboneconfig"></a>

## TransformerBackboneConfig

[源码位置](../../marl/models/transformer.py#L68) · [页内目录](#符号目录)

`class TransformerBackboneConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
kind: Literal['transformer'] = 'transformer'
model_dim: int = 128
num_heads: int = 4
num_layers: int = 2
```

<a id="transformerbackboneconfig-__post_init__"></a>

## TransformerBackboneConfig.__post_init__

[源码位置](../../marl/models/transformer.py#L74) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 TransformerBackboneConfig 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="transformerbackboneconfig-build"></a>

## TransformerBackboneConfig.build

[源码位置](../../marl/models/transformer.py#L82) · [页内目录](#符号目录)

```python
def build(self, input_dim: int) -> TransformerBackbone
```

从 TransformerBackboneConfig 的静态参数构造目标对象；输入与具体返回类型见签名，不执行训练。
