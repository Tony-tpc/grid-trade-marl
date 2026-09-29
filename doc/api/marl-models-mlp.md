# marl/models/mlp.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/models/mlp.py)

模块职责见类与函数说明。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [MLPBackbone](#mlpbackbone)
- [MLPBackbone.__init__](#mlpbackbone-__init__)
- [MLPBackbone.forward](#mlpbackbone-forward)
- [MLPBackboneConfig](#mlpbackboneconfig)
- [MLPBackboneConfig.__post_init__](#mlpbackboneconfig-__post_init__)
- [MLPBackboneConfig.build](#mlpbackboneconfig-build)

<a id="mlpbackbone"></a>

## MLPBackbone

[源码位置](../../marl/models/mlp.py#L13) · [页内目录](#符号目录)

`class MLPBackbone(BaseBackbone)`

输入 [...,input_dim]，输出 [...,output_dim]，保留所有样本/智能体维。

hidden_dims 控制中间层，output_dim 控制最后的特征层；每层均接可选 LayerNorm
和 ReLU。MLPBackboneConfig 分别控制中间层及输出层，不隐式改变默认层宽。

<a id="mlpbackbone-__init__"></a>

## MLPBackbone.__init__

[源码位置](../../marl/models/mlp.py#L20) · [页内目录](#符号目录)

```python
def __init__(self, input_dim: int, hidden_dims: Sequence[int]=(128, 128), output_dim: int=128, layer_norm: bool=False) -> None
```

构造 MLPBackbone，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="mlpbackbone-forward"></a>

## MLPBackbone.forward

[源码位置](../../marl/models/mlp.py#L40) · [页内目录](#符号目录)

```python
def forward(self, inputs: Tensor, hidden_state: RecurrentState=None, **kwargs: Tensor) -> BackboneOutput
```

逐位置编码 [...,input_dim] 为 [...,output_dim]；返回 BackboneOutput，无循环状态。

<a id="mlpbackboneconfig"></a>

## MLPBackboneConfig

[源码位置](../../marl/models/mlp.py#L50) · [页内目录](#符号目录)

`class MLPBackboneConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
kind: Literal['mlp'] = 'mlp'
hidden_dims: tuple[int, ...] = (128, 128)
output_dim: int = 128
layer_norm: bool = False
```

<a id="mlpbackboneconfig-__post_init__"></a>

## MLPBackboneConfig.__post_init__

[源码位置](../../marl/models/mlp.py#L56) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 MLPBackboneConfig 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="mlpbackboneconfig-build"></a>

## MLPBackboneConfig.build

[源码位置](../../marl/models/mlp.py#L64) · [页内目录](#符号目录)

```python
def build(self, input_dim: int) -> MLPBackbone
```

从 MLPBackboneConfig 的静态参数构造目标对象；输入与具体返回类型见签名，不执行训练。
