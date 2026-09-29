# marl/modules/encoding.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/modules/encoding.py)

集中式输入的显式装配，模型层无需认识 EnvironmentSpec。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [ObservationEncoder](#observationencoder)
- [ObservationEncoder.__init__](#observationencoder-__init__)
- [ObservationEncoder.forward](#observationencoder-forward)
- [JointActionEncoder](#jointactionencoder)
- [JointActionEncoder.__init__](#jointactionencoder-__init__)
- [JointActionEncoder.forward](#jointactionencoder-forward)
- [StateEncoder](#stateencoder)
- [StateEncoder.__init__](#stateencoder-__init__)
- [StateEncoder.forward](#stateencoder-forward)
- [state_encoder](#state_encoder)

<a id="observationencoder"></a>

## ObservationEncoder

[源码位置](../../marl/modules/encoding.py#L11) · [页内目录](#符号目录)

`class ObservationEncoder(InputEncoder)`

独立编码 N 个局部观测，再可选聚合静态图；输入输出均保留 N。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
adjacency: Tensor | None
```

<a id="observationencoder-__init__"></a>

## ObservationEncoder.__init__

[源码位置](../../marl/modules/encoding.py#L16) · [页内目录](#符号目录)

```python
def __init__(self, spec: EnvironmentSpec, encoder: EncoderConfig, graph: GNNBackboneConfig | None=None) -> None
```

输入 EnvironmentSpec、encoder 与可选 graph Config；创建 N 个独立局部编码器，可选共享图层及邻接 buffer。forward 接收 [...,N,O]，输出 [...,N,E]。

<a id="observationencoder-forward"></a>

## ObservationEncoder.forward

[源码位置](../../marl/modules/encoding.py#L36) · [页内目录](#符号目录)

```python
def forward(self, inputs: Tensor) -> Tensor
```

[*B,N,O] -> [*B,N,E]；图只读观测，不接收动作。

<a id="jointactionencoder"></a>

## JointActionEncoder

[源码位置](../../marl/modules/encoding.py#L53) · [页内目录](#符号目录)

`class JointActionEncoder(InputEncoder)`

保留 critic 的 [全部观测,全部动作] 接口，仅编码观测分支。

<a id="jointactionencoder-__init__"></a>

## JointActionEncoder.__init__

[源码位置](../../marl/modules/encoding.py#L56) · [页内目录](#符号目录)

```python
def __init__(self, spec: EnvironmentSpec, encoder: EncoderConfig, graph: GNNBackboneConfig | None) -> None
```

输入 spec、encoder 与可选 graph Config；装配联合观测编码器，保留全部动作分支。输入按全部观测后接全部动作排列，宽度 N*(O+A)，输出宽度 N*(E+A)。

<a id="jointactionencoder-forward"></a>

## JointActionEncoder.forward

[源码位置](../../marl/modules/encoding.py#L64) · [页内目录](#符号目录)

```python
def forward(self, inputs: Tensor) -> Tensor
```

[*B,N*(O+A)] -> [*B,N*(E+A)]，动作梯度直接传入 Q backbone。

<a id="stateencoder"></a>

## StateEncoder

[源码位置](../../marl/modules/encoding.py#L74) · [页内目录](#符号目录)

`class StateEncoder(InputEncoder)`

按 state_layout 取节点，编码后拼接剩余全局特征。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
node_indices: Tensor
current_indices: Tensor
adjacency: Tensor | None
```

<a id="stateencoder-__init__"></a>

## StateEncoder.__init__

[源码位置](../../marl/modules/encoding.py#L81) · [页内目录](#符号目录)

```python
def __init__(self, spec: EnvironmentSpec, encoder: EncoderConfig, graph: GNNBackboneConfig | None) -> None
```

输入 spec、encoder 与可选 graph Config；要求 state_layout，创建节点间共享的局部编码器及可选图层，注册节点/当前索引与邻接 buffer。输出宽度 N*E+G。

<a id="stateencoder-forward"></a>

## StateEncoder.forward

[源码位置](../../marl/modules/encoding.py#L101) · [页内目录](#符号目录)

```python
def forward(self, inputs: Tensor) -> Tensor
```

[*B,S] -> [*B,N*E+G]；节点与时间均来自布局，绝不猜测。

<a id="state_encoder"></a>

## state_encoder

[源码位置](../../marl/modules/encoding.py#L114) · [页内目录](#符号目录)

```python
def state_encoder(spec: EnvironmentSpec, encoder: EncoderConfig, graph: GNNBackboneConfig | None) -> InputEncoder
```

默认 identity 保留原始 state 顺序、参数初始化顺序和数值。
