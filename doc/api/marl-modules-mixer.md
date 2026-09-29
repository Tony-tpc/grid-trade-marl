# marl/modules/mixer.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/modules/mixer.py)

模块职责见类与函数说明。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [_hypernetwork](#_hypernetwork)
- [MixingNetwork](#mixingnetwork)
- [MixingNetwork.__call__](#mixingnetwork-__call__)
- [VDNMixer](#vdnmixer)
- [VDNMixer.forward](#vdnmixer-forward)
- [QMixer](#qmixer)
- [QMixer.__init__](#qmixer-__init__)
- [QMixer.forward](#qmixer-forward)
- [QMixerConfig](#qmixerconfig)
- [QMixerConfig.__post_init__](#qmixerconfig-__post_init__)
- [QMixerConfig.build](#qmixerconfig-build)

<a id="_hypernetwork"></a>

## _hypernetwork

[源码位置](../../marl/modules/mixer.py#L17) · [页内目录](#符号目录)

```python
def _hypernetwork(input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]) -> nn.Module
```

线性输出头；空隐藏层保持原始 Linear 参数与初始化顺序。

<a id="mixingnetwork"></a>

## MixingNetwork

[源码位置](../../marl/modules/mixer.py#L29) · [页内目录](#符号目录)

`class MixingNetwork(Protocol)`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

<a id="mixingnetwork-__call__"></a>

## MixingNetwork.__call__

[源码位置](../../marl/modules/mixer.py#L30) · [页内目录](#符号目录)

```python
def __call__(self, agent_q_values: Tensor, state: Tensor) -> Tensor
```

输入局部 Q [...,N] 与 state [...,S]，返回团队 Q [...,1]。

<a id="vdnmixer"></a>

## VDNMixer

[源码位置](../../marl/modules/mixer.py#L33) · [页内目录](#符号目录)

`class VDNMixer(nn.Module)`

Value Decomposition Network：直接求和个体 Q 值。

<a id="vdnmixer-forward"></a>

## VDNMixer.forward

[源码位置](../../marl/modules/mixer.py#L36) · [页内目录](#符号目录)

```python
def forward(self, agent_q_values: Tensor, state: Tensor | None=None) -> Tensor
```

输入 [...,N] 局部 Q，沿 N 求和返回 [...,1]；忽略可选 state。

<a id="qmixer"></a>

## QMixer

[源码位置](../../marl/modules/mixer.py#L40) · [页内目录](#符号目录)

`class QMixer(nn.Module)`

QMIX 单调 mixing network。

Hypernetwork 由全局 state 生成非负权重，从结构上保证
``d Q_tot / d Q_agent >= 0``。

<a id="qmixer-__init__"></a>

## QMixer.__init__

[源码位置](../../marl/modules/mixer.py#L47) · [页内目录](#符号目录)

```python
def __init__(self, num_agents: int, state_dim: int, mixing_dim: int=32, encoder: InputEncoder | None=None, hyper_hidden_dims: tuple[int, ...]=(), value_backbone: MLPBackboneConfig=_DEFAULT_VALUE) -> None
```

构造 QMixer，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="qmixer-forward"></a>

## QMixer.forward

[源码位置](../../marl/modules/mixer.py#L67) · [页内目录](#符号目录)

```python
def forward(self, agent_q_values: Tensor, state: Tensor) -> Tensor
```

输入 agent_q_values [...,N] 和同批维 state [...,S]，编码 state 后生成非负混合权重，返回团队 Q [...,1]；校验批维和末维，保留计算图但不更新参数。

<a id="qmixerconfig"></a>

## QMixerConfig

[源码位置](../../marl/modules/mixer.py#L86) · [页内目录](#符号目录)

`class QMixerConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
kind: Literal['qmix_mixer'] = 'qmix_mixer'
mixing_dim: int = 32
encoder: EncoderConfig = IdentityEncoderConfig()
graph: GNNBackboneConfig | None = None
hyper_hidden_dims: tuple[int, ...] = ()
value_backbone: MLPBackboneConfig = MLPBackboneConfig(hidden_dims=(), output_dim=32)
```

<a id="qmixerconfig-__post_init__"></a>

## QMixerConfig.__post_init__

[源码位置](../../marl/modules/mixer.py#L94) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 QMixerConfig 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="qmixerconfig-build"></a>

## QMixerConfig.build

[源码位置](../../marl/modules/mixer.py#L104) · [页内目录](#符号目录)

```python
def build(self, spec: EnvironmentSpec) -> QMixer
```

输入 EnvironmentSpec，创建 state 编码、权重超网络及 value 分支并返回 QMixer；局部 Q 数量来自 N，状态维度来自 S，不增加共享的全局注册状态。
