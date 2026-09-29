# marl/modules/critic.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/modules/critic.py)

模块职责见类与函数说明。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [centralized_critic_input](#centralized_critic_input)
- [ValueNetwork](#valuenetwork)
- [ValueNetwork.__call__](#valuenetwork-__call__)
- [ValueNetwork.parameters](#valuenetwork-parameters)
- [RecurrentValueNetwork](#recurrentvaluenetwork)
- [RecurrentValueNetwork.__call__](#recurrentvaluenetwork-__call__)
- [AttentionQNetwork](#attentionqnetwork)
- [AttentionQNetwork.__call__](#attentionqnetwork-__call__)
- [AttentionQNetwork.parameters](#attentionqnetwork-parameters)
- [QEnsemble](#qensemble)
- [QEnsemble.critics](#qensemble-critics)
- [TwinQEnsemble](#twinqensemble)
- [TwinQEnsemble.first](#twinqensemble-first)
- [TwinQEnsemble.second](#twinqensemble-second)
- [TwinQEnsemble.__call__](#twinqensemble-__call__)
- [TwinQEnsemble.parameters](#twinqensemble-parameters)
- [ValueCritic](#valuecritic)
- [ValueCritic.__init__](#valuecritic-__init__)
- [ValueCritic.forward](#valuecritic-forward)
- [CentralizedCritic](#centralizedcritic)
- [CentralizedCritic.__init__](#centralizedcritic-__init__)
- [CentralizedCritic.initial_state](#centralizedcritic-initial_state)
- [CentralizedCritic.forward](#centralizedcritic-forward)
- [IndependentCentralizedCritics](#independentcentralizedcritics)
- [IndependentCentralizedCritics.__init__](#independentcentralizedcritics-__init__)
- [IndependentCentralizedCritics.forward](#independentcentralizedcritics-forward)
- [TwinIndependentCentralizedCritics](#twinindependentcentralizedcritics)
- [TwinIndependentCentralizedCritics.__init__](#twinindependentcentralizedcritics-__init__)
- [TwinIndependentCentralizedCritics.forward](#twinindependentcentralizedcritics-forward)
- [AttentionCritic](#attentioncritic)
- [AttentionCritic.__init__](#attentioncritic-__init__)
- [AttentionCritic.forward](#attentioncritic-forward)
- [CentralizedValueConfig](#centralizedvalueconfig)
- [CentralizedValueConfig.build](#centralizedvalueconfig-build)
- [AttentionQConfig](#attentionqconfig)
- [AttentionQConfig.__post_init__](#attentionqconfig-__post_init__)
- [AttentionQConfig.build](#attentionqconfig-build)
- [IndependentQConfig](#independentqconfig)
- [IndependentQConfig.build](#independentqconfig-build)
- [TwinQConfig](#twinqconfig)
- [TwinQConfig.build](#twinqconfig-build)

<a id="centralized_critic_input"></a>

## centralized_critic_input

[源码位置](../../marl/modules/critic.py#L31) · [页内目录](#符号目录)

```python
def centralized_critic_input(observations: Tensor, actions: Tensor) -> Tensor
```

把 ``[..., N, O]`` 与 ``[..., N, A]`` 拼成 ``[..., N*O + N*A]``。

<a id="valuenetwork"></a>

## ValueNetwork

[源码位置](../../marl/modules/critic.py#L44) · [页内目录](#符号目录)

`class ValueNetwork(Protocol)`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

<a id="valuenetwork-__call__"></a>

## ValueNetwork.__call__

[源码位置](../../marl/modules/critic.py#L45) · [页内目录](#符号目录)

```python
def __call__(self, inputs: Tensor) -> Tensor
```

输入 state [...,S]，返回逐智能体 value [...,N]。

<a id="valuenetwork-parameters"></a>

## ValueNetwork.parameters

[源码位置](../../marl/modules/critic.py#L46) · [页内目录](#符号目录)

```python
def parameters(self, recurse: bool=True) -> Iterator[nn.Parameter]
```

返回可遍历的 nn.Parameter；recurse 决定是否包含子模块，用于优化器参数装配。

<a id="recurrentvaluenetwork"></a>

## RecurrentValueNetwork

[源码位置](../../marl/modules/critic.py#L50) · [页内目录](#符号目录)

`class RecurrentValueNetwork(Protocol)`

可选的状态返回能力；外部无状态 nn.Sequential 无需实现。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
is_recurrent: bool
```

<a id="recurrentvaluenetwork-__call__"></a>

## RecurrentValueNetwork.__call__

[源码位置](../../marl/modules/critic.py#L55) · [页内目录](#符号目录)

```python
def __call__(self, inputs: Tensor, *, hidden_state: RecurrentState=None, return_state: Literal[True]) -> tuple[Tensor, RecurrentState]
```

输入 state 与可选隐藏状态，按 return_state 返回 value 或 (value,新状态)。

<a id="attentionqnetwork"></a>

## AttentionQNetwork

[源码位置](../../marl/modules/critic.py#L65) · [页内目录](#符号目录)

`class AttentionQNetwork(Protocol)`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

<a id="attentionqnetwork-__call__"></a>

## AttentionQNetwork.__call__

[源码位置](../../marl/modules/critic.py#L66) · [页内目录](#符号目录)

```python
def __call__(self, observations: Tensor, actions: Tensor) -> Tensor
```

输入联合观测和动作，返回候选动作 Q [...,N,A]。

<a id="attentionqnetwork-parameters"></a>

## AttentionQNetwork.parameters

[源码位置](../../marl/modules/critic.py#L67) · [页内目录](#符号目录)

```python
def parameters(self, recurse: bool=True) -> Iterator[nn.Parameter]
```

返回可遍历的 nn.Parameter；recurse 决定是否包含子模块，用于优化器参数装配。

<a id="qensemble"></a>

## QEnsemble

[源码位置](../../marl/modules/critic.py#L71) · [页内目录](#符号目录)

`class QEnsemble(ValueNetwork, Protocol)`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

<a id="qensemble-critics"></a>

## QEnsemble.critics

[源码位置](../../marl/modules/critic.py#L73) · [页内目录](#符号目录)

```python
def critics(self) -> nn.ModuleList
```

返回逐智能体独立 critic 序列，供算法仅对选定玩家动作保留梯度。

<a id="twinqensemble"></a>

## TwinQEnsemble

[源码位置](../../marl/modules/critic.py#L77) · [页内目录](#符号目录)

`class TwinQEnsemble(Protocol)`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

<a id="twinqensemble-first"></a>

## TwinQEnsemble.first

[源码位置](../../marl/modules/critic.py#L79) · [页内目录](#符号目录)

```python
def first(self) -> QEnsemble
```

返回第一组独立集中式 critic。

<a id="twinqensemble-second"></a>

## TwinQEnsemble.second

[源码位置](../../marl/modules/critic.py#L82) · [页内目录](#符号目录)

```python
def second(self) -> QEnsemble
```

返回第二组独立集中式 critic，与第一组参数独立。

<a id="twinqensemble-__call__"></a>

## TwinQEnsemble.__call__

[源码位置](../../marl/modules/critic.py#L84) · [页内目录](#符号目录)

```python
def __call__(self, inputs: Tensor) -> tuple[Tensor, Tensor]
```

输入联合特征，返回 (Q1,Q2)，每个 [...,N]。

<a id="twinqensemble-parameters"></a>

## TwinQEnsemble.parameters

[源码位置](../../marl/modules/critic.py#L85) · [页内目录](#符号目录)

```python
def parameters(self, recurse: bool=True) -> Iterator[nn.Parameter]
```

返回可遍历的 nn.Parameter；recurse 决定是否包含子模块，用于优化器参数装配。

<a id="valuecritic"></a>

## ValueCritic

[源码位置](../../marl/modules/critic.py#L88) · [页内目录](#符号目录)

`class ValueCritic(nn.Module)`

对每个输入位置输出标量 V/Q，适用于独立 critic。

<a id="valuecritic-__init__"></a>

## ValueCritic.__init__

[源码位置](../../marl/modules/critic.py#L91) · [页内目录](#符号目录)

```python
def __init__(self, backbone: BaseBackbone) -> None
```

构造 ValueCritic，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="valuecritic-forward"></a>

## ValueCritic.forward

[源码位置](../../marl/modules/critic.py#L96) · [页内目录](#符号目录)

```python
def forward(self, inputs: Tensor, **backbone_kwargs: Tensor) -> Tensor
```

输入 [...,O] 特征，返回 squeeze(-1) 后的标量 value Tensor [...]。

<a id="centralizedcritic"></a>

## CentralizedCritic

[源码位置](../../marl/modules/critic.py#L102) · [页内目录](#符号目录)

`class CentralizedCritic(nn.Module)`

CTDE 的集中式 critic。

调用者负责把 global state、联合观测或联合动作拼成最后一维；这种显式约定避免
critic 猜测各算法不同的数据布局。

<a id="centralizedcritic-__init__"></a>

## CentralizedCritic.__init__

[源码位置](../../marl/modules/critic.py#L109) · [页内目录](#符号目录)

```python
def __init__(self, backbone: BaseBackbone, output_dim: int=1, encoder: InputEncoder | None=None) -> None
```

构造 CentralizedCritic，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="centralizedcritic-initial_state"></a>

## CentralizedCritic.initial_state

[源码位置](../../marl/modules/critic.py#L119) · [页内目录](#符号目录)

```python
def initial_state(self, batch_size: int) -> RecurrentState
```

根据 batch 尺寸创建零循环状态，继承参数设备/dtype；MLP 返回 None，GRU 返回 h，LSTM 返回 (h,c)。

<a id="centralizedcritic-forward"></a>

## CentralizedCritic.forward

[源码位置](../../marl/modules/critic.py#L140) · [页内目录](#符号目录)

```python
def forward(self, centralized_input: Tensor, *, hidden_state: RecurrentState=None, return_state: bool=False) -> Tensor | tuple[Tensor, RecurrentState]
```

单步 [B,S] 或连续序列 [B,T,S]，value 最后一维始终保留 N。

<a id="independentcentralizedcritics"></a>

## IndependentCentralizedCritics

[源码位置](../../marl/modules/critic.py#L160) · [页内目录](#符号目录)

`class IndependentCentralizedCritics(nn.Module)`

参数独立的逐智能体集中式 Q 网络。

<a id="independentcentralizedcritics-__init__"></a>

## IndependentCentralizedCritics.__init__

[源码位置](../../marl/modules/critic.py#L163) · [页内目录](#符号目录)

```python
def __init__(self, spec: EnvironmentSpec, backbone: MLPBackboneConfig=_DEFAULT_BACKBONE, encoder: EncoderConfig=_DEFAULT_ENCODER, graph: GNNBackboneConfig | None=None) -> None
```

构造 IndependentCentralizedCritics，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="independentcentralizedcritics-forward"></a>

## IndependentCentralizedCritics.forward

[源码位置](../../marl/modules/critic.py#L180) · [页内目录](#符号目录)

```python
def forward(self, centralized_input: Tensor) -> Tensor
```

输入集中式联合特征，返回各独立玩家 Q [...,N]。

<a id="twinindependentcentralizedcritics"></a>

## TwinIndependentCentralizedCritics

[源码位置](../../marl/modules/critic.py#L187) · [页内目录](#符号目录)

`class TwinIndependentCentralizedCritics(nn.Module)`

MASAC 使用的两组独立集中式 Q 网络。

<a id="twinindependentcentralizedcritics-__init__"></a>

## TwinIndependentCentralizedCritics.__init__

[源码位置](../../marl/modules/critic.py#L190) · [页内目录](#符号目录)

```python
def __init__(self, spec: EnvironmentSpec, backbone: MLPBackboneConfig=_DEFAULT_BACKBONE, encoder: EncoderConfig=_DEFAULT_ENCODER, graph: GNNBackboneConfig | None=None) -> None
```

构造 TwinIndependentCentralizedCritics，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="twinindependentcentralizedcritics-forward"></a>

## TwinIndependentCentralizedCritics.forward

[源码位置](../../marl/modules/critic.py#L201) · [页内目录](#符号目录)

```python
def forward(self, centralized_input: Tensor) -> tuple[Tensor, Tensor]
```

输入集中式联合特征，返回两组独立 Q Tensor，各为 [...,N]。

<a id="attentioncritic"></a>

## AttentionCritic

[源码位置](../../marl/modules/critic.py#L205) · [页内目录](#符号目录)

`class AttentionCritic(nn.Module)`

MAAC 的逐智能体离散候选 Q 网络。

每个智能体拥有独立的 observation encoder、state-action encoder 与 Q head；
只有 ``nn.MultiheadAttention`` 内的 query/key/value 投影在智能体间共享。
输入为 observations ``[*B,N,O]`` 与离散 actions ``[*B,N]``，输出为
每个智能体所有候选动作的 Q 值 ``[*B,N,A]``。

<a id="attentioncritic-__init__"></a>

## AttentionCritic.__init__

[源码位置](../../marl/modules/critic.py#L214) · [页内目录](#符号目录)

```python
def __init__(self, spec: EnvironmentSpec, embedding: MLPBackboneConfig=_DEFAULT_EMBEDDING, backbone: MLPBackboneConfig=_DEFAULT_EMBEDDING, encoder: EncoderConfig=_DEFAULT_ENCODER, attention_heads: int=4, graph: GNNBackboneConfig | None=None) -> None
```

构造 AttentionCritic，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="attentioncritic-forward"></a>

## AttentionCritic.forward

[源码位置](../../marl/modules/critic.py#L262) · [页内目录](#符号目录)

```python
def forward(self, observations: Tensor, actions: Tensor) -> Tensor
```

输入联合观测和联合离散动作，返回每个玩家候选动作 Q 表 [...,N,A]。

<a id="centralizedvalueconfig"></a>

## CentralizedValueConfig

[源码位置](../../marl/modules/critic.py#L312) · [页内目录](#符号目录)

`class CentralizedValueConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
kind: Literal['centralized_value'] = 'centralized_value'
backbone: BackboneConfig = MLPBackboneConfig()
encoder: EncoderConfig = IdentityEncoderConfig()
graph: GNNBackboneConfig | None = None
```

<a id="centralizedvalueconfig-build"></a>

## CentralizedValueConfig.build

[源码位置](../../marl/modules/critic.py#L318) · [页内目录](#符号目录)

```python
def build(self, spec: EnvironmentSpec) -> CentralizedCritic
```

从 CentralizedValueConfig 的静态参数构造目标对象；输入与具体返回类型见签名，不执行训练。

<a id="attentionqconfig"></a>

## AttentionQConfig

[源码位置](../../marl/modules/critic.py#L332) · [页内目录](#符号目录)

`class AttentionQConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
kind: Literal['attention_q'] = 'attention_q'
encoder: EncoderConfig = IdentityEncoderConfig()
embedding: MLPBackboneConfig = MLPBackboneConfig(hidden_dims=())
backbone: MLPBackboneConfig = MLPBackboneConfig(hidden_dims=())
graph: GNNBackboneConfig | None = None
attention_heads: int = 4
```

<a id="attentionqconfig-__post_init__"></a>

## AttentionQConfig.__post_init__

[源码位置](../../marl/modules/critic.py#L340) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 AttentionQConfig 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="attentionqconfig-build"></a>

## AttentionQConfig.build

[源码位置](../../marl/modules/critic.py#L348) · [页内目录](#符号目录)

```python
def build(self, spec: EnvironmentSpec) -> AttentionCritic
```

从 AttentionQConfig 的静态参数构造目标对象；输入与具体返回类型见签名，不执行训练。

<a id="independentqconfig"></a>

## IndependentQConfig

[源码位置](../../marl/modules/critic.py#L360) · [页内目录](#符号目录)

`class IndependentQConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
kind: Literal['independent_centralized_q'] = 'independent_centralized_q'
encoder: EncoderConfig = IdentityEncoderConfig()
backbone: MLPBackboneConfig = MLPBackboneConfig()
graph: GNNBackboneConfig | None = None
```

<a id="independentqconfig-build"></a>

## IndependentQConfig.build

[源码位置](../../marl/modules/critic.py#L366) · [页内目录](#符号目录)

```python
def build(self, spec: EnvironmentSpec) -> IndependentCentralizedCritics
```

从 IndependentQConfig 的静态参数构造目标对象；输入与具体返回类型见签名，不执行训练。

<a id="twinqconfig"></a>

## TwinQConfig

[源码位置](../../marl/modules/critic.py#L376) · [页内目录](#符号目录)

`class TwinQConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
kind: Literal['twin_independent_centralized_q'] = 'twin_independent_centralized_q'
encoder: EncoderConfig = IdentityEncoderConfig()
backbone: MLPBackboneConfig = MLPBackboneConfig()
graph: GNNBackboneConfig | None = None
```

<a id="twinqconfig-build"></a>

## TwinQConfig.build

[源码位置](../../marl/modules/critic.py#L382) · [页内目录](#符号目录)

```python
def build(self, spec: EnvironmentSpec) -> TwinIndependentCentralizedCritics
```

从 TwinQConfig 的静态参数构造目标对象；输入与具体返回类型见签名，不执行训练。
