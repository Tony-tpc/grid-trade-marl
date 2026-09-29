# marl/modules/policy.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/modules/policy.py)

多智能体策略拓扑。

``Actor`` 负责一个智能体的 backbone 和 action head；本模块只负责参数共享关系、
逐智能体输入选择、联合形状校验和输出堆叠，不拥有 rollout 或 loss 生命周期。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [PolicyTopology](#policytopology)
- [PolicyTopology.act](#policytopology-act)
- [PolicyTopology.parameters](#policytopology-parameters)
- [DiscretePolicy](#discretepolicy)
- [DiscretePolicy.act](#discretepolicy-act)
- [DiscretePolicy.logits](#discretepolicy-logits)
- [DiscretePolicy.evaluate](#discretepolicy-evaluate)
- [AgentLogitsPolicy](#agentlogitspolicy)
- [AgentLogitsPolicy.logits_for_agent](#agentlogitspolicy-logits_for_agent)
- [LocalQPolicy](#localqpolicy)
- [LocalQPolicy.q_values](#localqpolicy-q_values)
- [_stack_scalar](#_stack_scalar)
- [_stack_parameter](#_stack_parameter)
- [IndependentDiscretePolicy](#independentdiscretepolicy)
- [IndependentDiscretePolicy.__init__](#independentdiscretepolicy-__init__)
- [IndependentDiscretePolicy.initial_state](#independentdiscretepolicy-initial_state)
- [IndependentDiscretePolicy._agent_state](#independentdiscretepolicy-_agent_state)
- [IndependentDiscretePolicy._agent_state.select](#independentdiscretepolicy-_agent_state-select)
- [IndependentDiscretePolicy._validate](#independentdiscretepolicy-_validate)
- [IndependentDiscretePolicy._mask_for](#independentdiscretepolicy-_mask_for)
- [IndependentDiscretePolicy.logits](#independentdiscretepolicy-logits)
- [IndependentDiscretePolicy.logits_for_agent](#independentdiscretepolicy-logits_for_agent)
- [IndependentDiscretePolicy.act](#independentdiscretepolicy-act)
- [IndependentDiscretePolicy.evaluate](#independentdiscretepolicy-evaluate)
- [IndependentDeterministicPolicy](#independentdeterministicpolicy)
- [IndependentDeterministicPolicy.__init__](#independentdeterministicpolicy-__init__)
- [IndependentDeterministicPolicy.act](#independentdeterministicpolicy-act)
- [IndependentGaussianPolicy](#independentgaussianpolicy)
- [IndependentGaussianPolicy.__init__](#independentgaussianpolicy-__init__)
- [IndependentGaussianPolicy.act](#independentgaussianpolicy-act)
- [SharedDiscreteQPolicy](#shareddiscreteqpolicy)
- [SharedDiscreteQPolicy.__init__](#shareddiscreteqpolicy-__init__)
- [SharedDiscreteQPolicy.q_values](#shareddiscreteqpolicy-q_values)
- [SharedDiscreteQPolicy.act](#shareddiscreteqpolicy-act)
- [IndependentDiscreteConfig](#independentdiscreteconfig)
- [IndependentDiscreteConfig.build](#independentdiscreteconfig-build)
- [IndependentDeterministicConfig](#independentdeterministicconfig)
- [IndependentDeterministicConfig.build](#independentdeterministicconfig-build)
- [IndependentGaussianConfig](#independentgaussianconfig)
- [IndependentGaussianConfig.build](#independentgaussianconfig-build)
- [SharedDiscreteQConfig](#shareddiscreteqconfig)
- [SharedDiscreteQConfig.build](#shareddiscreteqconfig-build)

<a id="policytopology"></a>

## PolicyTopology

[源码位置](../../marl/modules/policy.py#L41) · [页内目录](#符号目录)

`class PolicyTopology(Protocol)`

网络能力协议，不是算法基类；外部 nn.Module 可用结构化类型直接接入。

<a id="policytopology-act"></a>

## PolicyTopology.act

[源码位置](../../marl/modules/policy.py#L44) · [页内目录](#符号目录)

```python
def act(self, observations: Tensor, *, deterministic: bool=False, action_mask: Tensor | None=None) -> MARLModelOutput
```

输入逐智能体观测和可选 mask，返回 MARLModelOutput；动作形状 [...,N] 或 [...,N,A]，随机性由策略实现决定。

<a id="policytopology-parameters"></a>

## PolicyTopology.parameters

[源码位置](../../marl/modules/policy.py#L52) · [页内目录](#符号目录)

```python
def parameters(self, recurse: bool=True) -> Iterator[nn.Parameter]
```

返回可遍历的 nn.Parameter；recurse 决定是否包含子模块，用于优化器参数装配。

<a id="discretepolicy"></a>

## DiscretePolicy

[源码位置](../../marl/modules/policy.py#L56) · [页内目录](#符号目录)

`class DiscretePolicy(PolicyTopology, Protocol)`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

<a id="discretepolicy-act"></a>

## DiscretePolicy.act

[源码位置](../../marl/modules/policy.py#L57) · [页内目录](#符号目录)

```python
def act(self, observations: Tensor, *, deterministic: bool=False, action_mask: Tensor | None=None, hidden_state: RecurrentState=None) -> MARLModelOutput
```

输入逐智能体观测和可选 mask，返回 MARLModelOutput；动作形状 [...,N] 或 [...,N,A]，随机性由策略实现决定。

<a id="discretepolicy-logits"></a>

## DiscretePolicy.logits

[源码位置](../../marl/modules/policy.py#L66) · [页内目录](#符号目录)

```python
def logits(self, observations: Tensor, action_mask: Tensor | None=None) -> Tensor
```

输入观测/mask，输出离散 logits；联合 [...,N,A] 或指定玩家 [...,A]，不采样动作。

<a id="discretepolicy-evaluate"></a>

## DiscretePolicy.evaluate

[源码位置](../../marl/modules/policy.py#L68) · [页内目录](#符号目录)

```python
def evaluate(self, observations: Tensor, actions: Tensor, *, action_mask: Tensor | None=None, hidden_state: RecurrentState=None) -> MARLModelOutput
```

对输入观测下已经给定的 actions 计算 log_prob/entropy 并返回 MARLModelOutput；不重新选择动作。

<a id="agentlogitspolicy"></a>

## AgentLogitsPolicy

[源码位置](../../marl/modules/policy.py#L79) · [页内目录](#符号目录)

`class AgentLogitsPolicy(Protocol)`

可选的单智能体 logits 快速路径；外部策略无需强制实现。

<a id="agentlogitspolicy-logits_for_agent"></a>

## AgentLogitsPolicy.logits_for_agent

[源码位置](../../marl/modules/policy.py#L82) · [页内目录](#符号目录)

```python
def logits_for_agent(self, observations: Tensor, agent_index: int, action_mask: Tensor | None=None) -> Tensor
```

输入观测/mask，输出离散 logits；联合 [...,N,A] 或指定玩家 [...,A]，不采样动作。

<a id="localqpolicy"></a>

## LocalQPolicy

[源码位置](../../marl/modules/policy.py#L91) · [页内目录](#符号目录)

`class LocalQPolicy(PolicyTopology, Protocol)`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

<a id="localqpolicy-q_values"></a>

## LocalQPolicy.q_values

[源码位置](../../marl/modules/policy.py#L92) · [页内目录](#符号目录)

```python
def q_values(self, observations: Tensor) -> Tensor
```

输入观测 [...,N,O]，返回局部离散 Q Tensor [...,N,A]；不混合团队价值。

<a id="_stack_scalar"></a>

## _stack_scalar

[源码位置](../../marl/modules/policy.py#L95) · [页内目录](#符号目录)

```python
def _stack_scalar(values: list[Tensor | None]) -> Tensor
```

把各 Actor 的 [*B] 标量结果堆叠为 [*B,N]。

<a id="_stack_parameter"></a>

## _stack_parameter

[源码位置](../../marl/modules/policy.py#L102) · [页内目录](#符号目录)

```python
def _stack_parameter(outputs: list[ActionHeadOutput], name: str) -> Tensor
```

把每个 Actor 的 ``[*B, A]`` 分布参数堆叠为 ``[*B, N, A]``。

<a id="independentdiscretepolicy"></a>

## IndependentDiscretePolicy

[源码位置](../../marl/modules/policy.py#L111) · [页内目录](#符号目录)

`class IndependentDiscretePolicy(nn.Module)`

每个同构智能体拥有独立离散 Actor 的策略拓扑。

<a id="independentdiscretepolicy-__init__"></a>

## IndependentDiscretePolicy.__init__

[源码位置](../../marl/modules/policy.py#L114) · [页内目录](#符号目录)

```python
def __init__(self, num_agents: int, observation_dim: int, action_dim: int, backbone: BackboneConfig | None=None, encoder: EncoderConfig=_DEFAULT_ENCODER, history: HistoryLayout | None=None) -> None
```

构造 IndependentDiscretePolicy，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="independentdiscretepolicy-initial_state"></a>

## IndependentDiscretePolicy.initial_state

[源码位置](../../marl/modules/policy.py#L141) · [页内目录](#符号目录)

```python
def initial_state(self, batch_size: int) -> RecurrentState
```

根据 batch 尺寸创建零循环状态，继承参数设备/dtype；MLP 返回 None，GRU 返回 h，LSTM 返回 (h,c)。

<a id="independentdiscretepolicy-_agent_state"></a>

## IndependentDiscretePolicy._agent_state

[源码位置](../../marl/modules/policy.py#L148) · [页内目录](#符号目录)

```python
def _agent_state(self, state: RecurrentState, index: int, batch: int) -> RecurrentState
```

从 [K,B,N,H] 隐藏状态中选择一个玩家，返回 [K,B,H]；LSTM 同时选择 h/c。

<a id="independentdiscretepolicy-_agent_state-select"></a>

## IndependentDiscretePolicy._agent_state.select

[源码位置](../../marl/modules/policy.py#L149) · [页内目录](#符号目录)

```python
def select(value: Tensor) -> Tensor
```

验证单个 h/c 张量维度并选取目标玩家，返回连续 [K,B,H] 张量。

<a id="independentdiscretepolicy-_validate"></a>

## IndependentDiscretePolicy._validate

[源码位置](../../marl/modules/policy.py#L156) · [页内目录](#符号目录)

```python
def _validate(self, observations: Tensor, action_mask: Tensor | None) -> None
```

检查输入张量尺寸、mask 或循环布局与实例规格一致；返回 None，错误早报。

<a id="independentdiscretepolicy-_mask_for"></a>

## IndependentDiscretePolicy._mask_for

[源码位置](../../marl/modules/policy.py#L170) · [页内目录](#符号目录)

```python
def _mask_for(self, action_mask: Tensor | None, index: int) -> Tensor | None
```

从联合 action_mask 选一个玩家，返回 [...,A]；输入缺失则为 None。

<a id="independentdiscretepolicy-logits"></a>

## IndependentDiscretePolicy.logits

[源码位置](../../marl/modules/policy.py#L173) · [页内目录](#符号目录)

```python
def logits(self, observations: Tensor, action_mask: Tensor | None=None) -> Tensor
```

返回联合分类参数 ``[*B, N, A]``，不采样动作。

<a id="independentdiscretepolicy-logits_for_agent"></a>

## IndependentDiscretePolicy.logits_for_agent

[源码位置](../../marl/modules/policy.py#L188) · [页内目录](#符号目录)

```python
def logits_for_agent(self, observations: Tensor, agent_index: int, action_mask: Tensor | None=None) -> Tensor
```

只执行目标智能体 Actor，返回 ``[*B,A]`` logits。

<a id="independentdiscretepolicy-act"></a>

## IndependentDiscretePolicy.act

[源码位置](../../marl/modules/policy.py#L206) · [页内目录](#符号目录)

```python
def act(self, observations: Tensor, *, deterministic: bool=False, action_mask: Tensor | None=None, hidden_state: RecurrentState=None) -> MARLModelOutput
```

对各 Actor 采样并返回 ``actions/log_prob/entropy`` 的逐智能体结果。

<a id="independentdiscretepolicy-evaluate"></a>

## IndependentDiscretePolicy.evaluate

[源码位置](../../marl/modules/policy.py#L234) · [页内目录](#符号目录)

```python
def evaluate(self, observations: Tensor, actions: Tensor, *, action_mask: Tensor | None=None, hidden_state: RecurrentState=None) -> MARLModelOutput
```

评估给定联合动作；不会重新采样。

<a id="independentdeterministicpolicy"></a>

## IndependentDeterministicPolicy

[源码位置](../../marl/modules/policy.py#L267) · [页内目录](#符号目录)

`class IndependentDeterministicPolicy(nn.Module)`

每个智能体拥有独立确定性 Actor；不把参数共享当作批处理优化。

act 与高斯拓扑的组装步骤相似，但动作分布、默认确定性和探索机制不同。
保留这段直观组装，不为少量堆叠代码另建策略基类或万能分布工厂。

<a id="independentdeterministicpolicy-__init__"></a>

## IndependentDeterministicPolicy.__init__

[源码位置](../../marl/modules/policy.py#L274) · [页内目录](#符号目录)

```python
def __init__(self, num_agents: int, observation_dim: int, action_dim: int, backbone: MLPBackboneConfig=_DEFAULT_BACKBONE, encoder: EncoderConfig=_DEFAULT_ENCODER, history: HistoryLayout | None=None) -> None
```

构造 IndependentDeterministicPolicy，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="independentdeterministicpolicy-act"></a>

## IndependentDeterministicPolicy.act

[源码位置](../../marl/modules/policy.py#L297) · [页内目录](#符号目录)

```python
def act(self, observations: Tensor, *, deterministic: bool=True, action_mask: Tensor | None=None) -> MARLModelOutput
```

输入逐智能体观测和可选 mask，返回 MARLModelOutput；动作形状 [...,N] 或 [...,N,A]，随机性由策略实现决定。

<a id="independentgaussianpolicy"></a>

## IndependentGaussianPolicy

[源码位置](../../marl/modules/policy.py#L319) · [页内目录](#符号目录)

`class IndependentGaussianPolicy(nn.Module)`

每个智能体拥有独立 tanh-Gaussian Actor 的策略拓扑。

<a id="independentgaussianpolicy-__init__"></a>

## IndependentGaussianPolicy.__init__

[源码位置](../../marl/modules/policy.py#L322) · [页内目录](#符号目录)

```python
def __init__(self, num_agents: int, observation_dim: int, action_dim: int, backbone: MLPBackboneConfig=_DEFAULT_BACKBONE, encoder: EncoderConfig=_DEFAULT_ENCODER, history: HistoryLayout | None=None) -> None
```

构造 IndependentGaussianPolicy，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="independentgaussianpolicy-act"></a>

## IndependentGaussianPolicy.act

[源码位置](../../marl/modules/policy.py#L345) · [页内目录](#符号目录)

```python
def act(self, observations: Tensor, *, deterministic: bool=False, action_mask: Tensor | None=None) -> MARLModelOutput
```

输入逐智能体观测和可选 mask，返回 MARLModelOutput；动作形状 [...,N] 或 [...,N,A]，随机性由策略实现决定。

<a id="shareddiscreteqpolicy"></a>

## SharedDiscreteQPolicy

[源码位置](../../marl/modules/policy.py#L367) · [页内目录](#符号目录)

`class SharedDiscreteQPolicy(nn.Module)`

同构智能体共享局部 Q 网络的 value-based 执行拓扑。

<a id="shareddiscreteqpolicy-__init__"></a>

## SharedDiscreteQPolicy.__init__

[源码位置](../../marl/modules/policy.py#L370) · [页内目录](#符号目录)

```python
def __init__(self, observation_dim: int, action_dim: int, backbone: MLPBackboneConfig=_DEFAULT_BACKBONE, encoder: EncoderConfig=_DEFAULT_ENCODER, history: HistoryLayout | None=None) -> None
```

构造 SharedDiscreteQPolicy，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="shareddiscreteqpolicy-q_values"></a>

## SharedDiscreteQPolicy.q_values

[源码位置](../../marl/modules/policy.py#L387) · [页内目录](#符号目录)

```python
def q_values(self, observations: Tensor) -> Tensor
```

输入观测 [...,N,O]，返回局部离散 Q Tensor [...,N,A]；不混合团队价值。

<a id="shareddiscreteqpolicy-act"></a>

## SharedDiscreteQPolicy.act

[源码位置](../../marl/modules/policy.py#L393) · [页内目录](#符号目录)

```python
def act(self, observations: Tensor, *, deterministic: bool=True, action_mask: Tensor | None=None) -> MARLModelOutput
```

输入逐智能体观测和可选 mask，返回 MARLModelOutput；动作形状 [...,N] 或 [...,N,A]，随机性由策略实现决定。

<a id="independentdiscreteconfig"></a>

## IndependentDiscreteConfig

[源码位置](../../marl/modules/policy.py#L410) · [页内目录](#符号目录)

`class IndependentDiscreteConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
kind: Literal['independent_discrete'] = 'independent_discrete'
backbone: BackboneConfig = MLPBackboneConfig()
encoder: EncoderConfig = IdentityEncoderConfig()
```

<a id="independentdiscreteconfig-build"></a>

## IndependentDiscreteConfig.build

[源码位置](../../marl/modules/policy.py#L415) · [页内目录](#符号目录)

```python
def build(self, spec: EnvironmentSpec) -> IndependentDiscretePolicy
```

从 IndependentDiscreteConfig 的静态参数构造目标对象；输入与具体返回类型见签名，不执行训练。

<a id="independentdeterministicconfig"></a>

## IndependentDeterministicConfig

[源码位置](../../marl/modules/policy.py#L427) · [页内目录](#符号目录)

`class IndependentDeterministicConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
kind: Literal['independent_deterministic'] = 'independent_deterministic'
backbone: MLPBackboneConfig = MLPBackboneConfig()
encoder: EncoderConfig = IdentityEncoderConfig()
```

<a id="independentdeterministicconfig-build"></a>

## IndependentDeterministicConfig.build

[源码位置](../../marl/modules/policy.py#L432) · [页内目录](#符号目录)

```python
def build(self, spec: EnvironmentSpec) -> IndependentDeterministicPolicy
```

从 IndependentDeterministicConfig 的静态参数构造目标对象；输入与具体返回类型见签名，不执行训练。

<a id="independentgaussianconfig"></a>

## IndependentGaussianConfig

[源码位置](../../marl/modules/policy.py#L444) · [页内目录](#符号目录)

`class IndependentGaussianConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
kind: Literal['independent_gaussian'] = 'independent_gaussian'
backbone: MLPBackboneConfig = MLPBackboneConfig()
encoder: EncoderConfig = IdentityEncoderConfig()
```

<a id="independentgaussianconfig-build"></a>

## IndependentGaussianConfig.build

[源码位置](../../marl/modules/policy.py#L449) · [页内目录](#符号目录)

```python
def build(self, spec: EnvironmentSpec) -> IndependentGaussianPolicy
```

从 IndependentGaussianConfig 的静态参数构造目标对象；输入与具体返回类型见签名，不执行训练。

<a id="shareddiscreteqconfig"></a>

## SharedDiscreteQConfig

[源码位置](../../marl/modules/policy.py#L461) · [页内目录](#符号目录)

`class SharedDiscreteQConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
kind: Literal['shared_discrete_q'] = 'shared_discrete_q'
backbone: MLPBackboneConfig = MLPBackboneConfig()
encoder: EncoderConfig = IdentityEncoderConfig()
```

<a id="shareddiscreteqconfig-build"></a>

## SharedDiscreteQConfig.build

[源码位置](../../marl/modules/policy.py#L466) · [页内目录](#符号目录)

```python
def build(self, spec: EnvironmentSpec) -> SharedDiscreteQPolicy
```

从 SharedDiscreteQConfig 的静态参数构造目标对象；输入与具体返回类型见签名，不执行训练。
