# marl/envs/mpe2_simple_adversary_adapter.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/envs/mpe2_simple_adversary_adapter.py)

MPE2 simple_adversary 与仓库固定形状协议之间的适配器。

MPE2 是 Farama 的 Gymnasium 风格多智能体环境集合。该场景同时提供离散和连续
动作，并保留对手与善意智能体各自的奖励，因此适合检查一般和博弈算法的数据链。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [ParallelMPEEnvironment](#parallelmpeenvironment)
- [ParallelMPEEnvironment.observation_space](#parallelmpeenvironment-observation_space)
- [ParallelMPEEnvironment.action_space](#parallelmpeenvironment-action_space)
- [ParallelMPEEnvironment.reset](#parallelmpeenvironment-reset)
- [ParallelMPEEnvironment.step](#parallelmpeenvironment-step)
- [ParallelMPEEnvironment.state](#parallelmpeenvironment-state)
- [ParallelMPEEnvironment.close](#parallelmpeenvironment-close)
- [MPE2SimpleAdversaryConfig](#mpe2simpleadversaryconfig)
- [MPE2SimpleAdversaryConfig.__post_init__](#mpe2simpleadversaryconfig-__post_init__)
- [MPE2SimpleAdversaryAdapter](#mpe2simpleadversaryadapter)
- [MPE2SimpleAdversaryAdapter.__init__](#mpe2simpleadversaryadapter-__init__)
- [MPE2SimpleAdversaryAdapter._flat_dim](#mpe2simpleadversaryadapter-_flat_dim)
- [MPE2SimpleAdversaryAdapter._validate_action_spaces](#mpe2simpleadversaryadapter-_validate_action_spaces)
- [MPE2SimpleAdversaryAdapter.spec](#mpe2simpleadversaryadapter-spec)
- [MPE2SimpleAdversaryAdapter._observations](#mpe2simpleadversaryadapter-_observations)
- [MPE2SimpleAdversaryAdapter._snapshot](#mpe2simpleadversaryadapter-_snapshot)
- [MPE2SimpleAdversaryAdapter.reset](#mpe2simpleadversaryadapter-reset)
- [MPE2SimpleAdversaryAdapter.step](#mpe2simpleadversaryadapter-step)
- [MPE2SimpleAdversaryAdapter.close](#mpe2simpleadversaryadapter-close)
- [build_mpe2_simple_adversary](#build_mpe2_simple_adversary)

<a id="parallelmpeenvironment"></a>

## ParallelMPEEnvironment

[源码位置](../../marl/envs/mpe2_simple_adversary_adapter.py#L25) · [页内目录](#符号目录)

`class ParallelMPEEnvironment(Protocol)`

只描述适配器实际使用的 PettingZoo Parallel API 能力。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
possible_agents: list[str]
agents: list[str]
state_space: spaces.Space[Any]
```

<a id="parallelmpeenvironment-observation_space"></a>

## ParallelMPEEnvironment.observation_space

[源码位置](../../marl/envs/mpe2_simple_adversary_adapter.py#L32) · [页内目录](#符号目录)

```python
def observation_space(self, agent: str) -> spaces.Space[Any]
```

输入 agent 名称，返回底层 MPE 该玩家的观测空间对象。

<a id="parallelmpeenvironment-action_space"></a>

## ParallelMPEEnvironment.action_space

[源码位置](../../marl/envs/mpe2_simple_adversary_adapter.py#L34) · [页内目录](#符号目录)

```python
def action_space(self, agent: str) -> spaces.Space[Any]
```

输入 agent 名称，返回底层 MPE 该玩家的动作空间对象。

<a id="parallelmpeenvironment-reset"></a>

## ParallelMPEEnvironment.reset

[源码位置](../../marl/envs/mpe2_simple_adversary_adapter.py#L36) · [页内目录](#符号目录)

```python
def reset(self, seed: int | None=None) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]
```

按可选 seed 重置环境并返回初始结果；adapter 返回统一 EnvironmentStep，底层协议返回原生格式。

<a id="parallelmpeenvironment-step"></a>

## ParallelMPEEnvironment.step

[源码位置](../../marl/envs/mpe2_simple_adversary_adapter.py#L40) · [页内目录](#符号目录)

```python
def step(self, actions: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict[str, Any]]]
```

执行输入动作并推进环境；adapter 返回下一 EnvironmentStep，底层协议返回原生观测/奖励/终止信息。

<a id="parallelmpeenvironment-state"></a>

## ParallelMPEEnvironment.state

[源码位置](../../marl/envs/mpe2_simple_adversary_adapter.py#L50) · [页内目录](#符号目录)

```python
def state(self) -> NDArray[np.floating[Any]]
```

无输入，返回底层环境的全局 state NumPy 数组。

<a id="parallelmpeenvironment-close"></a>

## ParallelMPEEnvironment.close

[源码位置](../../marl/envs/mpe2_simple_adversary_adapter.py#L52) · [页内目录](#符号目录)

```python
def close(self) -> None
```

释放底层环境资源，返回 None；无 optimizer 或模型副作用。

<a id="mpe2simpleadversaryconfig"></a>

## MPE2SimpleAdversaryConfig

[源码位置](../../marl/envs/mpe2_simple_adversary_adapter.py#L56) · [页内目录](#符号目录)

`class MPE2SimpleAdversaryConfig()`

只包含场景动力学；训练超参数仍属于算法配置。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
num_good_agents: int = 2
horizon: int = 25
dynamic_rescaling: bool = False
```

<a id="mpe2simpleadversaryconfig-__post_init__"></a>

## MPE2SimpleAdversaryConfig.__post_init__

[源码位置](../../marl/envs/mpe2_simple_adversary_adapter.py#L63) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 MPE2SimpleAdversaryConfig 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="mpe2simpleadversaryadapter"></a>

## MPE2SimpleAdversaryAdapter

[源码位置](../../marl/envs/mpe2_simple_adversary_adapter.py#L80) · [页内目录](#符号目录)

`class MPE2SimpleAdversaryAdapter(EnvironmentAdapter)`

把 MPE2 字典数据整理成固定 [N,O] / [S] 张量语义。

simple_adversary 的 adversary 观测比 good agent 短。仓库当前网络要求同构
输入，因此只在环境边界对右侧补零，并在 info 中保留原始长度。离散动作全部
合法；连续策略输出 [-1,1]，这里线性映射到 MPE2 的 [0,1] Box。

<a id="mpe2simpleadversaryadapter-__init__"></a>

## MPE2SimpleAdversaryAdapter.__init__

[源码位置](../../marl/envs/mpe2_simple_adversary_adapter.py#L88) · [页内目录](#符号目录)

```python
def __init__(self, environment: ParallelMPEEnvironment, action_kind: ActionKind, *, horizon: int) -> None
```

构造 MPE2SimpleAdversaryAdapter，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="mpe2simpleadversaryadapter-_flat_dim"></a>

## MPE2SimpleAdversaryAdapter._flat_dim

[源码位置](../../marl/envs/mpe2_simple_adversary_adapter.py#L116) · [页内目录](#符号目录)

```python
def _flat_dim(space: spaces.Space[Any]) -> int
```

检查空间 shape 并计算展平维度；返回 int，异常空间显式拒绝。

<a id="mpe2simpleadversaryadapter-_validate_action_spaces"></a>

## MPE2SimpleAdversaryAdapter._validate_action_spaces

[源码位置](../../marl/envs/mpe2_simple_adversary_adapter.py#L122) · [页内目录](#符号目录)

```python
def _validate_action_spaces(self) -> int
```

检查所有玩家的动作空间与配置动作类型一致，返回 None。

<a id="mpe2simpleadversaryadapter-spec"></a>

## MPE2SimpleAdversaryAdapter.spec

[源码位置](../../marl/envs/mpe2_simple_adversary_adapter.py#L152) · [页内目录](#符号目录)

```python
def spec(self) -> EnvironmentSpec
```

无输入；返回静态 EnvironmentSpec，作为所有算法尺寸与动作/奖励语义的唯一来源。

<a id="mpe2simpleadversaryadapter-_observations"></a>

## MPE2SimpleAdversaryAdapter._observations

[源码位置](../../marl/envs/mpe2_simple_adversary_adapter.py#L155) · [页内目录](#符号目录)

```python
def _observations(self, values: dict[str, Any]) -> NDArray[np.float32]
```

按固定 agent 顺序整理并补齐观测，返回 float32 [N,O]。

<a id="mpe2simpleadversaryadapter-_snapshot"></a>

## MPE2SimpleAdversaryAdapter._snapshot

[源码位置](../../marl/envs/mpe2_simple_adversary_adapter.py#L168) · [页内目录](#符号目录)

```python
def _snapshot(self, observations: dict[str, Any], rewards: dict[str, float], terminations: dict[str, bool], truncations: dict[str, bool], infos: dict[str, dict[str, Any]]) -> EnvironmentStep
```

将 MPE 原始输出打包为固定玩家顺序的 EnvironmentStep，并校验统一接口。

<a id="mpe2simpleadversaryadapter-reset"></a>

## MPE2SimpleAdversaryAdapter.reset

[源码位置](../../marl/envs/mpe2_simple_adversary_adapter.py#L209) · [页内目录](#符号目录)

```python
def reset(self, seed: int | None=None) -> EnvironmentStep
```

按可选 seed 重置环境并返回初始结果；adapter 返回统一 EnvironmentStep，底层协议返回原生格式。

<a id="mpe2simpleadversaryadapter-step"></a>

## MPE2SimpleAdversaryAdapter.step

[源码位置](../../marl/envs/mpe2_simple_adversary_adapter.py#L216) · [页内目录](#符号目录)

```python
def step(self, actions: np.ndarray) -> EnvironmentStep
```

执行输入动作并推进环境；adapter 返回下一 EnvironmentStep，底层协议返回原生观测/奖励/终止信息。

<a id="mpe2simpleadversaryadapter-close"></a>

## MPE2SimpleAdversaryAdapter.close

[源码位置](../../marl/envs/mpe2_simple_adversary_adapter.py#L244) · [页内目录](#符号目录)

```python
def close(self) -> None
```

释放底层环境资源，返回 None；无 optimizer 或模型副作用。

<a id="build_mpe2_simple_adversary"></a>

## build_mpe2_simple_adversary

[源码位置](../../marl/envs/mpe2_simple_adversary_adapter.py#L248) · [页内目录](#符号目录)

```python
def build_mpe2_simple_adversary(config: MPE2SimpleAdversaryConfig | None=None, *, action_kind: ActionKind=ActionKind.DISCRETE) -> MPE2SimpleAdversaryAdapter
```

按需导入可选的 MPE2 依赖并构造官方 simple_adversary_v3。
