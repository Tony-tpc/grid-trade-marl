# marl/algorithms/sn_mappo.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/algorithms/sn_mappo.py)

修订 SN-MAPPO：组合已有 MAPPO 角色和高阶响应，不复制 PPO/GAE 实现。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [SNMAPPOConfig](#snmappoconfig)
- [SNMAPPOConfig.__post_init__](#snmappoconfig-__post_init__)
- [SNMAPPOConfig.build](#snmappoconfig-build)
- [SNMAPPO](#snmappo)
- [SNMAPPO.__init__](#snmappo-__init__)
- [SNMAPPO.act](#snmappo-act)
- [SNMAPPO.compute_loss_bundle](#snmappo-compute_loss_bundle)
- [SNMAPPO.response_objectives](#snmappo-response_objectives)
- [SNMAPPO.response_objectives.objective](#snmappo-response_objectives-objective)
- [SNMAPPO.update](#snmappo-update)

<a id="snmappoconfig"></a>

## SNMAPPOConfig

[源码位置](../../marl/algorithms/sn_mappo.py#L40) · [页内目录](#符号目录)

`class SNMAPPOConfig()`

各角色保留完整 typed MAPPOConfig；角色尺寸只由 SequentialSpec 注入。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
schema_version: Literal[2] = 2
algorithm: Literal['sn_mappo'] = 'sn_mappo'
leader: MAPPOConfig = replace(_ROLE, policy=IndependentGaussianConfig(backbone=GRUBackboneConfig(hidden_dim=32, num_layers=2)), update=replace(_ROLE.update, actor_learning_rate=0.0001))
coordinator: MAPPOConfig = _ROLE
followers: MAPPOConfig = _ROLE
implicit: bool = True
response: ImplicitResponseConfig = ImplicitResponseConfig()
kl_coefficient: float = 0.01
response_trajectories: int = 8
response_baseline: Literal['leave_one_out', 'none'] = 'leave_one_out'
```

<a id="snmappoconfig-__post_init__"></a>

## SNMAPPOConfig.__post_init__

[源码位置](../../marl/algorithms/sn_mappo.py#L58) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

```text
绑定并验证本实例所负责的组件与状态。

Inputs:
    无显式参数；读取当前实例的配置或训练状态。

Returns:
    None；非法配置在构造时抛 ValueError，不修改训练状态。
```

<a id="snmappoconfig-build"></a>

## SNMAPPOConfig.build

[源码位置](../../marl/algorithms/sn_mappo.py#L82) · [页内目录](#符号目录)

```python
def build(self, spec: EnvironmentSpec | SequentialSpec) -> SNMAPPO
```

```text
SNMAPPO 实例。

Args:
    spec: 环境给出的联合/角色规格；包含 N/O/A/S 和完整事件 horizon。

Returns:
    SNMAPPO 实例；没有创建 optimizer 或开始采样。
```

<a id="snmappo"></a>

## SNMAPPO

[源码位置](../../marl/algorithms/sn_mappo.py#L96) · [页内目录](#符号目录)

`class SNMAPPO(BaseMARLAlgorithm)`

DS 更新 UC/共享计划，DN 更新个体 decoder；两个阶段都只消费新鲜轨迹。

DS 的直接梯度、Hessian、交叉导数统一来自归一化折扣经济回报 + KL 的 DiCE
目标。阻尼定义局部正则化响应近似；记录驻点梯度范数，不声称已达到 best response。
共享计划是唯一求响应的参数组，decoder 固定；这不是完整消费者 Nash 均衡导数。
DS 全轨迹单次 actor 更新，DN 复用 MAPPO 多 epoch 更新；PPO clip/entropy 仅用于 DN。

<a id="snmappo-__init__"></a>

## SNMAPPO.__init__

[源码位置](../../marl/algorithms/sn_mappo.py#L104) · [页内目录](#符号目录)

```python
def __init__(self, spec: SequentialSpec, config: SNMAPPOConfig)
```

```text
绑定并验证本实例所负责的组件与状态。

Args:
    spec: 环境给出的联合/角色规格；包含 N/O/A/S 和完整事件 horizon。
    config: 冻结算法/求解配置，控制网络或 damping、残差阈值及迭代上限。

Returns:
    None；绑定组件/状态，不开始环境交互或参数更新。
```

<a id="snmappo-act"></a>

## SNMAPPO.act

[源码位置](../../marl/algorithms/sn_mappo.py#L129) · [页内目录](#符号目录)

```python
def act(self, observations: Tensor, *, deterministic: bool=False, action_mask: Tensor | None=None, **kwargs: Tensor) -> Tensor
```

```text
不返回。

Args:
    observations: 联合观测 [...,N,O]；顺序算法要求分阶段采样。
    deterministic: 是否确定性动作；本联合 act 入口不执行采样。
    action_mask: 连续动作只能为 None。
    kwargs: 基类动作接口保留的关键字；此入口不使用。

Returns:
    不返回；抛 RuntimeError，必须使用顺序角色 sample。
```

<a id="snmappo-compute_loss_bundle"></a>

## SNMAPPO.compute_loss_bundle

[源码位置](../../marl/algorithms/sn_mappo.py#L144) · [页内目录](#符号目录)

```python
def compute_loss_bundle(self, batch: MARLBatch) -> LossBundle
```

```text
不返回。

Args:
    batch: 角色 MARLBatch；循环 [B,T,N,O]，否则 [T*B,N,O]，extras 含旧概率/GAE/raw_actions。

Returns:
    不返回；抛 RuntimeError，角色 loss 和阶段更新有不同数据契约。
```

<a id="snmappo-response_objectives"></a>

## SNMAPPO.response_objectives

[源码位置](../../marl/algorithms/sn_mappo.py#L155) · [页内目录](#符号目录)

```python
def response_objectives(self, batches: tuple[MARLBatch, MARLBatch, MARLBatch]) -> tuple[LossBundle, LossBundle]
```

```text
同一 fresh 批次构造上下层目标；直接项和隐式项共用这些图。

Args:
    batches: UC/coordinator/consumer 数据，GRU 为 [B,T,N,O]，MLP 为
        按完整 episode 排列的 [B*T,N,O]。每条轨迹的奖励、动作和 old logp 已 detach。

Returns:
    (upper, lower) LossBundle；经济回报除以 sum(gamma**t)，KL 对时间取均值。
    baseline 只用同场景其他独立轨迹的奖励，不使用本轨迹 GAE/critic。
```

<a id="snmappo-response_objectives-objective"></a>

## SNMAPPO.response_objectives.objective

[源码位置](../../marl/algorithms/sn_mappo.py#L187) · [页内目录](#符号目录)

```python
def objective(reward: Tensor, kl: Tensor) -> LossBundle
```

```text
保留逐轨迹因果依赖，构造最小化目标。

Args:
    reward: detached 经济奖励 [B,T]；下层已显式求消费者总收益。
    kl: 该角色策略的标量 KL 正则，保留参数计算图。

Returns:
    LossBundle，包含经济目标、KL 与总目标；不更新参数。
```

<a id="snmappo-update"></a>

## SNMAPPO.update

[源码位置](../../marl/algorithms/sn_mappo.py#L205) · [页内目录](#符号目录)

```python
def update(self, rollout: SequentialRollout, runtimes: tuple[OptimizerRuntime, OptimizerRuntime, OptimizerRuntime]) -> dict[str, float]
```

```text
命名 float 指标。

Args:
    rollout: 包含三个角色 fresh PreparedRollout、阶段和策略版本的 SequentialRollout。
    runtimes: leader/coordinator/followers 的三个 OptimizerRuntime，参数集合互不重叠。

Returns:
    命名 float 指标；成功消费新鲜阶段数据并推进策略版本/下一阶段。
```
