# marl/envs/demand_response.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/envs/demand_response.py)

一致性修订的 UC/Consumer 电力市场；算法无关的阶段和确定性清算。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [DemandResponseConfig](#demandresponseconfig)
- [DemandResponseConfig.__post_init__](#demandresponseconfig-__post_init__)
- [retail_tariff](#retail_tariff)
- [quadratic_ratio](#quadratic_ratio)
- [LeaderCommitment](#leadercommitment)
- [LeaderCommitment.array](#leadercommitment-array)
- [DemandResponseEnv](#demandresponseenv)
- [DemandResponseEnv.__init__](#demandresponseenv-__init__)
- [DemandResponseEnv.spec](#demandresponseenv-spec)
- [DemandResponseEnv.leader_spec](#demandresponseenv-leader_spec)
- [DemandResponseEnv.follower_spec](#demandresponseenv-follower_spec)
- [DemandResponseEnv.sequential_spec](#demandresponseenv-sequential_spec)
- [DemandResponseEnv.checkpoint_context](#demandresponseenv-checkpoint_context)
- [DemandResponseEnv.reset](#demandresponseenv-reset)
- [DemandResponseEnv._observation](#demandresponseenv-_observation)
- [DemandResponseEnv._shift_bounds](#demandresponseenv-_shift_bounds)
- [DemandResponseEnv._actions](#demandresponseenv-_actions)
- [DemandResponseEnv.commit_leader](#demandresponseenv-commit_leader)
- [DemandResponseEnv._project](#demandresponseenv-_project)
- [DemandResponseEnv.settle](#demandresponseenv-settle)
- [DemandResponseEnv.step](#demandresponseenv-step)
- [FixedLeaderDemandAdapter](#fixedleaderdemandadapter)
- [FixedLeaderDemandAdapter.__init__](#fixedleaderdemandadapter-__init__)
- [FixedLeaderDemandAdapter.spec](#fixedleaderdemandadapter-spec)
- [FixedLeaderDemandAdapter._followers](#fixedleaderdemandadapter-_followers)
- [FixedLeaderDemandAdapter.reset](#fixedleaderdemandadapter-reset)
- [FixedLeaderDemandAdapter.step](#fixedleaderdemandadapter-step)
- [DailyDemandAdapter](#dailydemandadapter)
- [DailyDemandAdapter.__init__](#dailydemandadapter-__init__)
- [DailyDemandAdapter._build](#dailydemandadapter-_build)
- [DailyDemandAdapter.spec](#dailydemandadapter-spec)
- [DailyDemandAdapter.reset](#dailydemandadapter-reset)
- [DailyDemandAdapter.step](#dailydemandadapter-step)
- [DailyDemandAdapter.sequential_spec](#dailydemandadapter-sequential_spec)
- [DailyDemandAdapter.checkpoint_context](#dailydemandadapter-checkpoint_context)
- [DailyDemandAdapter.commit_leader](#dailydemandadapter-commit_leader)
- [DailyDemandAdapter.settle](#dailydemandadapter-settle)

<a id="demandresponseconfig"></a>

## DemandResponseConfig

[源码位置](../../marl/envs/demand_response.py#L14) · [页内目录](#符号目录)

`class DemandResponseConfig()`

功率 kW、能量 kWh、价格/收益系数按小时；未给出的系数属于公开假设。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
storage_kwh: float = 4.0
storage_kw: float = 2.0
charge_efficiency: float = 0.8
discharge_efficiency: float = 0.9
initial_soc: float = 0.5
decay_per_step: float = 0.0
taper_soc: float = 0.8
taper_ratio: float = 0.5
dr_fraction: float = 0.2
denominator_fraction: float = 0.001
enable_dr: bool = True
purchase_price: float = 0.04
der_intercept: float = 0.025
der_slope: float = 0.001
curtailment_price: float = 0.01
dr_benefit: float = 0.01
dr_incentive: float = 0.05
comfort_price: float = 0.01
projection_tolerance: float = 1e-07
```

<a id="demandresponseconfig-__post_init__"></a>

## DemandResponseConfig.__post_init__

[源码位置](../../marl/envs/demand_response.py#L36) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

验证数值范围、形状和时间间隔；非法物理或数据配置尽早抛 ValueError。

<a id="retail_tariff"></a>

## retail_tariff

[源码位置](../../marl/envs/demand_response.py#L56) · [页内目录](#符号目录)

```python
def retail_tariff(local_slot: int, cumulative_kwh: np.ndarray) -> np.ndarray
```

Table II 三档价格，返回逐消费者价格 [N]；月初累计 UC 购电为零。

<a id="quadratic_ratio"></a>

## quadratic_ratio

[源码位置](../../marl/envs/demand_response.py#L67) · [页内目录](#符号目录)

```python
def quadratic_ratio(adjustment: np.ndarray, denominator: np.ndarray) -> np.ndarray
```

显式定义 d²/u：零/零为零；任何非零 d 配零 u 拒绝，不加 epsilon。

<a id="leadercommitment"></a>

## LeaderCommitment

[源码位置](../../marl/envs/demand_response.py#L81) · [页内目录](#符号目录)

`class LeaderCommitment()`

本时段公开承诺；pd=DER→ESS，pe=PG→ESS，po=ESS→UC 渠道。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
pc: float
pd: float
pe: float
po: float
pm: float
der_offer: float
```

<a id="leadercommitment-array"></a>

## LeaderCommitment.array

[源码位置](../../marl/envs/demand_response.py#L90) · [页内目录](#符号目录)

```python
def array(self) -> np.ndarray
```

返回承诺向量 [pc,pd,pe,po,pm,der_offer]，形状 [6]，单位 kW。

<a id="demandresponseenv"></a>

## DemandResponseEnv

[源码位置](../../marl/envs/demand_response.py#L95) · [页内目录](#符号目录)

`class DemandResponseEnv(EnvironmentAdapter)`

阶段顺序 reset → commit_leader → settle；仅 settle 推进 15 分钟。

普通 step([1+N,4]) 是同步基线适配：UC 与消费者同时看前一结算，消费者
没有当前承诺。顺序算法在 commit 后读取 follower_observations [N,14]。
仅支持完整本地自然日窗口（96 步倍数），无夏令时切换；月数据维持 SOC。

<a id="demandresponseenv-__init__"></a>

## DemandResponseEnv.__init__

[源码位置](../../marl/envs/demand_response.py#L104) · [页内目录](#符号目录)

```python
def __init__(self, profiles: DemandProfiles, config: DemandResponseConfig | None=None)
```

绑定经验证的数据和环境配置，声明静态尺寸；不执行动作或训练参数更新。

<a id="demandresponseenv-spec"></a>

## DemandResponseEnv.spec

[源码位置](../../marl/envs/demand_response.py#L125) · [页内目录](#符号目录)

```python
def spec(self) -> EnvironmentSpec
```

返回静态 EnvironmentSpec；智能体数、观测/动作/state 尺寸和 horizon 由当前环境决定。

<a id="demandresponseenv-leader_spec"></a>

## DemandResponseEnv.leader_spec

[源码位置](../../marl/envs/demand_response.py#L130) · [页内目录](#符号目录)

```python
def leader_spec(self) -> EnvironmentSpec
```

返回单 UC 的 [1,14] 观测、[1,4] 动作和公共全局 state 规格。

<a id="demandresponseenv-follower_spec"></a>

## DemandResponseEnv.follower_spec

[源码位置](../../marl/envs/demand_response.py#L136) · [页内目录](#符号目录)

```python
def follower_spec(self) -> EnvironmentSpec
```

返回固定 UC 下的消费者规格；观测 [N,14]、动作 [N,4]，逐消费者奖励。

<a id="demandresponseenv-sequential_spec"></a>

## DemandResponseEnv.sequential_spec

[源码位置](../../marl/envs/demand_response.py#L142) · [页内目录](#符号目录)

```python
def sequential_spec(self) -> SequentialSpec
```

返回联合、UC、随机计划和条件消费者规格；消费者输入追加广播计划维度。

<a id="demandresponseenv-checkpoint_context"></a>

## DemandResponseEnv.checkpoint_context

[源码位置](../../marl/envs/demand_response.py#L151) · [页内目录](#符号目录)

```python
def checkpoint_context(self) -> dict[str, object]
```

返回数据 SHA、时间窗口、训练尺度及物理配置，用于拒绝不同数据/环境的 checkpoint 恢复。

<a id="demandresponseenv-reset"></a>

## DemandResponseEnv.reset

[源码位置](../../marl/envs/demand_response.py#L158) · [页内目录](#符号目录)

```python
def reset(self, seed: int | None=None) -> EnvironmentStep
```

按 seed 选择完整日或重置固定事件，清空债务/累计购电和上次动作；返回统一初始观测。

<a id="demandresponseenv-_observation"></a>

## DemandResponseEnv._observation

[源码位置](../../marl/envs/demand_response.py#L169) · [页内目录](#符号目录)

```python
def _observation(self, rewards: np.ndarray, info: dict) -> EnvironmentStep
```

将当前可观测功率、时间、SOC、移峰债务、前次实绩及已公开承诺编码为
[1+N,14]，并验证 EnvironmentStep。

<a id="demandresponseenv-_shift_bounds"></a>

## DemandResponseEnv._shift_bounds

[源码位置](../../marl/envs/demand_response.py#L196) · [页内目录](#符号目录)

```python
def _shift_bounds(self) -> tuple[np.ndarray, np.ndarray]
```

返回各消费者本步可行移峰上下界 [N]；仅用当前负荷、现有债务和训练确定的未来偿还容量。

<a id="demandresponseenv-_actions"></a>

## DemandResponseEnv._actions

[源码位置](../../marl/envs/demand_response.py#L211) · [页内目录](#符号目录)

```python
def _actions(self, actions: np.ndarray, shape: tuple[int, ...]) -> np.ndarray
```

验证指定形状的请求为有限 [-1,1]，返回 float NumPy 数组；不裁剪非法输入。

<a id="demandresponseenv-commit_leader"></a>

## DemandResponseEnv.commit_leader

[源码位置](../../marl/envs/demand_response.py#L218) · [页内目录](#符号目录)

```python
def commit_leader(self, action: np.ndarray) -> EnvironmentStep
```

提交 [4] UC 请求并返回当前公开承诺；不得重复承诺或提前推进时间。

<a id="demandresponseenv-_project"></a>

## DemandResponseEnv._project

[源码位置](../../marl/envs/demand_response.py#L248) · [页内目录](#符号目录)

```python
def _project(self, requests: np.ndarray) -> tuple[np.ndarray, float]
```

用 SciPy SLSQP 将消费者请求投影至固定模式下的线性可行域，返回成交
[N,4] 和最大约束残差。

<a id="demandresponseenv-settle"></a>

## DemandResponseEnv.settle

[源码位置](../../marl/envs/demand_response.py#L311) · [页内目录](#符号目录)

```python
def settle(self, actions: np.ndarray) -> EnvironmentStep
```

清算 [N,4] 请求，返回逐个体利润/负成本和完整物理经济账本。

<a id="demandresponseenv-step"></a>

## DemandResponseEnv.step

[源码位置](../../marl/envs/demand_response.py#L370) · [页内目录](#符号目录)

```python
def step(self, actions: np.ndarray) -> EnvironmentStep
```

执行一次环境动作并返回统一下一步；仅一次 settle 推进物理时间，固定 UC
wrapper 会准备下一承诺。

<a id="fixedleaderdemandadapter"></a>

## FixedLeaderDemandAdapter

[源码位置](../../marl/envs/demand_response.py#L380) · [页内目录](#符号目录)

`class FixedLeaderDemandAdapter(EnvironmentAdapter)`

固定已公开 UC 承诺，仅训练 N 个消费者；复用标准 OnPolicyTrainer。

<a id="fixedleaderdemandadapter-__init__"></a>

## FixedLeaderDemandAdapter.__init__

[源码位置](../../marl/envs/demand_response.py#L382) · [页内目录](#符号目录)

```python
def __init__(self, environment: DemandResponseEnv, leader_action: tuple[float, ...]=(0.0, 0.0, 1.0, 0.0))
```

绑定经验证的数据和环境配置，声明静态尺寸；不执行动作或训练参数更新。

<a id="fixedleaderdemandadapter-spec"></a>

## FixedLeaderDemandAdapter.spec

[源码位置](../../marl/envs/demand_response.py#L389) · [页内目录](#符号目录)

```python
def spec(self) -> EnvironmentSpec
```

返回静态 EnvironmentSpec；智能体数、观测/动作/state 尺寸和 horizon 由当前环境决定。

<a id="fixedleaderdemandadapter-_followers"></a>

## FixedLeaderDemandAdapter._followers

[源码位置](../../marl/envs/demand_response.py#L393) · [页内目录](#符号目录)

```python
def _followers(self, step: EnvironmentStep) -> EnvironmentStep
```

从联合 EnvironmentStep 提取消费者观测/个体奖励，保留公共 state、终止标记和审计 info。

<a id="fixedleaderdemandadapter-reset"></a>

## FixedLeaderDemandAdapter.reset

[源码位置](../../marl/envs/demand_response.py#L400) · [页内目录](#符号目录)

```python
def reset(self, seed: int | None=None) -> EnvironmentStep
```

按 seed 选择完整日或重置固定事件，清空债务/累计购电和上次动作；返回统一初始观测。

<a id="fixedleaderdemandadapter-step"></a>

## FixedLeaderDemandAdapter.step

[源码位置](../../marl/envs/demand_response.py#L405) · [页内目录](#符号目录)

```python
def step(self, actions: np.ndarray) -> EnvironmentStep
```

执行一次环境动作并返回统一下一步；仅一次 settle 推进物理时间，固定 UC
wrapper 会准备下一承诺。

<a id="dailydemandadapter"></a>

## DailyDemandAdapter

[源码位置](../../marl/envs/demand_response.py#L418) · [页内目录](#符号目录)

`class DailyDemandAdapter(EnvironmentAdapter)`

按 reset seed 在无插补的完整训练日中抽样，复用相同静态 spec。

<a id="dailydemandadapter-__init__"></a>

## DailyDemandAdapter.__init__

[源码位置](../../marl/envs/demand_response.py#L420) · [页内目录](#符号目录)

```python
def __init__(self, profiles: DemandProfiles, config: DemandResponseConfig | None=None, fixed_leader: tuple[float, ...] | None=None)
```

绑定经验证的数据和环境配置，声明静态尺寸；不执行动作或训练参数更新。

<a id="dailydemandadapter-_build"></a>

## DailyDemandAdapter._build

[源码位置](../../marl/envs/demand_response.py#L432) · [页内目录](#符号目录)

```python
def _build(self, start: int) -> EnvironmentAdapter
```

从已验证的 start 位置构造完整 96 步环境，可选包装固定 UC；不拟合新的数据尺度。

<a id="dailydemandadapter-spec"></a>

## DailyDemandAdapter.spec

[源码位置](../../marl/envs/demand_response.py#L439) · [页内目录](#符号目录)

```python
def spec(self) -> EnvironmentSpec
```

返回静态 EnvironmentSpec；智能体数、观测/动作/state 尺寸和 horizon 由当前环境决定。

<a id="dailydemandadapter-reset"></a>

## DailyDemandAdapter.reset

[源码位置](../../marl/envs/demand_response.py#L443) · [页内目录](#符号目录)

```python
def reset(self, seed: int | None=None) -> EnvironmentStep
```

按 seed 选择完整日或重置固定事件，清空债务/累计购电和上次动作；返回统一初始观测。

<a id="dailydemandadapter-step"></a>

## DailyDemandAdapter.step

[源码位置](../../marl/envs/demand_response.py#L451) · [页内目录](#符号目录)

```python
def step(self, actions: np.ndarray) -> EnvironmentStep
```

执行一次环境动作并返回统一下一步；仅一次 settle 推进物理时间，固定 UC
wrapper 会准备下一承诺。

<a id="dailydemandadapter-sequential_spec"></a>

## DailyDemandAdapter.sequential_spec

[源码位置](../../marl/envs/demand_response.py#L459) · [页内目录](#符号目录)

```python
def sequential_spec(self) -> SequentialSpec
```

返回联合、UC、随机计划和条件消费者规格；消费者输入追加广播计划维度。

<a id="dailydemandadapter-checkpoint_context"></a>

## DailyDemandAdapter.checkpoint_context

[源码位置](../../marl/envs/demand_response.py#L466) · [页内目录](#符号目录)

```python
def checkpoint_context(self) -> dict[str, object]
```

返回数据 SHA、时间窗口、训练尺度及物理配置，用于拒绝不同数据/环境的 checkpoint 恢复。

<a id="dailydemandadapter-commit_leader"></a>

## DailyDemandAdapter.commit_leader

[源码位置](../../marl/envs/demand_response.py#L473) · [页内目录](#符号目录)

```python
def commit_leader(self, action: np.ndarray) -> EnvironmentStep
```

提交 [4] UC 请求，返回含公开承诺的响应阶段观测；不推进物理时间。

<a id="dailydemandadapter-settle"></a>

## DailyDemandAdapter.settle

[源码位置](../../marl/envs/demand_response.py#L479) · [页内目录](#符号目录)

```python
def settle(self, actions: np.ndarray) -> EnvironmentStep
```

在已承诺 UC 动作下清算消费者 [N,4] 请求，推进一次物理步，返回逐个体奖励及账本。
