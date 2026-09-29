# marl/envs/energy_trading_adapter.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/envs/energy_trading_adapter.py)

把电能交易环境转换为所有算法共用的观测、动作和 batch 协议。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [EnergyTradingAdapter](#energytradingadapter)
- [EnergyTradingAdapter.__init__](#energytradingadapter-__init__)
- [EnergyTradingAdapter.spec](#energytradingadapter-spec)
- [EnergyTradingAdapter._action_mask](#energytradingadapter-_action_mask)
- [EnergyTradingAdapter._snapshot](#energytradingadapter-_snapshot)
- [EnergyTradingAdapter.reset](#energytradingadapter-reset)
- [EnergyTradingAdapter.step](#energytradingadapter-step)

<a id="energytradingadapter"></a>

## EnergyTradingAdapter

[源码位置](../../marl/envs/energy_trading_adapter.py#L21) · [页内目录](#符号目录)

`class EnergyTradingAdapter(EnvironmentAdapter)`

论文环境专属适配器；算法只接触本类返回的 EnvironmentStep。

原论文同时使用 3 个连续动作和 1 个二元动作。目前本仓库的算法按纯离散/纯连续
动作设计，因此提供两种显式编码：

* discrete：把 EV/储能 {-1,0,1}、HVAC {0,0.5,1}、SA {0,1} 的笛卡尔积
  编成 54 个离散选项，供 MAAC/MAPPO 使用；这是动作量化近似。
  本环境保持个体奖励，因此 QMIX 即使支持离散动作，也不满足奖励兼容性。
* continuous：使用长度为 4 的连续动作，前 3 维映射到物理功率，最后一维用
  阈值决定 SA 是否启动，适合 MADDPG/MASAC；阈值是离散化近似。

两种编码均不声称精确复现论文中的混合 Bernoulli/Gaussian 策略。

<a id="energytradingadapter-__init__"></a>

## EnergyTradingAdapter.__init__

[源码位置](../../marl/envs/energy_trading_adapter.py#L36) · [页内目录](#符号目录)

```python
def __init__(self, environment: EnergyTradingEnv, action_kind: ActionKind) -> None
```

构造 EnergyTradingAdapter，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="energytradingadapter-spec"></a>

## EnergyTradingAdapter.spec

[源码位置](../../marl/envs/energy_trading_adapter.py#L73) · [页内目录](#符号目录)

```python
def spec(self) -> EnvironmentSpec
```

无输入；返回静态 EnvironmentSpec，作为所有算法尺寸与动作/奖励语义的唯一来源。

<a id="energytradingadapter-_action_mask"></a>

## EnergyTradingAdapter._action_mask

[源码位置](../../marl/envs/energy_trading_adapter.py#L76) · [页内目录](#符号目录)

```python
def _action_mask(self) -> NDArray[np.bool_] | None
```

根据当前设备可行性返回离散合法动作 bool [N,A]；连续动作返回 None。

<a id="energytradingadapter-_snapshot"></a>

## EnergyTradingAdapter._snapshot

[源码位置](../../marl/envs/energy_trading_adapter.py#L100) · [页内目录](#符号目录)

```python
def _snapshot(self, rewards: NDArray[np.float32], terminated: bool=False, info: dict[str, float] | None=None) -> EnvironmentStep
```

打包能源环境当前观测、state、收益和终止信息，返回经过 validate_step 的 EnvironmentStep。

<a id="energytradingadapter-reset"></a>

## EnergyTradingAdapter.reset

[源码位置](../../marl/envs/energy_trading_adapter.py#L121) · [页内目录](#符号目录)

```python
def reset(self, seed: int | None=None) -> EnvironmentStep
```

按可选 seed 重置环境并返回初始结果；adapter 返回统一 EnvironmentStep，底层协议返回原生格式。

<a id="energytradingadapter-step"></a>

## EnergyTradingAdapter.step

[源码位置](../../marl/envs/energy_trading_adapter.py#L125) · [页内目录](#符号目录)

```python
def step(self, actions: np.ndarray) -> EnvironmentStep
```

执行输入动作并推进环境；adapter 返回下一 EnvironmentStep，底层协议返回原生观测/奖励/终止信息。
