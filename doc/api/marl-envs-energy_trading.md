# marl/envs/energy_trading.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/envs/energy_trading.py)

参考 Ye 等人 P2P 电能交易论文的可运行社区环境。

论文环境包含半小时交易、EV/储能/空调/智能电器、mid-market-rate 结算，以及
成本、舒适度、EV 出行和网络峰值惩罚。本模块实现这些机制的可配置基础版本。

论文使用真实 Ausgrid 等数据及各家庭异质参数；本项目默认生成合成曲线，仅用于
验证接口和训练流程。要复现实验，应把真实曲线和精确设备参数传入本环境。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [EnergyTradingConfig](#energytradingconfig)
- [EnergyTradingConfig.__post_init__](#energytradingconfig-__post_init__)
- [EnergyProfiles](#energyprofiles)
- [EnergyProfiles.validate](#energyprofiles-validate)
- [EnergyProfiles.synthetic](#energyprofiles-synthetic)
- [EnergyTradingResult](#energytradingresult)
- [EnergyTradingEnv](#energytradingenv)
- [EnergyTradingEnv.__init__](#energytradingenv-__init__)
- [EnergyTradingEnv.ownership](#energytradingenv-ownership)
- [EnergyTradingEnv.reset](#energytradingenv-reset)
- [EnergyTradingEnv.ev_available](#energytradingenv-ev_available)
- [EnergyTradingEnv.sa_can_start](#energytradingenv-sa_can_start)
- [EnergyTradingEnv._history](#energytradingenv-_history)
- [EnergyTradingEnv.observe](#energytradingenv-observe)
- [EnergyTradingEnv._storage_power](#energytradingenv-_storage_power)
- [EnergyTradingEnv._market_prices](#energytradingenv-_market_prices)
- [EnergyTradingEnv.step](#energytradingenv-step)

<a id="energytradingconfig"></a>

## EnergyTradingConfig

[源码位置](../../marl/envs/energy_trading.py#L21) · [页内目录](#符号目录)

`class EnergyTradingConfig()`

环境物理和市场参数；功率单位 kW，电量单位 kWh，时间单位小时。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
num_agents: int = 3
horizon: int = 48
history_steps: int = 48
time_step_hours: float = 0.5
grid_limit_kw: float = 8.0
peak_penalty: float = 100.0
comfort_penalty: float = 5.0
ev_departure_penalty: float = 5.0
comfort_min_c: float = 20.0
comfort_max_c: float = 25.0
initial_indoor_c: float = 23.0
hvac_max_kw: float = 2.0
hvac_thermal_capacity: float = 4.0
hvac_thermal_resistance: float = 3.0
hvac_cooling_efficiency: float = 2.5
ev_capacity_kwh: float = 12.0
ev_initial_kwh: float = 6.0
ev_max_kw: float = 3.0
ev_efficiency: float = 0.9
ev_departure_step: int = 16
ev_arrival_step: int = 34
ev_trip_kwh: float = 4.0
storage_capacity_kwh: float = 8.0
storage_initial_kwh: float = 4.0
storage_max_kw: float = 2.0
storage_efficiency: float = 0.9
sa_earliest_step: int = 8
sa_latest_start_step: int = 39
sa_cycle_kw: tuple[float, ...] = (1.5, 1.0)
has_ev: tuple[bool, ...] | None = None
has_storage: tuple[bool, ...] | None = None
has_hvac: tuple[bool, ...] | None = None
has_sa: tuple[bool, ...] | None = None
```

<a id="energytradingconfig-__post_init__"></a>

## EnergyTradingConfig.__post_init__

[源码位置](../../marl/envs/energy_trading.py#L59) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

在 EnergyTradingConfig 数据类构造后校验字段范围/一致性；返回 None，非法配置立即报错。

<a id="energyprofiles"></a>

## EnergyProfiles

[源码位置](../../marl/envs/energy_trading.py#L85) · [页内目录](#符号目录)

`class EnergyProfiles()`

可替换的外生时序；真实论文数据可从 CSV 载入后填在这里。

demand_kw/pv_kw 形状 ``[T,N]``；价格/室外温度形状 ``[T]``。
同一时刻所有家庭使用相同零售价格和室外温度，但负荷/PV 各不相同。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
demand_kw: FloatArray
pv_kw: FloatArray
buy_price: FloatArray
sell_price: FloatArray
outdoor_c: FloatArray
```

<a id="energyprofiles-validate"></a>

## EnergyProfiles.validate

[源码位置](../../marl/envs/energy_trading.py#L98) · [页内目录](#符号目录)

```python
def validate(self, horizon: int, num_agents: int) -> None
```

依据配置验证外部能源曲线形状、范围和有限性；无返回，非法数据提前报错。

<a id="energyprofiles-synthetic"></a>

## EnergyProfiles.synthetic

[源码位置](../../marl/envs/energy_trading.py#L115) · [页内目录](#符号目录)

```python
def synthetic(cls, config: EnergyTradingConfig, seed: int | None=None) -> EnergyProfiles
```

生成能立即运行的小型示例曲线；它们不是论文的真实数据。

<a id="energytradingresult"></a>

## EnergyTradingResult

[源码位置](../../marl/envs/energy_trading.py#L132) · [页内目录](#符号目录)

`class EnergyTradingResult()`

环境内部一步结果；适配器负责把它包装成通用 EnvironmentStep。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
rewards: NDArray[np.float32]
terminated: bool
info: dict[str, float]
```

<a id="energytradingenv"></a>

## EnergyTradingEnv

[源码位置](../../marl/envs/energy_trading.py#L140) · [页内目录](#符号目录)

`class EnergyTradingEnv()`

P2P 电力交易的环境动力学与市场结算，不依赖任何 MARL 算法。

<a id="energytradingenv-__init__"></a>

## EnergyTradingEnv.__init__

[源码位置](../../marl/envs/energy_trading.py#L143) · [页内目录](#符号目录)

```python
def __init__(self, config: EnergyTradingConfig, profiles: EnergyProfiles | None=None) -> None
```

构造 EnergyTradingEnv，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="energytradingenv-ownership"></a>

## EnergyTradingEnv.ownership

[源码位置](../../marl/envs/energy_trading.py#L157) · [页内目录](#符号目录)

```python
def ownership(self, name: str) -> NDArray[np.bool_]
```

输入设备名称 name，返回逐家庭是否拥有该设备的 bool [N] 数组。

<a id="energytradingenv-reset"></a>

## EnergyTradingEnv.reset

[源码位置](../../marl/envs/energy_trading.py#L161) · [页内目录](#符号目录)

```python
def reset(self, seed: int | None=None) -> None
```

一个 episode 对应一天。固定曲线不随 seed 改变；合成曲线会重新生成。

<a id="energytradingenv-ev_available"></a>

## EnergyTradingEnv.ev_available

[源码位置](../../marl/envs/energy_trading.py#L172) · [页内目录](#符号目录)

```python
def ev_available(self) -> NDArray[np.bool_]
```

返回当前时刻各家庭 EV 是否接入的布尔数组；不涉及策略网络。

<a id="energytradingenv-sa_can_start"></a>

## EnergyTradingEnv.sa_can_start

[源码位置](../../marl/envs/energy_trading.py#L178) · [页内目录](#符号目录)

```python
def sa_can_start(self) -> NDArray[np.bool_]
```

返回各家庭可移时负荷是否能在当前时刻启动的布尔数组。

<a id="energytradingenv-_history"></a>

## EnergyTradingEnv._history

[源码位置](../../marl/envs/energy_trading.py#L188) · [页内目录](#符号目录)

```python
def _history(values: FloatArray, step: int, length: int) -> FloatArray
```

仅使用当前及过去数据；episode 前不足 length 的部分用首值填充。

<a id="energytradingenv-observe"></a>

## EnergyTradingEnv.observe

[源码位置](../../marl/envs/energy_trading.py#L195) · [页内目录](#符号目录)

```python
def observe(self) -> FloatArray
```

局部观测 [N,O]；全局 state 由适配器按论文设为联合观测。

<a id="energytradingenv-_storage_power"></a>

## EnergyTradingEnv._storage_power

[源码位置](../../marl/envs/energy_trading.py#L220) · [页内目录](#符号目录)

```python
def _storage_power(self, action: FloatArray, energy: FloatArray, capacity: float, max_kw: float, efficiency: float, available: NDArray[np.bool_]) -> tuple[FloatArray, FloatArray]
```

裁剪功率以满足 SoC 边界；返回实际净负荷和下一时刻电量。

<a id="energytradingenv-_market_prices"></a>

## EnergyTradingEnv._market_prices

[源码位置](../../marl/envs/energy_trading.py#L244) · [页内目录](#符号目录)

```python
def _market_prices(net_load: FloatArray, supplier_buy: float, supplier_sell: float) -> tuple[float, float]
```

论文式 mid-market-rate：价格差额按消费者/生产者净负荷比例分摊。

<a id="energytradingenv-step"></a>

## EnergyTradingEnv.step

[源码位置](../../marl/envs/energy_trading.py#L258) · [页内目录](#符号目录)

```python
def step(self, actions: FloatArray) -> EnergyTradingResult
```

执行 [N,4] 控制量，顺序为 EV、储能、HVAC、智能电器。
