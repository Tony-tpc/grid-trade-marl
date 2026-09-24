"""参考 Ye 等人 P2P 电能交易论文的可运行社区环境。

论文环境包含半小时交易、EV/储能/空调/智能电器、mid-market-rate 结算，以及
成本、舒适度、EV 出行和网络峰值惩罚。本模块实现这些机制的可配置基础版本。

论文使用真实 Ausgrid 等数据及各家庭异质参数；本项目默认生成合成曲线，仅用于
验证接口和训练流程。要复现实验，应把真实曲线和精确设备参数传入本环境。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class EnergyTradingConfig:
    """环境物理和市场参数；功率单位 kW，电量单位 kWh，时间单位小时。"""

    num_agents: int = 3
    horizon: int = 48  # 论文每日 48 个半小时交易时段。
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
    # None 表示所有家庭都有该类设备；传 tuple 可模拟家庭资产异质性。
    has_ev: tuple[bool, ...] | None = None
    has_storage: tuple[bool, ...] | None = None
    has_hvac: tuple[bool, ...] | None = None
    has_sa: tuple[bool, ...] | None = None

    def __post_init__(self) -> None:
        if self.num_agents < 1 or self.horizon < 2 or self.history_steps < 1:
            raise ValueError("num_agents/horizon/history_steps 必须为正，horizon 至少为 2")
        if self.time_step_hours <= 0 or self.grid_limit_kw <= 0:
            raise ValueError("交易时长和电网功率阈值必须大于 0")
        if not 0 < self.ev_efficiency <= 1 or not 0 < self.storage_efficiency <= 1:
            raise ValueError("充放电效率必须位于 (0, 1]")
        if not 0 <= self.ev_initial_kwh <= self.ev_capacity_kwh:
            raise ValueError("EV 初始电量必须在容量范围内")
        if not 0 <= self.storage_initial_kwh <= self.storage_capacity_kwh:
            raise ValueError("储能初始电量必须在容量范围内")
        if not 0 <= self.ev_departure_step < self.ev_arrival_step < self.horizon:
            raise ValueError("EV 离开/归来时段必须依次落在 horizon 内")
        if not 0 <= self.sa_earliest_step <= self.sa_latest_start_step:
            raise ValueError("智能电器启动时间窗口无效")
        if self.sa_latest_start_step + len(self.sa_cycle_kw) > self.horizon:
            raise ValueError("智能电器最晚启动后必须能在当天结束前完成")
        if not self.sa_cycle_kw or any(power < 0 for power in self.sa_cycle_kw):
            raise ValueError("智能电器功率曲线不能为空或包含负功率")
        for name in ("has_ev", "has_storage", "has_hvac", "has_sa"):
            ownership = getattr(self, name)
            if ownership is not None and len(ownership) != self.num_agents:
                raise ValueError(f"{name} 的长度必须等于 num_agents")


@dataclass(frozen=True, slots=True)
class EnergyProfiles:
    """可替换的外生时序；真实论文数据可从 CSV 载入后填在这里。

    demand_kw/pv_kw 形状 ``[T,N]``；价格/室外温度形状 ``[T]``。
    同一时刻所有家庭使用相同零售价格和室外温度，但负荷/PV 各不相同。
    """

    demand_kw: FloatArray
    pv_kw: FloatArray
    buy_price: FloatArray
    sell_price: FloatArray
    outdoor_c: FloatArray

    def validate(self, horizon: int, num_agents: int) -> None:
        if self.demand_kw.shape != (horizon, num_agents):
            raise ValueError("demand_kw 形状必须为 [horizon, num_agents]")
        if self.pv_kw.shape != (horizon, num_agents):
            raise ValueError("pv_kw 形状必须为 [horizon, num_agents]")
        for name in ("buy_price", "sell_price", "outdoor_c"):
            if getattr(self, name).shape != (horizon,):
                raise ValueError(f"{name} 形状必须为 [horizon]")
        arrays = (self.demand_kw, self.pv_kw, self.buy_price, self.sell_price, self.outdoor_c)
        if not all(np.isfinite(values).all() for values in arrays):
            raise ValueError("曲线包含 NaN 或无穷大")
        if (self.demand_kw < 0).any() or (self.pv_kw < 0).any():
            raise ValueError("负荷和 PV 出力不能为负")
        if (self.buy_price < self.sell_price).any() or (self.sell_price < 0).any():
            raise ValueError("供电商购电价必须不低于售电价，且售电价不能为负")

    @classmethod
    def synthetic(cls, config: EnergyTradingConfig, seed: int | None = None) -> EnergyProfiles:
        """生成能立即运行的小型示例曲线；它们不是论文的真实数据。"""
        rng = np.random.default_rng(seed)
        t = np.arange(config.horizon, dtype=np.float64)
        phase = 2 * np.pi * t / config.horizon
        demand = 1.0 + 0.25 * np.sin(phase - 0.7)[:, None]
        demand = np.maximum(0.1, demand + rng.normal(0, 0.12, (config.horizon, config.num_agents)))
        daylight = np.maximum(0.0, np.sin(phase - np.pi / 2))[:, None]
        pv_scale = rng.uniform(0.8, 1.8, (1, config.num_agents))
        pv = daylight * pv_scale
        buy = 0.22 + 0.10 * np.maximum(0.0, np.sin(phase - 2.0))
        sell = 0.08 + 0.02 * np.maximum(0.0, np.sin(phase - 2.0))
        outdoor = 28.0 + 4.0 * np.sin(phase - 1.2)
        return cls(demand, pv, buy, sell, outdoor)


@dataclass(slots=True)
class EnergyTradingResult:
    """环境内部一步结果；适配器负责把它包装成通用 EnvironmentStep。"""

    rewards: NDArray[np.float32]
    terminated: bool
    info: dict[str, float]


class EnergyTradingEnv:
    """P2P 电力交易的环境动力学与市场结算，不依赖任何 MARL 算法。"""

    def __init__(self, config: EnergyTradingConfig, profiles: EnergyProfiles | None = None) -> None:
        self.config = config
        self._fixed_profiles = profiles
        if profiles is not None:
            profiles.validate(config.horizon, config.num_agents)
        self.profiles = profiles or EnergyProfiles.synthetic(config)
        self.time_step = 0
        self.ev_energy = np.empty(config.num_agents)
        self.storage_energy = np.empty(config.num_agents)
        self.indoor_c = np.empty(config.num_agents)
        self.sa_started = np.zeros(config.num_agents, dtype=bool)
        self.sa_progress = np.zeros(config.num_agents, dtype=np.int64)
        self.reset()

    def ownership(self, name: str) -> NDArray[np.bool_]:
        values = getattr(self.config, f"has_{name}")
        return np.ones(self.config.num_agents, dtype=bool) if values is None else np.asarray(values)

    def reset(self, seed: int | None = None) -> None:
        """一个 episode 对应一天。固定曲线不随 seed 改变；合成曲线会重新生成。"""
        if self._fixed_profiles is None:
            self.profiles = EnergyProfiles.synthetic(self.config, seed)
        self.time_step = 0
        self.ev_energy = np.full(self.config.num_agents, self.config.ev_initial_kwh)
        self.storage_energy = np.full(self.config.num_agents, self.config.storage_initial_kwh)
        self.indoor_c = np.full(self.config.num_agents, self.config.initial_indoor_c)
        self.sa_started = np.zeros(self.config.num_agents, dtype=bool)
        self.sa_progress = np.zeros(self.config.num_agents, dtype=np.int64)

    def ev_available(self) -> NDArray[np.bool_]:
        t = self.time_step
        return self.ownership("ev") & (
            (t < self.config.ev_departure_step) | (t >= self.config.ev_arrival_step)
        )

    def sa_can_start(self) -> NDArray[np.bool_]:
        t = self.time_step
        return (
            self.ownership("sa")
            & ~self.sa_started
            & (t >= self.config.sa_earliest_step)
            & (t <= self.config.sa_latest_start_step)
        )

    @staticmethod
    def _history(values: FloatArray, step: int, length: int) -> FloatArray:
        """仅使用当前及过去数据；episode 前不足 length 的部分用首值填充。"""
        start = max(0, step - length + 1)
        segment = values[start : step + 1]
        pad = length - len(segment)
        return np.concatenate((np.repeat(values[0:1], pad, axis=0), segment), axis=0)

    def observe(self) -> FloatArray:
        """局部观测 [N,O]；全局 state 由适配器按论文设为联合观测。"""
        c, t, n = self.config, min(self.time_step, self.config.horizon - 1), self.config.num_agents
        columns: list[FloatArray] = [np.full((n, 1), t / (c.horizon - 1))]
        for series in (self.profiles.buy_price, self.profiles.sell_price):
            columns.append(
                np.broadcast_to(self._history(series, t, c.history_steps), (n, c.history_steps))
            )
        for series in (self.profiles.pv_kw, self.profiles.demand_kw):
            columns.append(self._history(series, t, c.history_steps).T)
        features = (
            self.ev_energy / c.ev_capacity_kwh,
            self.storage_energy / c.storage_capacity_kwh,
            self.ev_available().astype(float),
            np.full(n, self.profiles.outdoor_c[t] / 40.0),
            self.indoor_c / 40.0,
            self.sa_can_start().astype(float),
            self.ownership("ev").astype(float),
            self.ownership("storage").astype(float),
            self.ownership("hvac").astype(float),
            self.ownership("sa").astype(float),
        )
        columns.extend(value[:, None] for value in features)
        return np.concatenate(columns, axis=-1).astype(np.float32)

    def _storage_power(
        self,
        action: FloatArray,
        energy: FloatArray,
        capacity: float,
        max_kw: float,
        efficiency: float,
        available: NDArray[np.bool_],
    ) -> tuple[FloatArray, FloatArray]:
        """裁剪功率以满足 SoC 边界；返回实际净负荷和下一时刻电量。"""
        c = self.config
        requested = np.clip(action, -1.0, 1.0) * max_kw * available
        charge = np.minimum(
            np.maximum(requested, 0.0), (capacity - energy) / (efficiency * c.time_step_hours)
        )
        discharge = np.minimum(np.maximum(-requested, 0.0), energy * efficiency / c.time_step_hours)
        next_energy = (
            energy
            + charge * efficiency * c.time_step_hours
            - discharge / efficiency * c.time_step_hours
        )
        return charge - discharge, np.clip(next_energy, 0.0, capacity)

    @staticmethod
    def _market_prices(
        net_load: FloatArray, supplier_buy: float, supplier_sell: float
    ) -> tuple[float, float]:
        """论文式 mid-market-rate：价格差额按消费者/生产者净负荷比例分摊。"""
        demand = float(np.maximum(net_load, 0.0).sum())
        generation = float(-np.minimum(net_load, 0.0).sum())
        imbalance = demand - generation
        midpoint = (supplier_buy + supplier_sell) / 2.0
        if imbalance > 1e-10:
            return (midpoint * generation + supplier_buy * imbalance) / demand, midpoint
        if imbalance < -1e-10:
            return midpoint, (midpoint * demand + supplier_sell * -imbalance) / generation
        return midpoint, midpoint

    def step(self, actions: FloatArray) -> EnergyTradingResult:
        """执行 [N,4] 控制量，顺序为 EV、储能、HVAC、智能电器。"""
        c, t = self.config, self.time_step
        if t >= c.horizon:
            raise RuntimeError("当天已结束，请先 reset()")
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (c.num_agents, 4) or not np.isfinite(actions).all():
            raise ValueError("能源环境动作必须是无 NaN 的 [num_agents,4] 数组")

        ev_power, self.ev_energy = self._storage_power(
            actions[:, 0],
            self.ev_energy,
            c.ev_capacity_kwh,
            c.ev_max_kw,
            c.ev_efficiency,
            self.ev_available(),
        )
        storage_power, self.storage_energy = self._storage_power(
            actions[:, 1],
            self.storage_energy,
            c.storage_capacity_kwh,
            c.storage_max_kw,
            c.storage_efficiency,
            self.ownership("storage"),
        )
        hvac_power = np.clip(actions[:, 2], 0.0, 1.0) * c.hvac_max_kw * self.ownership("hvac")
        self.indoor_c += c.time_step_hours * (
            (self.profiles.outdoor_c[t] - self.indoor_c)
            / (c.hvac_thermal_capacity * c.hvac_thermal_resistance)
            - c.hvac_cooling_efficiency * hvac_power / c.hvac_thermal_capacity
        )

        # 智能电器只允许启动一次，之后按固定功率曲线连续运行到结束。
        can_start = self.sa_can_start()
        start = can_start & ((actions[:, 3] >= 0.5) | (t == c.sa_latest_start_step))
        self.sa_started |= start
        sa_power = np.zeros(c.num_agents)
        for index in range(c.num_agents):
            if self.sa_started[index] and self.sa_progress[index] < len(c.sa_cycle_kw):
                sa_power[index] = c.sa_cycle_kw[self.sa_progress[index]]
                self.sa_progress[index] += 1

        # 出行前检查 EV 能否满足用车需求；离开时再扣除实际行驶电量。
        departure_penalty = np.zeros(c.num_agents)
        if t == c.ev_departure_step:
            departure_penalty = (
                -c.ev_departure_penalty
                * np.maximum(c.ev_trip_kwh - self.ev_energy, 0.0)
                * self.ownership("ev")
            )
            self.ev_energy = np.maximum(self.ev_energy - c.ev_trip_kwh * self.ownership("ev"), 0.0)

        flexible_load = ev_power + storage_power + hvac_power + sa_power
        net_load = self.profiles.demand_kw[t] - self.profiles.pv_kw[t] + flexible_load
        total_load = float(net_load.sum())
        local_buy, local_sell = self._market_prices(
            net_load, float(self.profiles.buy_price[t]), float(self.profiles.sell_price[t])
        )
        energy_cost = c.time_step_hours * (
            local_buy * np.maximum(net_load, 0.0) + local_sell * np.minimum(net_load, 0.0)
        )
        comfort_penalty = -c.comfort_penalty * (
            np.maximum(self.indoor_c - c.comfort_max_c, 0.0)
            + np.maximum(c.comfort_min_c - self.indoor_c, 0.0)
        )
        peak_penalty = np.zeros(c.num_agents)
        if total_load > c.grid_limit_kw:
            contribution = np.maximum(flexible_load, 0.0)
            if contribution.sum() > 0:
                peak_penalty = -c.peak_penalty * contribution / contribution.sum()
        elif total_load < -c.grid_limit_kw:
            contribution = np.maximum(-flexible_load, 0.0)
            if contribution.sum() > 0:
                peak_penalty = -c.peak_penalty * contribution / contribution.sum()

        rewards = -energy_cost + comfort_penalty + departure_penalty + peak_penalty
        self.time_step += 1
        return EnergyTradingResult(
            rewards=rewards.astype(np.float32),
            terminated=self.time_step >= c.horizon,
            info={
                "community_net_kw": total_load,
                "community_energy_cost": float(energy_cost.sum()),
                "local_buy_price": local_buy,
                "local_sell_price": local_sell,
                "peak_penalty_total": float(peak_penalty.sum()),
                "comfort_penalty_total": float(comfort_penalty.sum()),
                "ev_departure_penalty_total": float(departure_penalty.sum()),
            },
        )
