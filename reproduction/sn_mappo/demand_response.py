"""一致性修订的 UC/Consumer 电力市场；算法无关的阶段和确定性清算。"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from marl.envs.base import ActionKind, EnvironmentAdapter, EnvironmentSpec, EnvironmentStep
from marl.envs.sequential import SequentialSpec
from reproduction.sn_mappo.demand_response_data import DemandProfiles


@dataclass(frozen=True, slots=True)
class DemandResponseConfig:
    """功率 kW、能量 kWh、价格/收益系数按小时；未给出的系数属于公开假设。"""
    storage_kwh: float = 4.0
    storage_kw: float = 2.0
    charge_efficiency: float = .8
    discharge_efficiency: float = .9
    initial_soc: float = .5
    decay_per_step: float = 0.0
    taper_soc: float = .8
    taper_ratio: float = .5
    dr_fraction: float = .2
    denominator_fraction: float = 1e-3
    enable_dr: bool = True
    purchase_price: float = .04  # 作者实时价未公开；确定性诊断假设。
    der_intercept: float = .025  # 缺失参数的假设，不来自 Table II。
    der_slope: float = .001
    curtailment_price: float = .01
    dr_benefit: float = .01
    dr_incentive: float = .05
    comfort_price: float = .01
    projection_tolerance: float = 1e-7

    def __post_init__(self) -> None:
        """验证数值范围、形状和时间间隔；非法物理或数据配置尽早抛 ValueError。"""
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, (float, int)) and (not np.isfinite(value) or value < 0):
                raise ValueError(f'{name} 必须有限且非负')
        if self.storage_kwh <= 0 or self.storage_kw <= 0:
            raise ValueError('storage 容量和功率必须为正')
        if not 0 < self.charge_efficiency <= 1 or not 0 < self.discharge_efficiency <= 1:
            raise ValueError('efficiency 必须在 (0,1]')
        if not 0 <= self.initial_soc <= 1 or not 0 <= self.decay_per_step < 1:
            raise ValueError('SOC/decay 超出范围')
        if not 0 < self.taper_soc < 1 or not 0 < self.taper_ratio <= 1:
            raise ValueError('taper 参数超出范围')
        if not 0 < self.denominator_fraction < self.dr_fraction < 1:
            raise ValueError('要求 0 < denominator_fraction < dr_fraction < 1')
        if self.projection_tolerance <= 0:
            raise ValueError('projection_tolerance 必须为正')


def retail_tariff(local_slot: int, cumulative_kwh: np.ndarray) -> np.ndarray:
    """Table II 三档价格，返回逐消费者价格 [N]；月初累计 UC 购电为零。"""
    if not 0 <= local_slot <= 95 or (cumulative_kwh < 0).any():
        raise ValueError('slot/累计用电超出范围')
    tier = np.searchsorted([2.88e7, 4.8e7], cumulative_kwh, side='right')
    hour = local_slot / 4
    period = 0 if 20 <= hour < 22 else (1 if 9 <= hour < 15 else 2)
    prices = np.asarray([[.078, .068, .048], [.067, .059, .041], [.060, .053, .037]])
    return np.asarray(prices[tier, period])


def quadratic_ratio(adjustment: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """显式定义 d²/u：零/零为零；任何非零 d 配零 u 拒绝，不加 epsilon。"""
    if adjustment.shape != denominator.shape or (denominator < 0).any():
        raise ValueError('分子分母形状须相同且分母非负')
    if not np.isfinite(adjustment).all() or not np.isfinite(denominator).all():
        raise ValueError('经济输入必须有限')
    if ((denominator == 0) & (adjustment != 0)).any():
        raise ValueError('非零 DR 需要严格正分母')
    out = np.zeros_like(adjustment, dtype=float)
    np.divide(np.square(adjustment), denominator, out=out, where=denominator > 0)
    return out


@dataclass(frozen=True, slots=True)
class LeaderCommitment:
    """本时段公开承诺；pd=DER→ESS，pe=PG→ESS，po=ESS→UC 渠道。"""
    pc: float
    pd: float
    pe: float
    po: float
    pm: float
    der_offer: float

    def array(self) -> np.ndarray:
        """返回承诺向量 [pc,pd,pe,po,pm,der_offer]，形状 [6]，单位 kW。"""
        return np.asarray([self.pc, self.pd, self.pe, self.po, self.pm, self.der_offer])


class DemandResponseEnv(EnvironmentAdapter):
    """阶段顺序 reset → commit_leader → settle；仅 settle 推进 15 分钟。

    普通 step([1+N,4]) 是同步基线适配：UC 与消费者同时看前一结算，消费者
    没有当前承诺。顺序算法在 commit 后读取 follower_observations [N,14]。
    仅支持完整本地自然日窗口（96 步倍数），无夏令时切换；月数据维持 SOC。
    """
    dt = .25

    def __init__(self, profiles: DemandProfiles, config: DemandResponseConfig | None = None):
        """绑定经验证的数据和环境配置，声明静态尺寸；不执行动作或训练参数更新。"""
        self.profiles = profiles
        self.config = config or DemandResponseConfig()
        self.n = profiles.load_kw.shape[1]
        if len(profiles.load_kw) % 96 or not np.array_equal(
            profiles.local_slot, np.arange(len(profiles.load_kw)) % 96
        ):
            raise ValueError('环境需要完整 96 步本地日，不支持 DST 切换日')
        self.capacity = profiles.load_scale * self.config.dr_fraction
        self.domain_floor = profiles.load_scale * self.config.denominator_fraction
        self._spec = EnvironmentSpec(self.n + 1, 14, 4, (self.n + 1) * 14,
                                     ActionKind.CONTINUOUS, len(profiles.load_kw))
        self.time = 0
        self.commitment: LeaderCommitment | None = None
        self.energy = self.config.initial_soc * self.config.storage_kwh
        self.debt = np.zeros(self.n)
        self.cumulative = np.zeros(self.n)
        self.previous = np.zeros((self.n, 4))

    @property
    def spec(self) -> EnvironmentSpec:
        """返回静态 EnvironmentSpec；智能体数、观测/动作/state 尺寸和 horizon 由当前环境决定。"""
        return self._spec

    @property
    def leader_spec(self) -> EnvironmentSpec:
        """返回单 UC 的 [1,14] 观测、[1,4] 动作和公共全局 state 规格。"""
        return EnvironmentSpec(1, 14, 4, self.spec.state_dim, ActionKind.CONTINUOUS,
                               self.spec.horizon)

    @property
    def follower_spec(self) -> EnvironmentSpec:
        """返回固定 UC 下的消费者规格；观测 [N,14]、动作 [N,4]，逐消费者奖励。"""
        return EnvironmentSpec(self.n, 14, 4, self.spec.state_dim, ActionKind.CONTINUOUS,
                               self.spec.horizon)

    @property
    def sequential_spec(self) -> SequentialSpec:
        """返回联合、UC、随机计划和条件消费者规格；消费者输入追加广播计划维度。"""
        followers = EnvironmentSpec(self.n, self.spec.observation_dim+self.spec.action_dim,
                                    self.spec.action_dim, self.spec.state_dim,
                                    ActionKind.CONTINUOUS,
                                    self.spec.horizon)
        return SequentialSpec(self.spec, self.leader_spec, self.leader_spec, followers)

    @property
    def checkpoint_context(self) -> dict[str, object]:
        """返回数据 SHA、时间窗口、训练尺度及物理配置，用于拒绝不同数据/环境的 checkpoint 恢复。"""
        return {'source_sha256': self.profiles.source_sha256, 'config': asdict(self.config),
                'load_scale': self.profiles.load_scale.tolist(),
                'der_scale': self.profiles.der_scale, 'utc_start': str(self.profiles.utc[0]),
                'utc_end': str(self.profiles.utc[-1])}

    def reset(self, seed: int | None = None) -> EnvironmentStep:
        """按 seed 选择完整日或重置固定事件，清空债务/累计购电和上次动作；返回统一初始观测。"""
        del seed  # 历史数据、价格和清算确定；训练窗口抽样由实验入口的 seed 管理。
        self.time = 0
        self.energy = self.config.initial_soc * self.config.storage_kwh
        self.debt = np.zeros(self.n)
        self.cumulative = np.zeros(self.n)
        self.previous = np.zeros((self.n, 4))
        self.commitment = None
        return self._observation(np.zeros(self.n + 1), {})

    def _observation(self, rewards: np.ndarray, info: dict) -> EnvironmentStep:
        # 终点是真终止；占位终态沿用最后可观测数据，绝不读取窗口外测试/未来负荷。
        """
        将当前可观测功率、时间、SOC、移峰债务、前次实绩及已公开承诺编码为
        [1+N,14]，并验证 EnvironmentStep。
        """
        t = min(self.time, len(self.profiles.load_kw) - 1)
        load, der = self.profiles.load_kw[t], self.profiles.der_kw[t]
        phase = self.profiles.local_slot[t] / 96 * 2 * np.pi
        total_scale = self.profiles.load_scale.sum()
        common = [np.sin(phase), np.cos(phase), self.energy / self.config.storage_kwh,
                  der / self.profiles.der_scale]
        observations = np.zeros((self.n + 1, 14), dtype=np.float32)
        observations[:, :4] = common
        observations[0, 4:8] = [load.sum() / total_scale, self.debt.sum() / total_scale,
                                self.previous[:, 0].sum() / total_scale,
                                self.previous[:, 1].sum() / total_scale]
        observations[1:, 4] = load / self.profiles.load_scale
        observations[1:, 5] = self.debt / self.profiles.load_scale
        observations[1:, 6:8] = self.previous[:, :2] / self.profiles.load_scale[:, None]
        if self.commitment is not None:
            observations[:, 8:] = self.commitment.array() / total_scale
        return self.validate_step(EnvironmentStep(
            observations, observations.flatten(), np.asarray(rewards, dtype=np.float32),
            self.time == self.spec.horizon, False, info=info,
        ))

    def _shift_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """返回各消费者本步可行移峰上下界 [N]；仅用当前负荷、现有债务和训练确定的未来偿还容量。"""
        if not self.config.enable_dr:
            return np.zeros(self.n), np.zeros(self.n)
        remaining = 95 - int(self.profiles.local_slot[self.time])
        lower = np.maximum(-self.capacity, -self.debt / self.dt)
        upper = np.minimum.reduce([
            self.capacity, self.profiles.load_kw[self.time],
            remaining * self.capacity - self.debt / self.dt,
        ])
        if (lower-upper > self.config.projection_tolerance).any():
            raise RuntimeError('此前移峰决策已超出可偿还容量')
        upper = np.maximum(upper, lower)  # 仅消除可行边界的浮点舍入交叉。
        return lower, upper

    def _actions(self, actions: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        """验证指定形状的请求为有限 [-1,1]，返回 float NumPy 数组；不裁剪非法输入。"""
        value = np.asarray(actions, dtype=float)
        if value.shape != shape or not np.isfinite(value).all() or (np.abs(value) > 1).any():
            raise ValueError(f'动作必须是有限 [-1,1] {shape}')
        return value

    def commit_leader(self, action: np.ndarray) -> EnvironmentStep:
        """提交 [4] UC 请求并返回当前公开承诺；不得重复承诺或提前推进时间。"""
        if self.time >= self.spec.horizon or self.commitment is not None:
            raise RuntimeError('需要新的 leader 决策阶段')
        a = self._actions(action, (4,))
        c, load = self.config, self.profiles.load_kw[self.time]
        der = float(self.profiles.der_kw[self.time])
        lower, upper = self._shift_bounds()
        floor = np.minimum(self.domain_floor, load + self.debt / self.dt)
        if not c.enable_dr:
            floor[:] = 0
        initial_shift = np.minimum(np.clip(np.zeros(self.n), lower, upper), load - floor)
        supply = float((load - initial_shift).sum())
        available_energy = (1 - c.decay_per_step) * self.energy
        cap = c.storage_kw * (c.taper_ratio if available_energy >= c.taper_soc*c.storage_kwh
                             else 1.0)
        cap = min(cap, (c.storage_kwh - available_energy) / (c.charge_efficiency * self.dt))
        pc = max(float(a[0]), 0) * max(cap, 0)
        pd = min(pc * (a[1] + 1) / 2, der)
        po = max(-float(a[0]), 0) * min(c.storage_kw, available_energy * c.discharge_efficiency
                                      / self.dt, supply)
        pm = 0.0
        if c.enable_dr and supply > 0:
            maximum = min(float(self.capacity.sum()), supply)
            minimum = min(float(self.domain_floor.sum()), maximum)
            pm = minimum + (a[3] + 1) / 2 * (maximum - minimum)
        self.commitment = LeaderCommitment(pc, float(pd), float(pc-pd), po, pm,
                                            float((a[2]+1)/2 * (der-pd)))
        return self._observation(np.zeros(self.n + 1), {'phase': 'follower'})

    def _project(self, requests: np.ndarray) -> tuple[np.ndarray, float]:
        """
        用 SciPy SLSQP 将消费者请求投影至固定模式下的线性可行域，返回成交
        [N,4] 和最大约束残差。
        """
        from scipy.optimize import (  # type: ignore[import-untyped]
            Bounds,
            LinearConstraint,
            minimize,
        )

        assert self.commitment is not None
        load = self.profiles.load_kw[self.time]
        c, leader = self.config, self.commitment
        lower_s, upper_s = self._shift_bounds()
        capacity = self.capacity if c.enable_dr else np.zeros(self.n)
        floor = np.minimum(self.domain_floor, load + self.debt/self.dt)
        if not c.enable_dr:
            floor[:] = 0
        lower = np.column_stack([np.zeros(self.n), floor, np.zeros(self.n), lower_s]).ravel()
        upper = np.column_stack([load+capacity, load+capacity,
                                 np.minimum(capacity, load), upper_s]).ravel()
        scale = np.repeat(self.profiles.load_scale, 4)
        target = np.column_stack([
            (requests[:, 0]+1)/2 * (load+capacity),
            (requests[:, 1]+1)/2 * (load+capacity),
            (requests[:, 2]+1)/2 * np.minimum(capacity, load),
            requests[:, 3] * capacity,
        ]).ravel() / scale
        initial_s = np.minimum(np.clip(np.zeros(self.n), lower_s, upper_s), load-floor)
        initial = np.column_stack([np.zeros(self.n), load-initial_s,
                                   np.zeros(self.n), initial_s]).ravel() / scale
        balance = np.zeros((self.n, self.n*4))
        delta = np.zeros_like(balance)
        for i in range(self.n):
            balance[i, i*4:i*4+4] = 1
            delta[i, i*4+2:i*4+4] = 1
        resource = np.zeros((3, self.n*4))
        resource[0, 0::4] = 1
        resource[1, 1::4] = 1
        resource[2] = delta.sum(0)
        matrix = np.vstack([balance, delta, resource])
        lb = np.concatenate([load, -capacity, [0, max(leader.po, leader.pm), -np.inf]])
        ub = np.concatenate([load, capacity, [leader.der_offer, np.inf, leader.pm]])
        equality = lb == ub
        constraints = [
            LinearConstraint(matrix[equality] * scale, lb[equality], ub[equality]),
            LinearConstraint(matrix[~equality] * scale, lb[~equality], ub[~equality]),
        ]
        result = minimize(
            lambda y: .5 * np.square(y-target).sum(), initial,
            jac=lambda y: y-target, method='SLSQP', bounds=Bounds(lower/scale, upper/scale),
            constraints=constraints, options={'ftol': 1e-11, 'maxiter': 200},
        )
        solution = result.x * scale
        products = matrix @ solution
        violation = float(max(0, np.max(lb-products), np.max(products-ub),
                              np.max(lower-solution), np.max(solution-upper)))
        if (not result.success or not np.isfinite(solution).all()
                or violation > c.projection_tolerance):
            raise RuntimeError(f'清算失败 time={self.time}, residual={violation}: {result.message}')
        return solution.reshape(self.n, 4), violation

    def settle(self, actions: np.ndarray) -> EnvironmentStep:
        """清算 [N,4] 请求，返回逐个体利润/负成本和完整物理经济账本。"""
        if self.commitment is None or self.time >= self.spec.horizon:
            raise RuntimeError('settle 之前必须 commit_leader')
        requests = self._actions(actions, (self.n, 4))
        executed, residual = self._project(requests)
        pr, pu, cut, shift = executed.T
        # SLSQP 线性边界可能产生 1e-16 级负零；明确在残差门禁后消除此舍入。
        pr, pu, cut = np.maximum(pr, 0), np.maximum(pu, 0), np.maximum(cut, 0)
        delta = cut + shift
        if not self.config.enable_dr:
            delta = np.zeros(self.n)
        leader, c = self.commitment, self.config
        pg = float(self.profiles.der_kw[self.time])
        pa, ps = pg-leader.pd-float(pr.sum()), float(pu.sum())
        pb = leader.pe+ps-leader.po
        tariff = retail_tariff(int(self.profiles.local_slot[self.time]), self.cumulative)
        der_tariff = c.der_intercept + c.der_slope * pg
        consumer_grid = tariff * pu * self.dt
        consumer_der = der_tariff * pr * self.dt
        consumer_dr = (c.comfort_price * quadratic_ratio(delta, pu)
                       - c.dr_incentive * delta) * self.dt
        total_dr = float(delta.sum())
        uc_dr = float((c.dr_benefit * quadratic_ratio(np.array([total_dr]), np.array([leader.pm]))
                       - c.dr_incentive * total_dr)[0]) * self.dt
        components = np.array([consumer_grid.sum(), -c.purchase_price*pb*self.dt,
                               (der_tariff*pr.sum()-c.curtailment_price*pa)*self.dt, uc_dr])
        self.energy = ((1-c.decay_per_step)*self.energy + c.charge_efficiency*leader.pc*self.dt
                       - leader.po*self.dt/c.discharge_efficiency)
        self.debt += shift*self.dt
        self.cumulative += pu*self.dt
        if np.min(self.debt) < -c.projection_tolerance:
            raise RuntimeError('移峰债务为负')
        self.debt = np.maximum(self.debt, 0)
        if self.profiles.local_slot[self.time] == 95:
            if np.abs(self.debt).max() > c.projection_tolerance:
                raise RuntimeError('日末移峰未结清')
            self.debt[:] = 0  # 已通过真实残差检查，仅消除浮点舍入。
        total_residual = pb+pg+leader.po-pa-leader.pc-ps-float(pr.sum())
        info = {
            'flows': {'pb': pb, 'pg': pg, 'pa': pa, 'ps': ps, 'pc': leader.pc,
                      'pd': leader.pd, 'pe': leader.pe, 'po': leader.po, 'pm': leader.pm},
            'consumer_executed': np.column_stack([pr, pu, cut, shift]).tolist(),
            'consumer_requested': requests.tolist(), 'consumer_delta': delta.tolist(),
            'uc_components': components.tolist(),
            'consumer_components': np.column_stack([
                consumer_grid, consumer_der, consumer_dr
            ]).tolist(),
            'energy_kwh': self.energy, 'debt_kwh': self.debt.tolist(),
            'projection_residual': residual, 'system_residual': total_residual,
            'utc': str(self.profiles.utc[self.time]),
            'imputed': self.profiles.imputed[self.time].tolist(),
        }
        self.previous = np.column_stack([pr, pu, cut, shift])
        self.time += 1
        self.commitment = None
        return self._observation(np.r_[components.sum(), -(consumer_grid+consumer_der+consumer_dr)],
                                 info)

    def step(self, actions: np.ndarray) -> EnvironmentStep:
        """
        执行一次环境动作并返回统一下一步；仅一次 settle 推进物理时间，固定 UC
        wrapper 会准备下一承诺。
        """
        value = self._actions(actions, (self.n + 1, 4))
        self.commit_leader(value[0])
        return self.settle(value[1:])


class FixedLeaderDemandAdapter(EnvironmentAdapter):
    """固定已公开 UC 承诺，仅训练 N 个消费者；复用标准 OnPolicyTrainer。"""
    def __init__(self, environment: DemandResponseEnv, leader_action: tuple[float, ...] =
                 (0.0, 0.0, 1.0, 0.0)):
        """绑定经验证的数据和环境配置，声明静态尺寸；不执行动作或训练参数更新。"""
        self.environment = environment
        self.leader_action = environment._actions(np.asarray(leader_action), (4,))

    @property
    def spec(self) -> EnvironmentSpec:
        """返回静态 EnvironmentSpec；智能体数、观测/动作/state 尺寸和 horizon 由当前环境决定。"""
        return self.environment.follower_spec

    def _followers(self, step: EnvironmentStep) -> EnvironmentStep:
        """从联合 EnvironmentStep 提取消费者观测/个体奖励，保留公共 state、终止标记和审计 info。"""
        return self.validate_step(EnvironmentStep(
            step.observations[1:], step.state, step.rewards[1:], step.terminated, step.truncated,
            info=step.info,
        ))

    def reset(self, seed: int | None = None) -> EnvironmentStep:
        """按 seed 选择完整日或重置固定事件，清空债务/累计购电和上次动作；返回统一初始观测。"""
        self.environment.reset(seed)
        return self._followers(self.environment.commit_leader(self.leader_action))

    def step(self, actions: np.ndarray) -> EnvironmentStep:
        """
        执行一次环境动作并返回统一下一步；仅一次 settle 推进物理时间，固定 UC
        wrapper 会准备下一承诺。
        """
        step = self.environment.settle(actions)
        if step.done:
            return self._followers(step)
        committed = self.environment.commit_leader(self.leader_action)
        committed.rewards, committed.info = step.rewards, step.info
        return self._followers(committed)


class DailyDemandAdapter(EnvironmentAdapter):
    """按 reset seed 在无插补的完整训练日中抽样，复用相同静态 spec。"""
    def __init__(self, profiles: DemandProfiles, config: DemandResponseConfig | None = None,
                 fixed_leader: tuple[float, ...] | None = None):
        """绑定经验证的数据和环境配置，声明静态尺寸；不执行动作或训练参数更新。"""
        self.profiles = profiles
        self.config = config or DemandResponseConfig()
        self.fixed_leader = fixed_leader
        self.starts = [i for i in range(0, len(profiles.load_kw)-95, 96)
                       if not profiles.imputed[i:i+96].any()]
        if not self.starts:
            raise ValueError('没有无插补的完整日')
        self.adapter = self._build(self.starts[0])

    def _build(self, start: int) -> EnvironmentAdapter:
        """从已验证的 start 位置构造完整 96 步环境，可选包装固定 UC；不拟合新的数据尺度。"""
        environment = DemandResponseEnv(self.profiles.window(start, 96), self.config)
        return (FixedLeaderDemandAdapter(environment, self.fixed_leader)
                if self.fixed_leader is not None else environment)

    @property
    def spec(self) -> EnvironmentSpec:
        """返回静态 EnvironmentSpec；智能体数、观测/动作/state 尺寸和 horizon 由当前环境决定。"""
        return self.adapter.spec

    def reset(self, seed: int | None = None) -> EnvironmentStep:
        """按 seed 选择完整日或重置固定事件，清空债务/累计购电和上次动作；返回统一初始观测。"""
        start = int(np.random.default_rng(seed).choice(self.starts))
        self.adapter = self._build(start)
        result = self.adapter.reset(seed)
        result.info['day_start'] = start
        return self.validate_step(result)

    def step(self, actions: np.ndarray) -> EnvironmentStep:
        """
        执行一次环境动作并返回统一下一步；仅一次 settle 推进物理时间，固定 UC
        wrapper 会准备下一承诺。
        """
        return self.validate_step(self.adapter.step(actions))

    @property
    def sequential_spec(self) -> SequentialSpec:
        """返回联合、UC、随机计划和条件消费者规格；消费者输入追加广播计划维度。"""
        if not isinstance(self.adapter, DemandResponseEnv):
            raise ValueError('固定 leader adapter 不可同时训练 leader')
        return self.adapter.sequential_spec

    @property
    def checkpoint_context(self) -> dict[str, object]:
        """返回数据 SHA、时间窗口、训练尺度及物理配置，用于拒绝不同数据/环境的 checkpoint 恢复。"""
        return {'source_sha256': self.profiles.source_sha256, 'config': asdict(self.config),
                'load_scale': self.profiles.load_scale.tolist(),
                'der_scale': self.profiles.der_scale, 'utc_start': str(self.profiles.utc[0]),
                'utc_end': str(self.profiles.utc[-1]), 'eligible_day_starts': self.starts}

    def commit_leader(self, action: np.ndarray) -> EnvironmentStep:
        """提交 [4] UC 请求，返回含公开承诺的响应阶段观测；不推进物理时间。"""
        if not isinstance(self.adapter, DemandResponseEnv):
            raise ValueError('固定 leader adapter 不接受外部承诺')
        return self.adapter.commit_leader(action)

    def settle(self, actions: np.ndarray) -> EnvironmentStep:
        """在已承诺 UC 动作下清算消费者 [N,4] 请求，推进一次物理步，返回逐个体奖励及账本。"""
        if not isinstance(self.adapter, DemandResponseEnv):
            raise ValueError('固定 leader adapter 不接受外部阶段清算')
        return self.adapter.settle(actions)
