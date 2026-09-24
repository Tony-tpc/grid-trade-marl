"""把电能交易环境转换为所有算法共用的观测、动作和 batch 协议。"""

from __future__ import annotations

from itertools import product

import numpy as np
from numpy.typing import NDArray

from marl.envs.base import (
    ActionKind,
    EnvironmentAdapter,
    EnvironmentSpec,
    EnvironmentStep,
    RewardStructure,
)
from marl.envs.energy_trading import EnergyTradingEnv


class EnergyTradingAdapter(EnvironmentAdapter):
    """论文环境专属适配器；算法只接触本类返回的 EnvironmentStep。

    原论文同时使用 3 个连续动作和 1 个二元动作。目前本仓库的算法按纯离散/纯连续
    动作设计，因此提供两种显式编码：

    * discrete：把 EV/储能 {-1,0,1}、HVAC {0,0.5,1}、SA {0,1} 的笛卡尔积
      编成 54 个离散选项，适合 MAAC/MAPPO/QMIX；这是动作量化近似。
    * continuous：使用长度为 4 的连续动作，前 3 维映射到物理功率，最后一维用
      阈值决定 SA 是否启动，适合 MADDPG/MASAC；阈值是离散化近似。

    两种编码均不声称精确复现论文中的混合 Bernoulli/Gaussian 策略。
    """

    def __init__(self, environment: EnergyTradingEnv, action_kind: ActionKind) -> None:
        self.environment = environment
        self.action_kind = ActionKind(action_kind)
        # 列顺序固定：EV、储能、HVAC、智能电器。
        self.action_catalogue: NDArray[np.float64] = np.asarray(
            list(product((-1.0, 0.0, 1.0), (-1.0, 0.0, 1.0), (0.0, 0.5, 1.0), (0.0, 1.0))),
            dtype=np.float64,
        )
        initial_observation = environment.observe()
        n, observation_dim = initial_observation.shape
        self._spec = EnvironmentSpec(
            num_agents=n,
            observation_dim=observation_dim,
            action_dim=(
                len(self.action_catalogue) if self.action_kind == ActionKind.DISCRETE else 4
            ),
            state_dim=n * observation_dim,
            action_kind=self.action_kind,
            horizon=environment.config.horizon,
            reward_structure=RewardStructure.INDIVIDUAL,
        )

    @property
    def spec(self) -> EnvironmentSpec:
        return self._spec

    def _action_mask(self) -> NDArray[np.bool_] | None:
        if self.action_kind == ActionKind.CONTINUOUS:
            return None
        n, catalogue = self.spec.num_agents, self.action_catalogue
        mask = np.ones((n, len(catalogue)), dtype=bool)
        ev_available = self.environment.ev_available()
        sa_available = self.environment.sa_can_start()
        ownership = self.environment.ownership
        for i in range(n):
            if not ev_available[i]:
                mask[i] &= catalogue[:, 0] == 0.0
            if not ownership("storage")[i]:
                mask[i] &= catalogue[:, 1] == 0.0
            if not ownership("hvac")[i]:
                mask[i] &= catalogue[:, 2] == 0.0
            if not sa_available[i]:
                mask[i] &= catalogue[:, 3] == 0.0
            elif self.environment.time_step == self.environment.config.sa_latest_start_step:
                # 当天最后可启动时段必须开始，否则无法完成不可中断的工作周期。
                mask[i] &= catalogue[:, 3] == 1.0
        if not mask.any(axis=-1).all():
            raise RuntimeError("动作目录中没有合法动作，请检查适配器的离散网格")
        return mask

    def _snapshot(
        self,
        rewards: NDArray[np.float32],
        terminated: bool = False,
        info: dict[str, float] | None = None,
    ) -> EnvironmentStep:
        observations = self.environment.observe()
        # 论文第 IV-A 节将 global state 直接设为所有局部观测的联合向量。
        state = observations.reshape(-1).copy()
        return self.validate_step(
            EnvironmentStep(
                observations=observations,
                state=state,
                rewards=rewards,
                terminated=terminated,
                truncated=False,
                action_mask=self._action_mask(),
                info=info or {},
            )
        )

    def reset(self, seed: int | None = None) -> EnvironmentStep:
        self.environment.reset(seed)
        return self._snapshot(np.zeros(self.spec.num_agents, dtype=np.float32))

    def step(self, actions: np.ndarray) -> EnvironmentStep:
        if self.action_kind == ActionKind.DISCRETE:
            indices = np.asarray(actions)
            if indices.shape != (self.spec.num_agents,) or not np.issubdtype(
                indices.dtype, np.integer
            ):
                raise ValueError("离散动作应为整数数组 [num_agents]")
            if (indices < 0).any() or (indices >= self.spec.action_dim).any():
                raise ValueError("离散动作索引超出动作目录")
            mask = self._action_mask()
            assert mask is not None
            if not mask[np.arange(self.spec.num_agents), indices].all():
                raise ValueError("动作包含当前状态下不可用的 EV/SA/设备控制")
            physical_actions = self.action_catalogue[indices]
        else:
            normalized = np.asarray(actions, dtype=np.float64)
            if normalized.shape != (self.spec.num_agents, 4) or not np.isfinite(normalized).all():
                raise ValueError("连续动作应为无 NaN 的 [num_agents,4] 数组")
            physical_actions = np.clip(normalized, -1.0, 1.0).copy()
            physical_actions[:, 2] = (physical_actions[:, 2] + 1.0) / 2.0
            physical_actions[:, 3] = (physical_actions[:, 3] >= 0.0).astype(float)
        result = self.environment.step(physical_actions)
        return self._snapshot(result.rewards, result.terminated, result.info)
