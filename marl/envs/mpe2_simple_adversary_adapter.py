"""MPE2 simple_adversary 与仓库固定形状协议之间的适配器。

MPE2 是 Farama 的 Gymnasium 风格多智能体环境集合。该场景同时提供离散和连续
动作，并保留对手与善意智能体各自的奖励，因此适合检查一般和博弈算法的数据链。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from marl.envs.base import (
    ActionKind,
    EnvironmentAdapter,
    EnvironmentSpec,
    EnvironmentStep,
    RewardStructure,
)


class ParallelMPEEnvironment(Protocol):
    """只描述适配器实际使用的 PettingZoo Parallel API 能力。"""

    possible_agents: list[str]
    agents: list[str]
    state_space: spaces.Space[Any]

    def observation_space(self, agent: str) -> spaces.Space[Any]: ...

    def action_space(self, agent: str) -> spaces.Space[Any]: ...

    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]: ...

    def step(
        self, actions: dict[str, Any]
    ) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]: ...

    def state(self) -> NDArray[np.floating[Any]]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class MPE2SimpleAdversaryConfig:
    """只包含场景动力学；训练超参数仍属于算法配置。"""

    num_good_agents: int = 2
    horizon: int = 25
    dynamic_rescaling: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_good_agents, bool)
            or not isinstance(self.num_good_agents, int)
            or self.num_good_agents < 1
        ):
            raise ValueError("num_good_agents 必须大于 0")
        if (
            isinstance(self.horizon, bool)
            or not isinstance(self.horizon, int)
            or self.horizon < 1
        ):
            raise ValueError("horizon 必须大于 0")
        if not isinstance(self.dynamic_rescaling, bool):
            raise TypeError("dynamic_rescaling 必须是 bool")


class MPE2SimpleAdversaryAdapter(EnvironmentAdapter):
    """把 MPE2 字典数据整理成固定 [N,O] / [S] 张量语义。

    simple_adversary 的 adversary 观测比 good agent 短。仓库当前网络要求同构
    输入，因此只在环境边界对右侧补零，并在 info 中保留原始长度。离散动作全部
    合法；连续策略输出 [-1,1]，这里线性映射到 MPE2 的 [0,1] Box。
    """

    def __init__(
        self,
        environment: ParallelMPEEnvironment,
        action_kind: ActionKind,
        *,
        horizon: int,
    ) -> None:
        self.environment = environment
        self.action_kind = ActionKind(action_kind)
        self.agent_ids = tuple(environment.possible_agents)
        if not self.agent_ids:
            raise ValueError("MPE2 环境必须至少包含一个智能体")
        self.observation_dims = tuple(
            self._flat_dim(environment.observation_space(agent))
            for agent in self.agent_ids
        )
        self._spec = EnvironmentSpec(
            num_agents=len(self.agent_ids),
            observation_dim=max(self.observation_dims),
            action_dim=self._validate_action_spaces(),
            state_dim=self._flat_dim(environment.state_space),
            action_kind=self.action_kind,
            horizon=horizon,
            reward_structure=RewardStructure.INDIVIDUAL,
        )
        self._done = True

    @staticmethod
    def _flat_dim(space: spaces.Space[Any]) -> int:
        shape = getattr(space, "shape", None)
        if shape is None or len(shape) != 1 or shape[0] is None:
            raise ValueError("simple_adversary 的观测与 state 必须是一维定长空间")
        return int(shape[0])

    def _validate_action_spaces(self) -> int:
        action_spaces = tuple(
            self.environment.action_space(agent) for agent in self.agent_ids
        )
        if self.action_kind == ActionKind.DISCRETE:
            if not all(isinstance(space, spaces.Discrete) for space in action_spaces):
                raise ValueError("离散适配器要求每个智能体使用 Discrete 动作空间")
            dimensions = {
                int(space.n)
                for space in action_spaces
                if isinstance(space, spaces.Discrete)
            }
        else:
            if not all(isinstance(space, spaces.Box) for space in action_spaces):
                raise ValueError("连续适配器要求每个智能体使用 Box 动作空间")
            boxes = tuple(
                space for space in action_spaces if isinstance(space, spaces.Box)
            )
            dimensions = {self._flat_dim(space) for space in boxes}
            if any(
                not np.allclose(space.low, 0.0)
                or not np.allclose(space.high, 1.0)
                for space in boxes
            ):
                raise ValueError("连续 simple_adversary 动作空间必须为 [0,1]")
        if len(dimensions) != 1:
            raise ValueError("仓库当前要求所有智能体具有相同动作维度")
        return dimensions.pop()

    @property
    def spec(self) -> EnvironmentSpec:
        return self._spec

    def _observations(self, values: dict[str, Any]) -> NDArray[np.float32]:
        result = np.zeros(
            (self.spec.num_agents, self.spec.observation_dim), dtype=np.float32
        )
        for index, (agent, original_dim) in enumerate(
            zip(self.agent_ids, self.observation_dims, strict=True)
        ):
            value = np.asarray(values[agent], dtype=np.float32).reshape(-1)
            if value.shape != (original_dim,):
                raise ValueError(f"{agent} 观测维度发生变化：{value.shape}")
            result[index, :original_dim] = value
        return result

    def _snapshot(
        self,
        observations: dict[str, Any],
        rewards: dict[str, float],
        terminations: dict[str, bool],
        truncations: dict[str, bool],
        infos: dict[str, dict[str, Any]],
    ) -> EnvironmentStep:
        terminated_values = tuple(
            bool(terminations[agent]) for agent in self.agent_ids
        )
        truncated_values = tuple(
            bool(truncations[agent]) for agent in self.agent_ids
        )
        if len(set(terminated_values)) != 1 or len(set(truncated_values)) != 1:
            raise RuntimeError("固定智能体训练链暂不支持部分智能体提前结束")
        action_mask = None
        if self.action_kind == ActionKind.DISCRETE:
            action_mask = np.ones(
                (self.spec.num_agents, self.spec.action_dim), dtype=bool
            )
        step = EnvironmentStep(
            observations=self._observations(observations),
            state=np.asarray(self.environment.state(), dtype=np.float32).reshape(-1),
            rewards=np.asarray(
                [rewards[agent] for agent in self.agent_ids], dtype=np.float32
            ),
            terminated=terminated_values[0],
            truncated=truncated_values[0],
            action_mask=action_mask,
            info={
                "agent_ids": self.agent_ids,
                "observation_dims": self.observation_dims,
                "per_agent": {
                    agent: dict(infos[agent]) for agent in self.agent_ids
                },
            },
        )
        self._done = step.done
        return self.validate_step(step)

    def reset(self, seed: int | None = None) -> EnvironmentStep:
        observations, infos = self.environment.reset(seed=seed)
        zeros = {agent: 0.0 for agent in self.agent_ids}
        false = {agent: False for agent in self.agent_ids}
        self._done = False
        return self._snapshot(observations, zeros, false, false, infos)

    def step(self, actions: np.ndarray) -> EnvironmentStep:
        if self._done:
            raise RuntimeError("episode 已结束，请先调用 reset")
        values = np.asarray(actions)
        if self.action_kind == ActionKind.DISCRETE:
            if values.shape != (self.spec.num_agents,) or not np.issubdtype(
                values.dtype, np.integer
            ):
                raise ValueError("离散动作应为整数数组 [num_agents]")
            if (values < 0).any() or (values >= self.spec.action_dim).any():
                raise ValueError("离散动作索引超出范围")
            environment_actions: dict[str, Any] = {
                agent: int(values[index])
                for index, agent in enumerate(self.agent_ids)
            }
        else:
            values = np.asarray(actions, dtype=np.float32)
            expected = (self.spec.num_agents, self.spec.action_dim)
            if values.shape != expected or not np.isfinite(values).all():
                raise ValueError(f"连续动作应为无 NaN 的 {expected} 数组")
            scaled = (np.clip(values, -1.0, 1.0) + 1.0) / 2.0
            environment_actions = {
                agent: scaled[index]
                for index, agent in enumerate(self.agent_ids)
            }
        result = self.environment.step(environment_actions)
        return self._snapshot(*result)

    def close(self) -> None:
        self.environment.close()


def build_mpe2_simple_adversary(
    config: MPE2SimpleAdversaryConfig | None = None,
    *,
    action_kind: ActionKind = ActionKind.DISCRETE,
) -> MPE2SimpleAdversaryAdapter:
    """按需导入可选的 MPE2 依赖并构造官方 simple_adversary_v3。"""

    config = config or MPE2SimpleAdversaryConfig()
    try:
        from mpe2 import simple_adversary_v3  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - 取决于用户安装的 extras。
        raise ImportError(
            "使用 MPE2 基准前请安装可选依赖：pip install -e '.[benchmark]'"
        ) from error
    environment = simple_adversary_v3.parallel_env(
        N=config.num_good_agents,
        max_cycles=config.horizon,
        continuous_actions=ActionKind(action_kind) == ActionKind.CONTINUOUS,
        dynamic_rescaling=config.dynamic_rescaling,
    )
    return MPE2SimpleAdversaryAdapter(
        environment, action_kind, horizon=config.horizon
    )
