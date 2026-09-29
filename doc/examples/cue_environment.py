"""可见二值提示教学环境；用于演示 adapter，不用于记忆能力验收。"""

from __future__ import annotations

import numpy as np

from marl.envs import ActionKind, EnvironmentAdapter, EnvironmentSpec, EnvironmentStep


class CueEnvironment(EnvironmentAdapter):
    """两个玩家分别按可见提示作答，正确 +1、错误 -1，时间上限截断。"""

    def __init__(self, horizon: int = 8) -> None:
        """输入正整数 horizon；输出 None，初始化固定尺寸规格和私有 RNG。"""
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon 必须为正整数")
        self._spec = EnvironmentSpec(2, 2, 2, 4, ActionKind.DISCRETE, horizon)
        self._rng = np.random.default_rng(0)
        self._cue = np.zeros(2, dtype=np.int64)
        self._time = horizon  # 第一次执行动作前也必须 reset。

    @property
    def spec(self) -> EnvironmentSpec:
        """无输入；返回 N=2、O=2、A=2、S=4 的个体奖励规格。"""
        return self._spec

    def _snapshot(self, rewards: np.ndarray) -> EnvironmentStep:
        """输入本步 float32 奖励 [2]；输出独立数组组成且已校验的环境结果。"""
        observations = np.column_stack(
            (self._cue, np.full(2, self._time / self.spec.horizon))
        ).astype(np.float32)
        return self.validate_step(
            EnvironmentStep(
                observations=observations,
                state=observations.flatten().copy(),
                rewards=rewards.copy(),
                terminated=False,
                truncated=self._time >= self.spec.horizon,
                action_mask=np.ones((2, 2), dtype=np.bool_),
            )
        )

    def reset(self, seed: int | None = None) -> EnvironmentStep:
        """输入可选种子；输出初始观测和零奖励，重置时间与提示。"""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._time = 0
        self._cue = self._rng.integers(0, 2, size=2, dtype=np.int64)
        return self._snapshot(np.zeros(2, dtype=np.float32))

    def step(self, actions: np.ndarray) -> EnvironmentStep:
        """输入 int64 动作 [2]；按当前提示奖励后推进，输出下一观测与截断标记。"""
        if self._time >= self.spec.horizon:
            raise RuntimeError("episode 已结束，请先 reset")
        if actions.shape != (2,) or actions.dtype != np.int64:
            raise ValueError("actions 必须为 int64 [2]")
        if not np.isin(actions, (0, 1)).all():
            raise ValueError("动作必须为 0 或 1")
        rewards = np.where(actions == self._cue, 1.0, -1.0).astype(np.float32)
        self._time += 1
        self._cue = self._rng.integers(0, 2, size=2, dtype=np.int64)
        return self._snapshot(rewards)
