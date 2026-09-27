"""延迟提示记忆任务：观测别名使无状态策略不能利用当前帧恢复提示。"""
from __future__ import annotations

import numpy as np

from marl.envs.base import ActionKind, EnvironmentAdapter, EnvironmentSpec, EnvironmentStep


class MemoryCueAdapter(EnvironmentAdapter):
    """两个独立二值提示，最后一步按个体答案给 +/-1，中间动作被忽略。

    观测 [N,2] 是 (可见提示, 查询标记)；state 仅拼接当前观测，不泄漏历史。
    reset(seed=0..3) 平衡枚举四种联合提示；其他 seed 同样通过低两位确定提示。
    这是记忆链验收任务，不把它冒充非合作博弈的收敛证明。
    """

    def __init__(self, horizon: int = 10) -> None:
        if horizon < 2:
            raise ValueError("memory horizon 至少为 2")
        self._spec = EnvironmentSpec(2, 2, 2, 4, ActionKind.DISCRETE, horizon)
        self._time = horizon
        self._cue = np.zeros(2, dtype=np.int64)

    @property
    def spec(self) -> EnvironmentSpec:
        return self._spec

    def _step_result(self, rewards: np.ndarray) -> EnvironmentStep:
        observations = np.zeros((2, 2), dtype=np.float32)
        if self._time == 0:
            observations[:, 0] = self._cue * 2 - 1
        observations[:, 1] = float(self._time == self.spec.horizon - 1)
        return self.validate_step(EnvironmentStep(
            observations, observations.flatten().copy(), rewards,
            terminated=self._time >= self.spec.horizon, truncated=False,
            action_mask=np.ones((2, 2), dtype=np.bool_),
        ))

    def reset(self, seed: int | None = None) -> EnvironmentStep:
        code = int(np.random.default_rng().integers(4)) if seed is None else seed % 4
        self._cue = np.array([code & 1, (code >> 1) & 1], dtype=np.int64)
        self._time = 0
        return self._step_result(np.zeros(2, dtype=np.float32))

    def step(self, actions: np.ndarray) -> EnvironmentStep:
        if self._time >= self.spec.horizon:
            raise RuntimeError("episode 已结束，需 reset")
        if actions.shape != (2,) or not np.isin(actions, (0, 1)).all():
            raise ValueError("actions 必须是两个二值动作")
        reward = (np.where(actions == self._cue, 1.0, -1.0).astype(np.float32)
                  if self._time == self.spec.horizon - 1 else np.zeros(2, dtype=np.float32))
        self._time += 1
        return self._step_result(reward)
