"""通用 replay 与离策略 checkpoint 训练链；不包含具体算法更新顺序。"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from torch import Tensor

from marl.core import MARLBatch
from marl.envs.base import EnvironmentSpec, Transition
from marl.runtime import (
    SyncVectorEnv,
    TensorReplayBuffer,
    environment_steps_to_tensors,
)
from marl.training.checkpoint import (
    build_trainer_checkpoint_state,
    load_checkpoint_state,
    restore_torch_rng_state,
    save_checkpoint_state,
    validate_trainer_checkpoint_state,
)
from marl.training.optimization import OptimizerRuntime


class OffPolicyActor(Protocol):
    """批量 collector 所需的最小动作能力。"""

    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
    ) -> Tensor: ...


class OffPolicyAlgorithm(Protocol):
    """trainer 实际调用的最小能力，不是供算法继承的第二层基类。

    collector 单独依赖 OffPolicyActor；本协议不要求 act/parameters/target_pairs，
    因为 trainer 不负责动作选择、公式或 optimizer 分配。
    """

    def update(self, batch: MARLBatch, runtime: OptimizerRuntime) -> dict[str, float]: ...

    def state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]: ...

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> Any: ...


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    capacity: int = 100_000
    batch_size: int = 256

    def __post_init__(self) -> None:
        if self.capacity < 1 or self.batch_size < 1:
            raise ValueError("capacity 和 batch_size 必须大于 0")
        if self.batch_size > self.capacity:
            raise ValueError("batch_size 不能大于 capacity")


@dataclass(frozen=True, slots=True)
class OffPolicyUpdateConfig:
    """单 optimizer 的静态超参数；不包含更新循环或网络实例。"""

    learning_rate: float = 3e-4
    max_grad_norm: float | None = 10.0
    amp_dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必须大于 0")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm 必须大于 0 或为 None")
        if self.amp_dtype not in (None, torch.bfloat16):
            raise ValueError("amp_dtype 目前只支持 torch.bfloat16")


@dataclass(frozen=True, slots=True)
class ActorCriticUpdateConfig(OffPolicyUpdateConfig):
    """actor-critic 公共 optimizer 字段；专用值为空时回退兼容默认值。"""

    actor_learning_rate: float | None = None
    critic_learning_rate: float | None = None
    actor_max_grad_norm: float | None = None
    critic_max_grad_norm: float | None = None

    def __post_init__(self) -> None:
        OffPolicyUpdateConfig.__post_init__(self)
        for name, value in (
            ("actor_learning_rate", self.actor_learning_rate),
            ("critic_learning_rate", self.critic_learning_rate),
            ("actor_max_grad_norm", self.actor_max_grad_norm),
            ("critic_max_grad_norm", self.critic_max_grad_norm),
        ):
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} 必须大于 0 或为 None")

    @property
    def resolved_actor_learning_rate(self) -> float:
        return self.actor_learning_rate or self.learning_rate

    @property
    def resolved_critic_learning_rate(self) -> float:
        return self.critic_learning_rate or self.learning_rate

    @property
    def resolved_actor_max_grad_norm(self) -> float | None:
        return (
            self.actor_max_grad_norm
            if self.actor_max_grad_norm is not None
            else self.max_grad_norm
        )

    @property
    def resolved_critic_max_grad_norm(self) -> float | None:
        return (
            self.critic_max_grad_norm
            if self.critic_max_grad_norm is not None
            else self.max_grad_norm
        )


@dataclass(frozen=True, slots=True)
class TemperatureActorCriticUpdateConfig(ActorCriticUpdateConfig):
    temperature_learning_rate: float | None = None
    temperature_max_grad_norm: float | None = None

    def __post_init__(self) -> None:
        ActorCriticUpdateConfig.__post_init__(self)
        for name, value in (
            ("temperature_learning_rate", self.temperature_learning_rate),
            ("temperature_max_grad_norm", self.temperature_max_grad_norm),
        ):
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} 必须大于 0 或为 None")

    @property
    def resolved_temperature_learning_rate(self) -> float:
        return self.temperature_learning_rate or self.learning_rate

    @property
    def resolved_temperature_max_grad_norm(self) -> float | None:
        return (
            self.temperature_max_grad_norm
            if self.temperature_max_grad_norm is not None
            else self.max_grad_norm
        )


class OffPolicyCollector:
    """同步向量环境 collector：每个环境时刻只执行一次批量 actor 前向。"""

    def __init__(
        self,
        environment: SyncVectorEnv,
        algorithm: OffPolicyActor,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.environment = environment
        self.algorithm = algorithm
        self.device = torch.device(device)

    def rollout(
        self,
        seeds: Sequence[int],
        *,
        deterministic: bool = False,
        action_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> Iterator[tuple[Transition, ...]]:
        steps = self.environment.reset(seeds)
        while not all(step.done for step in steps):
            if any(step.done for step in steps):
                raise RuntimeError("离策略向量环境不支持部分 episode 提前结束")
            observations, _, action_masks = environment_steps_to_tensors(
                steps,
                self.environment.spec,
                device=self.device,
            )
            with torch.inference_mode():
                actions = self.algorithm.act(
                    observations,
                    deterministic=deterministic,
                    action_mask=action_masks,
                ).cpu().numpy()
            if action_transform is not None:
                actions = action_transform(actions)
            following = self.environment.step(actions)
            yield tuple(
                Transition(current, action, next_step)
                for current, action, next_step in zip(
                    steps, actions, following, strict=True
                )
            )
            steps = following


class OffPolicyTrainer:
    """拥有 replay 和 optimizer 状态的通用离策略 trainer。"""

    def __init__(
        self,
        spec: EnvironmentSpec,
        algorithm: OffPolicyAlgorithm,
        replay: TensorReplayBuffer,
        replay_config: ReplayConfig,
        optimization: OptimizerRuntime,
        *,
        device: torch.device | str = "cpu",
        config_data: Mapping[str, object] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.spec = spec
        self.algorithm = algorithm
        self.replay = replay
        self.replay_config = replay_config
        self.optimization = optimization
        self.device = torch.device(device)
        self.config_data = deepcopy(dict(config_data)) if config_data is not None else None
        self.rng = rng or np.random.default_rng()

    @property
    def ready(self) -> bool:
        return len(self.replay) >= self.replay_config.batch_size

    def record(self, transition: Transition) -> None:
        self.replay.add(transition)

    def record_many(self, transitions: Sequence[Transition]) -> None:
        self.replay.record_many(transitions)

    def update(self) -> dict[str, float]:
        """从 replay 抽取一批旧经验；只委派算法 update，不重复实现优化循环。"""

        if not self.ready:
            raise RuntimeError(
                f"replay 样本不足：{len(self.replay)}/{self.replay_config.batch_size}"
            )
        batch = self.replay.sample(
            self.replay_config.batch_size, self.rng, device=self.device
        )
        return self.algorithm.update(batch, self.optimization)

    def update_batch(self, batch: MARLBatch) -> dict[str, float]:
        """外部经验来源的入口：跳过 replay 抽样，但复用完全相同的算法 update。"""

        return self.algorithm.update(batch.to(self.device), self.optimization)

    def state_dict(self) -> dict[str, Any]:
        state = build_trainer_checkpoint_state(
            self.algorithm.state_dict(),
            self.config_data,
            self.optimization.state_dict(),
        )
        state["replay"] = self.replay.state_dict()
        state["rng"] = deepcopy(self.rng.bit_generator.state)
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        validate_trainer_checkpoint_state(state, self.config_data)
        restore_torch_rng_state(state)
        self.algorithm.load_state_dict(cast(Mapping[str, Any], state["algorithm"]))
        self.optimization.load_state_dict(cast(Mapping[str, Any], state["optimization"]))
        self.replay.load_state_dict(cast(dict[str, object], state["replay"]))
        self.rng.bit_generator.state = cast(dict[str, Any], deepcopy(state["rng"]))

    def save_checkpoint(self, path: str | Path) -> None:
        save_checkpoint_state(self.state_dict(), path)

    def load_checkpoint(
        self, path: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> None:
        self.load_state_dict(load_checkpoint_state(path, map_location=map_location))
