"""通用 replay 与离策略 checkpoint 训练链；不包含具体算法更新顺序。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from torch import nn

from marl.core import MARLBatch
from marl.envs.base import EnvironmentSpec, Transition
from marl.objectives import LossBundle
from marl.runtime import TensorReplayBuffer
from marl.training.optimization import OptimizerRuntime


class OffPolicyAlgorithm(Protocol):
    """trainer 所需能力；更新顺序由具体算法的 ``update`` 定义。"""

    spec: EnvironmentSpec

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...

    def compute_loss_bundle(self, batch: MARLBatch) -> LossBundle: ...

    def update(self, batch: MARLBatch, runtime: OptimizerRuntime) -> dict[str, float]: ...

    def target_pairs(self) -> Iterable[tuple[nn.Module, nn.Module]]: ...

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

    def update(self) -> dict[str, float]:
        if not self.ready:
            raise RuntimeError(
                f"replay 样本不足：{len(self.replay)}/{self.replay_config.batch_size}"
            )
        batch = self.replay.sample(
            self.replay_config.batch_size, self.rng, device=self.device
        )
        return self.algorithm.update(batch, self.optimization)

    def update_batch(self, batch: MARLBatch) -> dict[str, float]:
        return self.algorithm.update(batch.to(self.device), self.optimization)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "algorithm": deepcopy(dict(self.algorithm.state_dict())),
            "config": deepcopy(self.config_data),
            "torch_rng": torch.get_rng_state().clone(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "optimization": self.optimization.state_dict(),
            "replay": self.replay.state_dict(),
            "rng": deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != 2:
            raise ValueError(
                "不兼容的 trainer checkpoint：旧 schema 缺少分离 optimizer 状态"
            )
        if state.get("config") != self.config_data:
            raise ValueError("checkpoint config 与当前 trainer config 不一致")
        torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng"]])
        self.algorithm.load_state_dict(cast(Mapping[str, Any], state["algorithm"]))
        self.optimization.load_state_dict(cast(Mapping[str, Any], state["optimization"]))
        self.replay.load_state_dict(cast(dict[str, object], state["replay"]))
        self.rng.bit_generator.state = cast(dict[str, Any], deepcopy(state["rng"]))

    def save_checkpoint(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), target)

    def load_checkpoint(
        self, path: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> None:
        state = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(state, Mapping):
            raise TypeError("trainer checkpoint 顶层必须是 mapping")
        self.load_state_dict(cast(Mapping[str, Any], state))
