"""通用 replay、单步更新、target update 与 checkpoint 训练链。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import torch
from torch import nn

from marl.core import MARLBatch
from marl.envs.base import EnvironmentSpec, Transition
from marl.objectives import LossBundle
from marl.runtime import TensorReplayBuffer

if TYPE_CHECKING:
    from marl.recipes import AlgorithmRecipe
    from marl.registry import ComponentRegistry
    from marl.target_updates import TargetUpdate


class OffPolicyAlgorithm(Protocol):
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...

    def compute_loss_bundle(self, batch: MARLBatch) -> LossBundle: ...

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
class OffPolicyOptimizerConfig:
    learning_rate: float = 3e-4
    update: OffPolicyUpdateConfig = OffPolicyUpdateConfig()

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate 必须大于 0")


class OffPolicyUpdatePlan:
    """执行一次 loss/backward/clip/step/target-update。"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OffPolicyUpdateConfig,
        target_update: TargetUpdate,
    ) -> None:
        self.optimizer = optimizer
        self.config = config
        self.target_update = target_update
        self.update_count = 0

    def update(self, algorithm: OffPolicyAlgorithm, batch: MARLBatch) -> dict[str, float]:
        """执行一次标准 PyTorch 训练步骤。
        用来进行整体梯度更新（包括 Actor 和 Critic 的参数更新）。
        顺序为清除旧梯度 -> 计算 loss -> 反向传播 -> 裁剪梯度 -> 更新参数。
        """
        device = batch.observations.device
        if self.config.amp_dtype is not None and (
            device.type != "cuda" or not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("BF16 AMP 需要支持 BF16 的 CUDA 设备")

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=self.config.amp_dtype is not None,
        ):
            bundle = algorithm.compute_loss_bundle(batch)
        # backward() 沿计算图应用链式法则，把梯度写入每个 Parameter 的 .grad（求导）。
        bundle.total.backward()
        # target network 等冻结参数不属于该 optimizer，也不应进入梯度范数统计。
        parameters = tuple(
            parameter for parameter in algorithm.parameters() if parameter.requires_grad
        )
        if self.config.max_grad_norm is None:
            squared = [
                parameter.grad.detach().pow(2).sum()
                for parameter in parameters
                if parameter.grad is not None
            ]
            gradient_norm = (
                torch.stack(squared).sum().sqrt()
                if squared
                else torch.zeros((), device=device)
            )
        else:
            gradient_norm = nn.utils.clip_grad_norm_(
                parameters, self.config.max_grad_norm
            )
        # optimizer.step() 才真正修改神经网络参数。
        self.optimizer.step()
        self.update_count += 1
        should_update = getattr(self.target_update, "should_step", None)
        if should_update is None or should_update(self.update_count):
            # 更新目标网络，确保训练稳定性（DQN）。
            self.target_update.step(algorithm.target_pairs())
        metrics = dict(bundle.terms)
        metrics["gradient_norm"] = gradient_norm.detach()
        names = tuple(metrics)
        values = torch.stack([metrics[name].reshape(()) for name in names]).cpu().tolist()
        # 输出每个 loss 的数值，便于日志记录或 TensorBoard 可视化。
        return dict(zip(names, values, strict=True))

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer": deepcopy(self.optimizer.state_dict()),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.optimizer.load_state_dict(cast(dict[str, Any], state["optimizer"]))
        self.update_count = int(state["update_count"])


class OffPolicyTrainer:
    """拥有 replay 和更新计划的通用离策略 trainer。"""

    def __init__(
        self,
        spec: EnvironmentSpec,
        algorithm: OffPolicyAlgorithm,
        replay: TensorReplayBuffer,
        replay_config: ReplayConfig,
        update_plan: OffPolicyUpdatePlan,
        *,
        device: torch.device | str = "cpu",
        recipe: AlgorithmRecipe | None = None,
        config_data: Mapping[str, object] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.spec = spec
        self.algorithm = algorithm
        self.replay = replay
        self.replay_config = replay_config
        self.update_plan = update_plan
        self.device = torch.device(device)
        self.recipe = recipe
        self.config_data = deepcopy(dict(config_data)) if config_data is not None else None
        self.rng = rng or np.random.default_rng()

    @classmethod
    def from_recipe(
        cls,
        spec: EnvironmentSpec,
        algorithm: OffPolicyAlgorithm,
        recipe: AlgorithmRecipe,
        *,
        device: torch.device | str = "cpu",
        registry: ComponentRegistry | None = None,
        rng: np.random.Generator | None = None,
    ) -> OffPolicyTrainer:
        from marl.recipes import compile_recipe
        from marl.registry import DEFAULT_COMPONENT_REGISTRY, ComponentKind

        selected = DEFAULT_COMPONENT_REGISTRY if registry is None else registry
        compile_recipe(recipe, spec, registry=selected)
        replay_config = cast(
            ReplayConfig,
            selected.build(
                ComponentKind.EXPERIENCE_SOURCE,
                recipe.experience.type,
                recipe.experience.options,
                spec,
            ),
        )
        optimizer_config = cast(
            OffPolicyOptimizerConfig,
            selected.build(
                ComponentKind.UPDATE_PLAN,
                recipe.update.type,
                recipe.update.options,
                spec,
            ),
        )
        target_update = cast(
            "TargetUpdate",
            selected.build(
                ComponentKind.TARGET_UPDATE,
                recipe.target_update.type,
                recipe.target_update.options,
                spec,
            ),
        )
        trainable_parameters = tuple(
            parameter for parameter in algorithm.parameters() if parameter.requires_grad
        )
        if not trainable_parameters:
            raise ValueError("algorithm 没有可训练参数")
        optimizer = torch.optim.Adam(
            trainable_parameters, lr=optimizer_config.learning_rate
        )
        return cls(
            spec,
            algorithm,
            TensorReplayBuffer(spec, replay_config.capacity),
            replay_config,
            OffPolicyUpdatePlan(optimizer, optimizer_config.update, target_update),
            device=device,
            recipe=recipe,
            rng=rng,
        )

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
        return self.update_plan.update(self.algorithm, batch)

    def update_batch(self, batch: MARLBatch) -> dict[str, float]:
        """更新显式 batch；用于数学测试和外部经验源。"""

        return self.update_plan.update(self.algorithm, batch.to(self.device))

    def state_dict(self) -> dict[str, Any]:
        from marl.recipes import recipe_to_dict

        return {
            "algorithm": deepcopy(dict(self.algorithm.state_dict())),
            "config": deepcopy(self.config_data),
            "update_plan": self.update_plan.state_dict(),
            "replay": self.replay.state_dict(),
            "recipe": recipe_to_dict(self.recipe) if self.recipe is not None else None,
            "rng": deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        from marl.recipes import recipe_to_dict

        if state.get("config") != self.config_data:
            raise ValueError("checkpoint config 与当前 trainer config 不一致")
        current_recipe = recipe_to_dict(self.recipe) if self.recipe is not None else None
        if state.get("recipe") != current_recipe:
            raise ValueError("checkpoint recipe 与当前 trainer recipe 不一致")
        self.algorithm.load_state_dict(cast(Mapping[str, Any], state["algorithm"]))
        self.update_plan.load_state_dict(cast(Mapping[str, Any], state["update_plan"]))
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
