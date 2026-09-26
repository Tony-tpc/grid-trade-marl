"""通用 on-policy rollout 和 PPO mini-batch 更新计划。"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from torch import Tensor, nn

from marl.core import MARLBatch, MARLModelOutput
from marl.envs.base import ActionKind, EnvironmentSpec, EnvironmentStep
from marl.objectives import LossBundle
from marl.returns import AdvantageEstimator
from marl.runtime import SyncVectorEnv


class LossComputingModule(Protocol):
    """PPOUpdatePlan 对具体算法的唯一要求。"""

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...

    def compute_loss_bundle(self, batch: MARLBatch) -> LossBundle: ...


class OnPolicyActorCritic(LossComputingModule, Protocol):
    """采样器需要的 actor-critic 能力，不绑定具体算法名称。"""

    def sample(
        self,
        observations: Tensor,
        state: Tensor | None = None,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
    ) -> MARLModelOutput: ...

    def values(self, observations: Tensor, state: Tensor | None = None) -> Tensor: ...

    def state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]: ...

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> Any: ...


def _allocate(
    horizon: int, num_envs: int, *shape: int, dtype: torch.dtype
) -> Tensor:
    return torch.empty((horizon, num_envs, *shape), dtype=dtype)


def _index_optional(value: Tensor | None, indices: Tensor) -> Tensor | None:
    return value.index_select(0, indices) if value is not None else None


def _index_batch(batch: MARLBatch, indices: Tensor) -> MARLBatch:
    return MARLBatch(
        observations=batch.observations.index_select(0, indices),
        actions=_index_optional(batch.actions, indices),
        rewards=_index_optional(batch.rewards, indices),
        next_observations=_index_optional(batch.next_observations, indices),
        terminated=_index_optional(batch.terminated, indices),
        truncated=_index_optional(batch.truncated, indices),
        state=_index_optional(batch.state, indices),
        next_state=_index_optional(batch.next_state, indices),
        action_mask=_index_optional(batch.action_mask, indices),
        next_action_mask=_index_optional(batch.next_action_mask, indices),
        extras={key: value.index_select(0, indices) for key, value in batch.extras.items()},
    )


class PreparedRollout:
    """完成 GAE 后的一次性 on-policy 训练批次。"""

    def __init__(self, batch: MARLBatch) -> None:
        self.batch = batch
        self._consumed = False

    @property
    def size(self) -> int:
        return self.batch.observations.shape[0]

    @property
    def consumed(self) -> bool:
        return self._consumed

    def minibatches(
        self,
        *,
        epochs: int,
        mini_batch_size: int,
        generator: torch.Generator | None = None,
    ) -> Iterator[MARLBatch]:
        if self._consumed:
            raise RuntimeError("on-policy rollout 已消费，不能重复用于参数更新")
        if epochs < 1 or mini_batch_size < 1:
            raise ValueError("epochs 和 mini_batch_size 必须大于 0")
        self._consumed = True
        for _ in range(epochs):
            permutation = torch.randperm(self.size, generator=generator)
            for start in range(0, self.size, mini_batch_size):
                yield _index_batch(self.batch, permutation[start : start + mini_batch_size])


class RolloutBuffer:
    """预分配的固定长度 on-policy rollout；每个实例只能 finish 一次。"""

    def __init__(self, spec: EnvironmentSpec, *, horizon: int, num_envs: int) -> None:
        if horizon < 1 or num_envs < 1:
            raise ValueError("horizon 和 num_envs 必须大于 0")
        self.spec = spec
        self.horizon = horizon
        self.num_envs = num_envs
        self.position = 0
        self._finished = False
        n, o, s = spec.num_agents, spec.observation_dim, spec.state_dim
        self.observations = _allocate(horizon, num_envs, n, o, dtype=torch.float32)
        self.states = _allocate(horizon, num_envs, s, dtype=torch.float32)
        action_shape = (n,) if spec.action_kind == ActionKind.DISCRETE else (n, spec.action_dim)
        action_dtype = torch.long if spec.action_kind == ActionKind.DISCRETE else torch.float32
        self.actions = _allocate(horizon, num_envs, *action_shape, dtype=action_dtype)
        self.rewards = _allocate(horizon, num_envs, n, dtype=torch.float32)
        self.old_log_prob = _allocate(horizon, num_envs, n, dtype=torch.float32)
        self.old_values = _allocate(horizon, num_envs, n, dtype=torch.float32)
        self.terminated = _allocate(horizon, num_envs, n, dtype=torch.bool)
        self.truncated = _allocate(horizon, num_envs, n, dtype=torch.bool)
        self.action_masks = (
            _allocate(horizon, num_envs, n, spec.action_dim, dtype=torch.bool)
            if spec.action_kind == ActionKind.DISCRETE
            else None
        )

    def _copy(self, target: Tensor, value: Tensor, name: str) -> None:
        expected = target[self.position].shape
        if value.shape != expected:
            raise ValueError(f"{name} 应为 {tuple(expected)}，实际为 {tuple(value.shape)}")
        target[self.position].copy_(value.detach().to(device="cpu", dtype=target.dtype))

    def add(
        self,
        *,
        observations: Tensor,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        old_log_prob: Tensor,
        old_values: Tensor,
        terminated: Tensor,
        truncated: Tensor,
        action_masks: Tensor | None = None,
    ) -> None:
        if self._finished or self.position >= self.horizon:
            raise RuntimeError("rollout 已满或已经 finish")
        self._copy(self.observations, observations, "observations")
        self._copy(self.states, states, "states")
        self._copy(self.actions, actions, "actions")
        self._copy(self.rewards, rewards, "rewards")
        self._copy(self.old_log_prob, old_log_prob, "old_log_prob")
        self._copy(self.old_values, old_values, "old_values")
        self._copy(self.terminated, terminated, "terminated")
        self._copy(self.truncated, truncated, "truncated")
        if self.action_masks is None:
            if action_masks is not None:
                raise ValueError("连续动作 rollout 不应提供 action_masks")
        else:
            if action_masks is None:
                raise ValueError("离散动作 rollout 必须提供 action_masks")
            self._copy(self.action_masks, action_masks, "action_masks")
        self.position += 1

    def finish(self, next_value: Tensor, estimator: AdvantageEstimator) -> PreparedRollout:
        if self._finished:
            raise RuntimeError("rollout 已经 finish")
        if self.position != self.horizon:
            raise RuntimeError(f"rollout 尚未填满：{self.position}/{self.horizon}")
        expected_next = (self.num_envs, self.spec.num_agents)
        if next_value.shape != expected_next:
            raise ValueError(
                f"next_value 应为 {expected_next}，实际为 {tuple(next_value.shape)}"
            )
        self._finished = True
        advantages, returns = estimator.estimate(
            self.rewards,
            self.old_values,
            next_value.detach().to(device="cpu", dtype=torch.float32),
            self.terminated,
            self.truncated,
        )
        flat = self.horizon * self.num_envs
        batch = MARLBatch(
            observations=self.observations.reshape(flat, *self.observations.shape[2:]),
            actions=self.actions.reshape(flat, *self.actions.shape[2:]),
            rewards=self.rewards.reshape(flat, self.spec.num_agents),
            terminated=self.terminated.reshape(flat, self.spec.num_agents),
            truncated=self.truncated.reshape(flat, self.spec.num_agents),
            state=self.states.reshape(flat, self.spec.state_dim),
            action_mask=(
                self.action_masks.reshape(flat, self.spec.num_agents, self.spec.action_dim)
                if self.action_masks is not None
                else None
            ),
            extras={
                "old_log_prob": self.old_log_prob.reshape(flat, self.spec.num_agents).clone(),
                "old_values": self.old_values.reshape(flat, self.spec.num_agents).clone(),
                "advantages": advantages.reshape(flat, self.spec.num_agents),
                "returns": returns.reshape(flat, self.spec.num_agents),
            },
        )
        return PreparedRollout(batch)


@dataclass(frozen=True, slots=True)
class PPOUpdateConfig:
    learning_rate: float = 3e-4
    epochs: int = 4
    mini_batch_size: int = 256
    max_grad_norm: float | None = 10.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必须大于 0")
        if self.epochs < 1 or self.mini_batch_size < 1:
            raise ValueError("epochs 和 mini_batch_size 必须大于 0")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm 必须大于 0 或为 None")


class PPOUpdatePlan:
    """对 PreparedRollout 执行多 epoch、随机 mini-batch 更新。"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: PPOUpdateConfig,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.config = config
        self.generator = generator
        self.update_count = 0

    @staticmethod
    def _explained_variance(old_values: Tensor, returns: Tensor) -> Tensor:
        return_variance = returns.var(unbiased=False)
        if return_variance <= 1e-12:
            return torch.zeros((), dtype=returns.dtype, device=returns.device)
        return 1.0 - (returns - old_values).var(unbiased=False) / return_variance

    def update(
        self, algorithm: LossComputingModule, experience: PreparedRollout
    ) -> dict[str, float]:
        device = next(algorithm.parameters()).device
        totals: dict[str, Tensor] = {}
        mini_batch_count = 0
        for mini_batch in experience.minibatches(
            epochs=self.config.epochs,
            mini_batch_size=self.config.mini_batch_size,
            generator=self.generator,
        ):
            mini_batch = mini_batch.to(device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            bundle = algorithm.compute_loss_bundle(mini_batch)
            bundle.total.backward()
            if self.config.max_grad_norm is None:
                gradient_norm = torch.linalg.vector_norm(
                    torch.stack(
                        [
                            parameter.grad.detach().norm()
                            for parameter in algorithm.parameters()
                            if parameter.grad is not None
                        ]
                    )
                )
            else:
                gradient_norm = nn.utils.clip_grad_norm_(
                    algorithm.parameters(), self.config.max_grad_norm
                )
            self.optimizer.step()
            metrics = dict(bundle.terms)
            metrics["gradient_norm"] = gradient_norm.detach()
            for name, value in metrics.items():
                totals[name] = totals.get(name, torch.zeros_like(value)) + value.detach()
            mini_batch_count += 1

        if mini_batch_count == 0:
            raise RuntimeError("update plan 未产生任何 mini-batch")
        self.update_count += 1
        averages = {name: value / mini_batch_count for name, value in totals.items()}
        batch = experience.batch
        averages["explained_variance"] = self._explained_variance(
            batch.extras["old_values"], batch.extras["returns"]
        )
        names = tuple(averages)
        values = torch.stack([averages[name].reshape(()) for name in names]).cpu().tolist()
        return dict(zip(names, values, strict=True))

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer": deepcopy(self.optimizer.state_dict()),
            "generator": self.generator.get_state().clone() if self.generator is not None else None,
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.optimizer.load_state_dict(cast(dict[str, Any], state["optimizer"]))
        self.update_count = int(state["update_count"])
        if state["generator"] is not None:
            if self.generator is None:
                self.generator = torch.Generator()
            self.generator.set_state(state["generator"].cpu())


class OnPolicyTrainer:
    """从并行环境采集 fresh rollout，并交给通用 UpdatePlan。"""

    def __init__(
        self,
        environment: SyncVectorEnv,
        algorithm: OnPolicyActorCritic,
        estimator: AdvantageEstimator,
        rollout_horizon: int,
        update_plan: PPOUpdatePlan,
        *,
        device: torch.device | str = "cpu",
        config_data: Mapping[str, object] | None = None,
    ) -> None:
        if rollout_horizon < 1 or rollout_horizon > environment.spec.horizon:
            raise ValueError("rollout_horizon 必须位于 [1, environment horizon]")
        self.environment = environment
        self.algorithm = algorithm
        self.estimator = estimator
        self.rollout_horizon = rollout_horizon
        self.update_plan = update_plan
        self.device = torch.device(device)
        self.config_data = deepcopy(dict(config_data)) if config_data is not None else None

    def _tensors(
        self, steps: Sequence[EnvironmentStep]
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        observations = torch.as_tensor(
            np.stack([step.observations for step in steps]),
            dtype=torch.float32,
            device=self.device,
        )
        states = torch.as_tensor(
            np.stack([step.state for step in steps]),
            dtype=torch.float32,
            device=self.device,
        )
        action_masks = None
        if self.environment.spec.action_kind == ActionKind.DISCRETE:
            if any(step.action_mask is None for step in steps):
                raise ValueError("离散环境的每个 step 都必须提供 action_mask")
            action_masks = torch.as_tensor(
                np.stack([step.action_mask for step in steps if step.action_mask is not None]),
                dtype=torch.bool,
                device=self.device,
            )
        return observations, states, action_masks

    def collect(self, seeds: list[int]) -> PreparedRollout:
        if len(seeds) != len(self.environment.adapters):
            raise ValueError("seeds 数量必须与并行环境数量一致")
        steps = self.environment.reset(seeds)
        buffer = RolloutBuffer(
            self.environment.spec,
            horizon=self.rollout_horizon,
            num_envs=len(self.environment.adapters),
        )
        for index in range(self.rollout_horizon):
            observations, states, action_masks = self._tensors(steps)
            with torch.inference_mode():
                output = self.algorithm.sample(
                    observations,
                    states,
                    action_mask=action_masks,
                )
            if output.log_prob is None or output.values is None:
                raise RuntimeError("on-policy sample 必须返回 log_prob 和 values")
            next_steps = self.environment.step(output.actions.cpu().numpy())
            rewards = torch.as_tensor(
                np.stack([step.rewards for step in next_steps]), dtype=torch.float32
            )
            terminated = torch.as_tensor(
                np.asarray([step.terminated for step in next_steps]), dtype=torch.bool
            ).unsqueeze(-1).expand(-1, self.environment.spec.num_agents)
            truncated = torch.as_tensor(
                np.asarray([step.truncated for step in next_steps]), dtype=torch.bool
            ).unsqueeze(-1).expand(-1, self.environment.spec.num_agents)
            buffer.add(
                observations=observations,
                states=states,
                actions=output.actions,
                rewards=rewards,
                old_log_prob=output.log_prob,
                old_values=output.values,
                terminated=terminated,
                truncated=truncated,
                action_masks=action_masks,
            )
            steps = next_steps
            if index + 1 < self.rollout_horizon and any(step.done for step in steps):
                raise RuntimeError(
                    "环境在 rollout_horizon 之前结束；请缩短 horizon 或支持部分 reset"
                )

        next_observations, next_states, _ = self._tensors(steps)
        with torch.inference_mode():
            next_value = self.algorithm.values(next_observations, next_states)
        return buffer.finish(next_value, self.estimator)

    def train_rollout(self, seeds: list[int]) -> dict[str, float]:
        return self.update_plan.update(self.algorithm, self.collect(seeds))

    def state_dict(self) -> dict[str, Any]:
        return {
            "algorithm": deepcopy(dict(self.algorithm.state_dict())),
            "config": deepcopy(self.config_data),
            "torch_rng": torch.get_rng_state().clone(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "update_plan": self.update_plan.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("config") != self.config_data:
            raise ValueError("checkpoint config 与当前 trainer config 不一致")
        torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng"]])
        self.algorithm.load_state_dict(cast(Mapping[str, Any], state["algorithm"]))
        self.update_plan.load_state_dict(cast(Mapping[str, Any], state["update_plan"]))

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

@dataclass(frozen=True, slots=True)
class RolloutConfig:
    horizon: int | None = None

    def __post_init__(self) -> None:
        if self.horizon is not None and self.horizon < 1:
            raise ValueError("rollout horizon 必须大于 0 或为 None")
