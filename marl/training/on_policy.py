"""通用 on-policy rollout 采集与 checkpoint；更新顺序由具体算法定义。"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from torch import Tensor

from marl.core import MARLBatch, MARLModelOutput
from marl.core.recurrent import RecurrentState, map_state
from marl.envs.base import ActionKind, EnvironmentSpec, EnvironmentStep
from marl.returns import AdvantageEstimator
from marl.runtime import SyncVectorEnv, environment_steps_to_tensors
from marl.training.checkpoint import (
    build_trainer_checkpoint_state,
    load_checkpoint_state,
    restore_torch_rng_state,
    save_checkpoint_state,
    validate_trainer_checkpoint_state,
)
from marl.training.optimization import OptimizerRuntime


class OnPolicyActorCritic(Protocol):
    """采集和更新所需的结构化能力；不实现公式，也不要求算法继承此协议。"""

    def sample(
        self,
        observations: Tensor,
        state: Tensor | None = None,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
        policy_state: RecurrentState = None,
        value_state: RecurrentState = None,
    ) -> MARLModelOutput: ...

    def values(
        self,
        observations: Tensor,
        state: Tensor | None = None,
        *,
        value_state: RecurrentState = None,
    ) -> Tensor: ...

    def update(
        self, experience: PreparedRollout, runtime: OptimizerRuntime
    ) -> dict[str, float]: ...

    def state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]: ...

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> Any: ...


def _allocate(
    horizon: int,
    num_envs: int,
    *shape: int,
    dtype: torch.dtype,
    pin_memory: bool = False,
) -> Tensor:
    return torch.empty((horizon, num_envs, *shape), dtype=dtype, pin_memory=pin_memory)


def _index_optional(value: Tensor | None, indices: Tensor) -> Tensor | None:
    return value.index_select(0, indices) if value is not None else None


def _index_batch(
    batch: MARLBatch,
    indices: Tensor,
    valid_count: int | None = None,
) -> MARLBatch:
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
        policy_state=map_state(batch.policy_state, lambda t: t.index_select(1, indices)),
        value_state=map_state(batch.value_state, lambda t: t.index_select(1, indices)),
        sequence_mask=_index_optional(batch.sequence_mask, indices),
        valid_sample_count=valid_count,
    )


class PreparedRollout:
    """一次性 fresh rollout：MLP 为 [T*B,N,O]；循环为 [B,T,N,O]。

    完整隐藏历史为 [T,K,B,(N),H]，只用于选取片段起始状态。初态来自采样策略且
    detach；片段内部按时间反传，不实施 burn-in。迁移设备会转交消费权。
    """

    def __init__(
        self,
        batch: MARLBatch,
        *,
        policy_history: RecurrentState = None,
        value_history: RecurrentState = None,
    ) -> None:
        if batch.observations.ndim not in (3, 4) or batch.observations.shape[0] == 0:
            raise ValueError("PreparedRollout 需要非空的展平或 [B,T,N,O] 序列观测")
        self.recurrent = batch.observations.ndim == 4
        if self.recurrent and (
            batch.observations.shape[1] == 0 or (policy_history is None and value_history is None)
        ):
            raise ValueError("循环 rollout 需要非空时间维和采样时的隐藏状态历史")
        if not self.recurrent and (policy_history is not None or value_history is not None):
            raise ValueError("已展平 rollout 不得携带循环状态历史")
        if batch.sequence_mask is not None:
            raise ValueError("PreparedRollout 必须是未补齐的原始 rollout，不能重包装片段")
        if self.recurrent:
            for name, history, dimensions in (
                ("policy", policy_history, 5), ("value", value_history, 4),
            ):
                def check(tensor: Tensor, name: str = name, dimensions: int = dimensions) -> Tensor:
                    if (tensor.ndim != dimensions or
                        tensor.shape[0] != batch.observations.shape[1] or
                        tensor.shape[2] != batch.observations.shape[0] or
                        (name == "policy" and tensor.shape[3] != batch.observations.shape[2])):
                        raise ValueError(f"{name} 历史必须是 [T,K,B,(N),H]，与 rollout 一致")
                    return tensor
                map_state(history, check)
        self.batch = batch
        self.policy_history, self.value_history = policy_history, value_history
        self._consumed = False

    @property
    def size(self) -> int:
        shape = self.batch.observations.shape
        return shape[0] * shape[1] if self.recurrent else shape[0]

    @property
    def consumed(self) -> bool:
        return self._consumed

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> PreparedRollout:
        """移动失败不消费；隐藏历史也一起移动，不能只搬 observations。"""
        if self._consumed:
            raise RuntimeError("已消费 rollout 不能再次迁移设备")

        def move(t: Tensor) -> Tensor:
            return t.to(device, non_blocking=non_blocking)

        moved = PreparedRollout(
            self.batch.to(device, non_blocking=non_blocking),
            policy_history=map_state(self.policy_history, move),
            value_history=map_state(self.value_history, move),
        )
        self._consumed = True
        return moved

    def validate_minibatches(self, mini_batch_size: int, sequence_length: int | None) -> int:
        """在 target 统计/optimizer 变化前校验采样参数。"""
        if mini_batch_size < 1:
            raise ValueError("mini_batch_size 必须为正")
        if sequence_length is not None and (
            type(sequence_length) is not int or sequence_length < 1
        ):
            raise ValueError("sequence_length 必须为正整数或 None")
        if not self.recurrent:
            if sequence_length is not None:
                raise ValueError("纯 MLP rollout 不接受 sequence_length")
            return 1
        length = sequence_length or self.batch.observations.shape[1]
        if length < 1 or length > self.batch.observations.shape[1] or length > mini_batch_size:
            raise ValueError("sequence_length 必须在 [1,min(horizon,mini_batch_size)]")
        return length

    def _sequences(self, length: int) -> tuple[MARLBatch, list[int]]:
        """一次性右补齐和分块；各 epoch 只索引整条片段，不再次搬运原数据。"""
        batch = self.batch
        environments, time = batch.observations.shape[:2]
        chunks = (time + length - 1) // length
        padded_time = chunks * length

        def chunks_of(value: Tensor | None, fill: int = 0) -> Tensor | None:
            if value is None:
                return None
            padded = value.new_full((environments, padded_time, *value.shape[2:]), fill)
            padded[:, :time].copy_(value)
            return padded.reshape(environments * chunks, length, *value.shape[2:])

        def initial(history: Tensor) -> Tensor:
            # [C,K,B,...] → [K,B,C,...] → [K,B*C,...]；与上述数据分块完全同序。
            selected = history[::length]
            order = (1, 2, 0, *range(3, selected.ndim))
            return selected.permute(order).flatten(1, 2).detach().contiguous()

        observations = chunks_of(batch.observations)
        assert observations is not None
        mask = torch.arange(padded_time, device=observations.device) < time
        mask = mask.reshape(chunks, length).repeat(environments, 1)
        counts = [
            min(length, time - start)
            for _ in range(environments)
            for start in range(0, time, length)
        ]
        extras = {}
        for key, value in batch.extras.items():
            chunked = chunks_of(value)
            assert chunked is not None
            extras[key] = chunked
        return MARLBatch(
            observations=observations,
            actions=chunks_of(batch.actions),
            rewards=chunks_of(batch.rewards),
            terminated=chunks_of(batch.terminated),
            truncated=chunks_of(batch.truncated),
            state=chunks_of(batch.state),
            # padding 动作 0 必须合法；padding 不参与目标函数。
            action_mask=chunks_of(batch.action_mask, fill=1),
            extras=extras,
            policy_state=map_state(self.policy_history, initial),
            value_state=map_state(self.value_history, initial),
            sequence_mask=mask,
            valid_sample_count=self.size,
        ), counts

    def minibatches(
        self,
        *,
        epochs: int,
        mini_batch_size: int,
        generator: torch.Generator | None = None,
        sequence_length: int | None = None,
    ) -> Iterator[MARLBatch]:
        if self._consumed:
            raise RuntimeError("on-policy rollout 已消费，不能重复用于参数更新")
        if epochs < 1:
            raise ValueError("epochs 必须大于 0")
        length = self.validate_minibatches(mini_batch_size, sequence_length)
        batch, counts = self._sequences(length) if self.recurrent else (self.batch, [])
        self._consumed = True
        size = batch.observations.shape[0]
        chunk_batch_size = mini_batch_size // length
        for _ in range(epochs):
            permutation = torch.randperm(
                size, generator=generator, device=batch.observations.device
            )
            # 循环路径每 epoch 一次小整数同步，避免每个指标/片段对 GPU .item()。
            order = permutation.cpu().tolist() if self.recurrent else []
            for start in range(0, size, chunk_batch_size):
                valid = (
                    sum(counts[i] for i in order[start : start + chunk_batch_size])
                    if self.recurrent
                    else None
                )
                yield _index_batch(batch, permutation[start : start + chunk_batch_size], valid)


class RolloutBuffer:
    """预分配的固定长度 on-policy rollout；每个实例只能 finish 一次。"""

    def __init__(
        self,
        spec: EnvironmentSpec,
        *,
        horizon: int,
        num_envs: int,
        pin_memory: bool = False,
    ) -> None:
        if horizon < 1 or num_envs < 1:
            raise ValueError("horizon 和 num_envs 必须大于 0")
        self.spec = spec
        self.horizon = horizon
        self.num_envs = num_envs
        self.position = 0
        self._finished = False
        self.policy_history: RecurrentState = None
        self.value_history: RecurrentState = None
        n, o, s = spec.num_agents, spec.observation_dim, spec.state_dim
        self.pin_memory = pin_memory
        self.observations = _allocate(
            horizon, num_envs, n, o, dtype=torch.float32, pin_memory=pin_memory
        )
        self.states = _allocate(horizon, num_envs, s, dtype=torch.float32, pin_memory=pin_memory)
        action_shape = (n,) if spec.action_kind == ActionKind.DISCRETE else (n, spec.action_dim)
        action_dtype = torch.long if spec.action_kind == ActionKind.DISCRETE else torch.float32
        self.actions = _allocate(
            horizon,
            num_envs,
            *action_shape,
            dtype=action_dtype,
            pin_memory=pin_memory,
        )
        self.rewards = _allocate(horizon, num_envs, n, dtype=torch.float32, pin_memory=pin_memory)
        self.old_log_prob = _allocate(
            horizon, num_envs, n, dtype=torch.float32, pin_memory=pin_memory
        )
        self.old_values = _allocate(
            horizon, num_envs, n, dtype=torch.float32, pin_memory=pin_memory
        )
        self.terminated = _allocate(horizon, num_envs, n, dtype=torch.bool, pin_memory=pin_memory)
        self.truncated = _allocate(horizon, num_envs, n, dtype=torch.bool, pin_memory=pin_memory)
        self.action_masks = (
            _allocate(
                horizon,
                num_envs,
                n,
                spec.action_dim,
                dtype=torch.bool,
                pin_memory=pin_memory,
            )
            if spec.action_kind == ActionKind.DISCRETE
            else None
        )

    def _record_state(
        self,
        history: RecurrentState,
        state: RecurrentState,
    ) -> RecurrentState:
        if state is None:
            if history is not None:
                raise ValueError("循环状态不能在 rollout 中途消失")
            return None
        if history is None:
            if self.position != 0:
                raise ValueError("循环状态必须从 rollout 第一步开始记录")
            # 在 inference_mode 外创建普通存储；CUDA 状态不逐步拷回 CPU。
            history = map_state(state, lambda t: t.new_empty((self.horizon, *t.shape)))
        targets = history if isinstance(history, tuple) else (history,)
        sources = state if isinstance(state, tuple) else (state,)
        if len(targets) != len(sources):
            raise ValueError("rollout 中途循环状态类型改变")
        for target, source in zip(targets, sources, strict=True):
            assert target is not None
            if target.device != source.device or target.dtype != source.dtype:
                raise ValueError("rollout 中途循环状态 device/dtype 改变")
            self._copy(target, source, "hidden_state")
        return history

    def _copy(self, target: Tensor, value: Tensor, name: str) -> None:
        expected = target[self.position].shape
        if value.shape != expected:
            raise ValueError(f"{name} 应为 {tuple(expected)}，实际为 {tuple(value.shape)}")
        target[self.position].copy_(value.detach())

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
        policy_state: RecurrentState = None,
        value_state: RecurrentState = None,
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
        self.policy_history = self._record_state(self.policy_history, policy_state)
        self.value_history = self._record_state(self.value_history, value_state)
        self.position += 1

    def finish(self, next_value: Tensor, estimator: AdvantageEstimator) -> PreparedRollout:
        if self._finished:
            raise RuntimeError("rollout 已经 finish")
        if self.position != self.horizon:
            raise RuntimeError(f"rollout 尚未填满：{self.position}/{self.horizon}")
        expected_next = (self.num_envs, self.spec.num_agents)
        if next_value.shape != expected_next:
            raise ValueError(f"next_value 应为 {expected_next}，实际为 {tuple(next_value.shape)}")
        self._finished = True
        advantages, returns = estimator.estimate(
            self.rewards,
            self.old_values,
            next_value.detach().to(device="cpu", dtype=torch.float32),
            self.terminated,
            self.truncated,
        )
        recurrent = self.policy_history is not None or self.value_history is not None

        def layout(tensor: Tensor) -> Tensor:
            return (
                tensor.transpose(0, 1).contiguous()
                if recurrent
                else tensor.reshape(self.horizon * self.num_envs, *tensor.shape[2:])
            )

        batch = MARLBatch(
            observations=layout(self.observations),
            actions=layout(self.actions),
            rewards=layout(self.rewards),
            terminated=layout(self.terminated),
            truncated=layout(self.truncated),
            state=layout(self.states),
            action_mask=layout(self.action_masks) if self.action_masks is not None else None,
            extras={
                "old_log_prob": layout(self.old_log_prob).clone(),
                "old_values": layout(self.old_values).clone(),
                "advantages": layout(advantages),
                "returns": layout(returns),
            },
            policy_state=map_state(self.policy_history, lambda t: t[0]),
            value_state=map_state(self.value_history, lambda t: t[0]),
        )
        if self.pin_memory and recurrent:
            # transpose/contiguous 及 GAE 产生的新 CPU 张量不再继承预分配的 pinned 属性。
            # 在第一次赋予消费权之前统一 pin；CUDA 状态保留原设备，不重包装已消费对象。
            batch = batch.pin_memory()
        return PreparedRollout(
            batch,
            policy_history=self.policy_history,
            value_history=self.value_history,
        )


@dataclass(frozen=True, slots=True)
class PPOUpdateConfig:
    learning_rate: float = 3e-4
    actor_learning_rate: float | None = None
    critic_learning_rate: float | None = None
    epochs: int = 4
    mini_batch_size: int = 256
    max_grad_norm: float | None = 10.0
    actor_max_grad_norm: float | None = None
    critic_max_grad_norm: float | None = None
    value_clip_ratio: float | None = None
    sequence_length: int | None = None

    def __post_init__(self) -> None:
        if self.sequence_length is not None and (
            type(self.sequence_length) is not int or self.sequence_length < 1
        ):
            raise ValueError("sequence_length 必须为正整数或 None")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必须大于 0")
        for name, value in (
            ("actor_learning_rate", self.actor_learning_rate),
            ("critic_learning_rate", self.critic_learning_rate),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} 必须大于 0 或为 None")
        if self.epochs < 1 or self.mini_batch_size < 1:
            raise ValueError("epochs 和 mini_batch_size 必须大于 0")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm 必须大于 0 或为 None")
        for name, value in (
            ("actor_max_grad_norm", self.actor_max_grad_norm),
            ("critic_max_grad_norm", self.critic_max_grad_norm),
            ("value_clip_ratio", self.value_clip_ratio),
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
            self.actor_max_grad_norm if self.actor_max_grad_norm is not None else self.max_grad_norm
        )

    @property
    def resolved_critic_max_grad_norm(self) -> float | None:
        return (
            self.critic_max_grad_norm
            if self.critic_max_grad_norm is not None
            else self.max_grad_norm
        )


class OnPolicyTrainer:
    """采集 fresh rollout，并调用具体 on-policy 算法的 update。"""

    def __init__(
        self,
        environment: SyncVectorEnv,
        algorithm: OnPolicyActorCritic,
        estimator: AdvantageEstimator,
        rollout_horizon: int,
        optimization: OptimizerRuntime,
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
        self.optimization = optimization
        self.device = torch.device(device)
        self.config_data = deepcopy(dict(config_data)) if config_data is not None else None

    def _tensors(self, steps: Sequence[EnvironmentStep]) -> tuple[Tensor, Tensor, Tensor | None]:
        return environment_steps_to_tensors(
            steps,
            self.environment.spec,
            device=self.device,
        )

    def collect(self, seeds: list[int]) -> PreparedRollout:
        if len(seeds) != len(self.environment.adapters):
            raise ValueError("seeds 数量必须与并行环境数量一致")
        steps = self.environment.reset(seeds)
        buffer = RolloutBuffer(
            self.environment.spec,
            horizon=self.rollout_horizon,
            num_envs=len(self.environment.adapters),
            pin_memory=self.device.type == "cuda",
        )
        policy_state: RecurrentState = None
        value_state: RecurrentState = None
        for index in range(self.rollout_horizon):
            # 环境原本在 CPU；保留原始张量写 rollout，避免先上卡再拷回观测/state/mask。
            cpu_observations, cpu_states, cpu_masks = environment_steps_to_tensors(
                steps, self.environment.spec, device="cpu"
            )
            observations = cpu_observations.to(self.device)
            states = cpu_states.to(self.device)
            action_masks = cpu_masks.to(self.device) if cpu_masks is not None else None
            with torch.inference_mode():
                output = self.algorithm.sample(
                    observations,
                    states,
                    action_mask=action_masks,
                    policy_state=policy_state,
                    value_state=value_state,
                )
            if output.log_prob is None or output.values is None:
                raise RuntimeError("on-policy sample 必须返回 log_prob 和 values")
            cpu_actions = output.actions.cpu()
            next_steps = self.environment.step(cpu_actions.numpy())
            rewards = torch.as_tensor(
                np.stack([step.rewards for step in next_steps]), dtype=torch.float32
            )
            terminated = (
                torch.as_tensor(
                    np.asarray([step.terminated for step in next_steps]), dtype=torch.bool
                )
                .unsqueeze(-1)
                .expand(-1, self.environment.spec.num_agents)
            )
            truncated = (
                torch.as_tensor(
                    np.asarray([step.truncated for step in next_steps]), dtype=torch.bool
                )
                .unsqueeze(-1)
                .expand(-1, self.environment.spec.num_agents)
            )
            buffer.add(
                observations=cpu_observations,
                states=cpu_states,
                actions=cpu_actions,
                rewards=rewards,
                old_log_prob=output.log_prob,
                old_values=output.values,
                terminated=terminated,
                truncated=truncated,
                action_masks=cpu_masks,
                # 第一步 None 等价于零初态，模板尺寸从此次输出获得，保存的是输入状态。
                policy_state=(
                    policy_state
                    if policy_state is not None
                    else map_state(output.policy_state, torch.zeros_like)
                ),
                value_state=(
                    value_state
                    if value_state is not None
                    else map_state(output.value_state, torch.zeros_like)
                ),
            )
            policy_state, value_state = output.policy_state, output.value_state
            steps = next_steps
            if index + 1 < self.rollout_horizon and any(step.done for step in steps):
                raise RuntimeError(
                    "环境在 rollout_horizon 之前结束；请缩短 horizon 或支持部分 reset"
                )

        next_observations, next_states, _ = self._tensors(steps)
        with torch.inference_mode():
            next_value = self.algorithm.values(
                next_observations, next_states, value_state=value_state
            )
        return buffer.finish(next_value, self.estimator)

    def train_rollout(self, seeds: list[int]) -> dict[str, float]:
        rollout = self.collect(seeds)
        return self.algorithm.update(
            rollout.to(self.device, non_blocking=self.device.type == "cuda"),
            self.optimization,
        )

    def state_dict(self) -> dict[str, Any]:
        return build_trainer_checkpoint_state(
            self.algorithm.state_dict(),
            self.config_data,
            self.optimization.state_dict(),
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        validate_trainer_checkpoint_state(
            state, self.config_data, algorithm_state=self.algorithm.state_dict(),
        )
        # schema/config/网络形状已预检。对 optimizer/RNG 等损坏载荷提供事务回滚：
        # PyTorch 的 load_state_dict 可能先修改部分对象再抛错，不能留下半恢复状态。
        previous = self.state_dict()
        try:
            self.algorithm.load_state_dict(cast(Mapping[str, Any], state["algorithm"]))
            self.optimization.load_state_dict(cast(Mapping[str, Any], state["optimization"]))
            restore_torch_rng_state(state)
        except Exception:
            self.algorithm.load_state_dict(previous["algorithm"])
            self.optimization.load_state_dict(previous["optimization"])
            restore_torch_rng_state(previous)
            raise

    def save_checkpoint(self, path: str | Path) -> None:
        save_checkpoint_state(self.state_dict(), path)

    def load_checkpoint(
        self, path: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> None:
        self.load_state_dict(load_checkpoint_state(path, map_location=map_location))


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    horizon: int | None = None

    def __post_init__(self) -> None:
        if self.horizon is not None and self.horizon < 1:
            raise ValueError("rollout horizon 必须大于 0 或为 None")
