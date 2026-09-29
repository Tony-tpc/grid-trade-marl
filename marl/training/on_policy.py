"""通用 on-policy rollout 采集与 checkpoint；更新顺序由具体算法定义。"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
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
    ) -> MARLModelOutput:
        """声明采集动作、old log-prob/value 和循环状态的能力。

        Args:
            observations: 浮点观测 [...,N,O]；循环单步为 [B,N,O]，序列为 [B,T,N,O]。
            state: 可选全局 state [...,S]；None 时使用展平联合观测，尺寸必须与 critic 输入一致。
            deterministic: 是否采用确定性动作；具体算法的例外见本函数说明。
            action_mask: 可选 bool [...,N,A]，True 表示合法；连续动作必须为 None。
            policy_state: 处理当前观测之前的 actor 状态 [K,B,N,H]；LSTM 为 (h,c)，MLP 为 None。
            value_state: 处理当前观测之前的 critic 状态 [K,B,H]；LSTM 为 (h,c)，MLP 为 None。

        Returns:
            MARLModelOutput；actions/log_prob/values 与新状态由具体算法提供。
        """
        ...

    def values(
        self,
        observations: Tensor,
        state: Tensor | None = None,
        *,
        value_state: RecurrentState = None,
    ) -> Tensor:
        """声明只计算 bootstrap value 的能力；采集器在 inference_mode 内调用。

        Args:
            observations: 浮点观测 [...,N,O]；循环单步为 [B,N,O]，序列为 [B,T,N,O]。
            state: 可选全局 state [...,S]；None 时使用展平联合观测，尺寸必须与 critic 输入一致。
            value_state: 处理当前观测之前的 critic 状态 [K,B,H]；LSTM 为 (h,c)，MLP 为 None。

        Returns:
            逐智能体 value Tensor [...,N]。
        """
        ...

    def update(self, experience: PreparedRollout, runtime: OptimizerRuntime) -> dict[str, float]:
        """声明消费 fresh rollout 并执行算法专用更新的能力。

        Args:
            experience: 尚未消费的 PreparedRollout；携带固定
                old log-prob/value、advantage、return。
            runtime: 已装配的命名 optimizer、裁剪上限、目标更新器、随机生成器及计数状态。

        Returns:
            命名 float 指标字典；延迟同步模式允许返回 {}。
        """
        ...

    def state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        """声明与 nn.Module 兼容的模型状态导出接口。

        Args:
            args: 传递给 nn.Module.state_dict 的位置参数，通常省略。
            kwargs: nn.Module.state_dict 的可选关键字，如 destination/prefix/keep_vars。

        Returns:
            参数和 buffers 的 Mapping；不包含 optimizer。
        """
        ...

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> Any:
        """声明与 nn.Module 兼容的权重恢复能力。

        Args:
            state_dict: 待恢复的模型权重与 buffers，遵循 nn.Module.load_state_dict 协议。
            strict: 是否要求模型 state_dict 的参数名称严格匹配。

        Returns:
            沿用 nn.Module 的加载结果；协议不自行实现恢复。
        """
        ...


def _allocate(
    horizon: int,
    num_envs: int,
    *shape: int,
    dtype: torch.dtype,
    pin_memory: bool = False,
) -> Tensor:
    """在 CPU 预分配按时间排列的 rollout 张量；内容尚未初始化。

    Args:
        horizon: 预分配/采集的时间步数 T，必须为正。
        num_envs: 并行环境数量 B，必须为正。
        dtype: 输出张量的数据类型；动作、mask 和浮点数据分别选择相应类型。
        pin_memory: 是否使用锁页 CPU 内存以支持向 CUDA 异步搬运。
        shape: 时间和环境维之后的尾部尺寸，例如 (N,O)。

    Returns:
        Tensor [T,B,*shape]；读取前必须写满。
    """
    return torch.empty((horizon, num_envs, *shape), dtype=dtype, pin_memory=pin_memory)


def _index_optional(value: Tensor | None, indices: Tensor) -> Tensor | None:
    """对可选张量沿第 0 维取样，缺失字段保持缺失。

    Args:
        value: 待索引、分块或复制的 Tensor；可选类型为 None 时保留缺失字段。
        indices: 沿样本/片段维选择的 long 索引，设备须与待索引张量兼容。

    Returns:
        选中行组成的 Tensor，或 None。
    """
    return value.index_select(0, indices) if value is not None else None


def _index_batch(
    batch: MARLBatch,
    indices: Tensor,
    valid_count: int | None = None,
) -> MARLBatch:
    """同步索引 MARLBatch 的数据、extras 和循环初态，避免字段错位。

    Args:
        batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。
        indices: 沿样本/片段维选择的 long 索引，设备须与待索引张量兼容。
        valid_count: 本次选择中的有效时间位置数，不包含 padding，也不乘智能体数。

    Returns:
        新 MARLBatch；数据沿维 0、隐藏状态沿环境/序列维 1 索引。
    """
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
        """验证展平/序列布局与隐藏历史，授予一次性更新消费权。

        Args:
            batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。
            policy_history: actor 输入状态历史 [T,K,B,N,H]；LSTM 为同形 h/c 元组。
            value_history: critic 输入状态历史 [T,K,B,H]；LSTM 为同形 h/c 元组。

        Returns:
            None；保留传入张量引用，不自动复制或 detach 任意外部数据。
        """
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
                ("policy", policy_history, 5),
                ("value", value_history, 4),
            ):

                def check(tensor: Tensor, name: str = name, dimensions: int = dimensions) -> Tensor:
                    """验证单个 h/c 历史张量的维数及 T/B/N 与 rollout 一致。

                    Args:
                        tensor: 待校验或转换布局/设备的单个 Tensor，形状规则见函数说明。
                        name: 状态所属角色 policy/value，决定是否检查智能体维。
                        dimensions: 该角色期望的历史张量维数；actor 为 5，critic 为 4。

                    Returns:
                        原 Tensor；形状不符抛 ValueError。
                    """
                    if (
                        tensor.ndim != dimensions
                        or tensor.shape[0] != batch.observations.shape[1]
                        or tensor.shape[2] != batch.observations.shape[0]
                        or (name == "policy" and tensor.shape[3] != batch.observations.shape[2])
                    ):
                        raise ValueError(f"{name} 历史必须是 [T,K,B,(N),H]，与 rollout 一致")
                    return tensor

                map_state(history, check)
        self.batch = batch
        self.policy_history, self.value_history = policy_history, value_history
        self._consumed = False

    @property
    def size(self) -> int:
        """计算真实环境时间位置数；智能体维不算独立环境样本。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            int：展平路径为 T*B，循环路径也为 B*T。
        """
        shape = self.batch.observations.shape
        return shape[0] * shape[1] if self.recurrent else shape[0]

    @property
    def consumed(self) -> bool:
        """查询是否已开始 minibatch 消费或已转交设备迁移消费权。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            bool；只读属性。
        """
        return self._consumed

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> PreparedRollout:
        """迁移 batch 和全部 h/c 历史；成功后使旧对象失去消费权。

        移动失败不消费；隐藏历史也一起移动，不能只搬 observations。

        Args:
            device: 模型前向或张量迁移使用的设备，例如 cpu 或 cuda。
            non_blocking: 是否请求异步设备复制；实际异步效果取决于设备和 pinned memory。

        Returns:
            具有消费权的新 PreparedRollout；重复迁移旧对象抛 RuntimeError。
        """
        if self._consumed:
            raise RuntimeError("已消费 rollout 不能再次迁移设备")

        def move(t: Tensor) -> Tensor:
            """迁移一个循环状态张量，保留其形状和 dtype。

            Args:
                t: 待校验或转换布局/设备的单个 Tensor，形状规则见函数说明。

            Returns:
                目标设备上的 Tensor。
            """
            return t.to(device, non_blocking=non_blocking)

        moved = PreparedRollout(
            self.batch.to(device, non_blocking=non_blocking),
            policy_history=map_state(self.policy_history, move),
            value_history=map_state(self.value_history, move),
        )
        self._consumed = True
        return moved

    def validate_minibatches(self, mini_batch_size: int, sequence_length: int | None) -> int:
        """在任何参数/统计变化前验证 batch 预算与片段长度。

        在 target 统计/optimizer 变化前校验采样参数。

        Args:
            mini_batch_size: 每批时间位置预算；循环训练按整片段打包，不乘 N。
            sequence_length: 循环片段长度；None 使用整个 rollout，纯 MLP 必须为 None。

        Returns:
            有效片段长度 int；纯 MLP 返回 1。
        """
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
        """将每个环境轨迹切成连续片段，右补齐尾段并选取对应采样初态。

        一次性右补齐和分块；各 epoch 只索引整条片段，不再次搬运原数据。

        Args:
            length: 连续片段长度 L；调用前应完成 minibatch 参数校验。

        Returns:
            (片段 MARLBatch, 每段有效长度列表)；观测 [B*C,L,N,O]，mask [B*C,L]。
        """
        batch = self.batch
        environments, time = batch.observations.shape[:2]
        chunks = (time + length - 1) // length
        padded_time = chunks * length

        def chunks_of(value: Tensor | None, fill: int = 0) -> Tensor | None:
            """对一个时间张量按相同顺序补齐和分块，保持所有字段对齐。

            Args:
                value: 待索引、分块或复制的 Tensor；可选类型为 None 时保留缺失字段。
                fill: 右侧补齐值；动作 mask 用 1，其他数据通常用 0。

            Returns:
                Tensor [B*C,L,...]，输入 None 则返回 None。
            """
            if value is None:
                return None
            padded = value.new_full((environments, padded_time, *value.shape[2:]), fill)
            padded[:, :time].copy_(value)
            return padded.reshape(environments * chunks, length, *value.shape[2:])

        def initial(history: Tensor) -> Tensor:
            """从状态历史每隔 L 步取初态，再按环境优先排列各片段。

            Args:
                history: 循环状态历史；单 Tensor 为 [T,K,B,(N),H]，LSTM 使用同形 (h,c)。

            Returns:
                detach 且连续的状态 [K,B*C,(N),H]。
            """
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
        """单次消费内遍历多个 epoch；MLP 打乱样本，循环网络仅打乱完整片段。

        Args:
            epochs: 同一 fresh rollout 内的 PPO 遍历次数，必须为正。
            mini_batch_size: 每批时间位置预算；循环训练按整片段打包，不乘 N。
            generator: 打乱 mini-batch 顺序的 Torch Generator；设备须与 randperm 一致。
            sequence_length: 循环片段长度；None 使用整个 rollout，纯 MLP 必须为 None。

        Returns:
            MARLBatch 迭代器；首次推进即标记 consumed，禁止另开一次旧数据更新。
        """
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
        """按 spec 预分配 T×B 的 CPU 存储，并将写入位置设为 0。

        Args:
            spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。
            horizon: 预分配/采集的时间步数 T，必须为正。
            num_envs: 并行环境数量 B，必须为正。
            pin_memory: 是否使用锁页 CPU 内存以支持向 CUDA 异步搬运。

        Returns:
            None；隐藏状态历史在首次 add 时按实际网络尺寸分配。
        """
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
        """首次分配循环状态历史，随后保存处理当前观测前的 detached 状态。

        Args:
            history: 循环状态历史；单 Tensor 为 [T,K,B,(N),H]，LSTM 使用同形 (h,c)。
            state: 当前输入隐藏状态；GRU 为 Tensor，LSTM 为 (h,c)，无状态网络为 None。

        Returns:
            更新后的完整历史；无状态网络返回 None。
        """
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
        """校验当前时间片形状并复制 detached 数据，避免采样计算图进入缓存。

        Args:
            target: 预分配存储张量；本次写入 target[position]。
            value: 待索引、分块或复制的 Tensor；可选类型为 None 时保留缺失字段。
            name: 字段或 optimizer 角色名称；用于定位对应状态以及报告错误。

        Returns:
            None；原地写入存储的当前时间位置。
        """
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
        """保存一个时刻全部 B 个环境的采样数据，再推进写入位置。

        Args:
            observations: 浮点观测 [...,N,O]；循环单步为 [B,N,O]，序列为 [B,T,N,O]。
            states: 集中式 critic 的全局状态浮点 [B,S]。
            actions: 联合动作；离散 long [...,N]，连续浮点 [...,N,A]。
            rewards: 本次动作产生的逐智能体奖励浮点 [B,N]。
            old_log_prob: 采样策略对实际动作的 log-prob [B,N]；保存时 detach。
            old_values: 采样时逐智能体 value [B,N]；保存后在 PPO 各 epoch 保持固定。
            terminated: bool [B,N]，真实 MDP 终止，禁止下一状态 bootstrap。
            truncated: bool [B,N]，时间截断；允许当前 bootstrap，但不跨 episode 递推 GAE。
            action_masks: 采样时的 bool [B,N,A]；离散动作必填，连续动作必须为 None。
            policy_state: 处理当前观测之前的 actor 状态 [K,B,N,H]；LSTM 为 (h,c)，MLP 为 None。
            value_state: 处理当前观测之前的 critic 状态 [K,B,H]；LSTM 为 (h,c)，MLP 为 None。

        Returns:
            None；缓存已满、已 finish、mask 或字段形状不符时报错。
        """
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
        """在写满后用最终 bootstrap value 计算 GAE/return，形成一次性可训练 rollout。

        Args:
            next_value: 最后一步之后、reset 之前的 value [B,N]；终止处理交给估计器。
            estimator: AdvantageEstimator，按奖励、old value 与终止标记计算 GAE/return。

        Returns:
            PreparedRollout；MLP 展平为 [T*B,N,O]，循环转为 [B,T,N,O]。
        """
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
            """把时间优先存储转换成模型训练所需的展平或 batch 优先布局。

            Args:
                tensor: 待校验或转换布局/设备的单个 Tensor，形状规则见函数说明。

            Returns:
                MLP 为 [T*B,...]，循环为 [B,T,...] 的 Tensor。
            """
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
        """检查学习率、epoch、batch、裁剪阈值和 sequence_length 范围。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            None；非法配置在构造阶段抛 ValueError。
        """
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
        """解析 actor 的 learning_rate；专用值为 None 时回退公共 learning_rate。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            有效学习率 float。
        """
        return self.actor_learning_rate or self.learning_rate

    @property
    def resolved_critic_learning_rate(self) -> float:
        """解析 critic 的 learning_rate；专用值为 None 时回退公共 learning_rate。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            有效学习率 float。
        """
        return self.critic_learning_rate or self.learning_rate

    @property
    def resolved_actor_max_grad_norm(self) -> float | None:
        """解析 actor 的 max_grad_norm；专用值为 None 时回退公共 max_grad_norm。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            有效裁剪上限 float 或 None；不会修改冻结配置。
        """
        return (
            self.actor_max_grad_norm if self.actor_max_grad_norm is not None else self.max_grad_norm
        )

    @property
    def resolved_critic_max_grad_norm(self) -> float | None:
        """解析 critic 的 max_grad_norm；专用值为 None 时回退公共 max_grad_norm。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            有效裁剪上限 float 或 None；不会修改冻结配置。
        """
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
        """绑定环境、算法、优势估计器与优化状态，并验证 rollout horizon。

        Args:
            environment: 已构造的同步向量环境；每个 adapter 具有相同 EnvironmentSpec。
            algorithm: 满足本训练器/采集器能力协议的算法实例；复用实验中已构造的网络。
            estimator: AdvantageEstimator，按奖励、old value 与终止标记计算 GAE/return。
            rollout_horizon: 每次采集步数，必须位于 [1, environment.spec.horizon]。
            optimization: 算法共享的 OptimizerRuntime，负责状态管理而不定义算法更新顺序。
            device: 模型前向或张量迁移使用的设备，例如 cpu 或 cuda。
            config_data: 实验构造时生成的规范化配置快照；None 只与同为 None 的快照匹配。

        Returns:
            None；不开始环境交互。
        """
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
        """将同步环境结果打包到训练设备供策略或 bootstrap 前向。

        Args:
            steps: 每个并行环境当前的 EnvironmentStep 序列。

        Returns:
            (observations [B,N,O], state [B,S], action_mask [B,N,A] 或 None)。
        """
        return environment_steps_to_tensors(
            steps,
            self.environment.spec,
            device=self.device,
        )

    def collect(self, seeds: list[int]) -> PreparedRollout:
        """重置环境和 h/c，采集固定 T 步，并在最终观测上计算 bootstrap 后生成 rollout。

        Args:
            seeds: 各并行环境的 reset 种子，数量必须等于环境数量。

        Returns:
            未消费 PreparedRollout；中途任一环境结束会报错，不自动部分 reset。
        """
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
        """采集一份 fresh rollout，迁移到训练设备并立即委派算法更新。

        Args:
            seeds: 各并行环境的 reset 种子，数量必须等于环境数量。

        Returns:
            命名 float 指标；完成后旧 rollout 不可再次训练。
        """
        rollout = self.collect(seeds)
        return self.algorithm.update(
            rollout.to(self.device, non_blocking=self.device.type == "cuda"),
            self.optimization,
        )

    def state_dict(self) -> dict[str, Any]:
        """创建完整 rollout 更新边界上的训练快照。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            schema 5 状态字典；保存输入布局，不保存活跃环境、半段 rollout 或外部 seed 调度。
        """
        return build_trainer_checkpoint_state(
            self.algorithm.state_dict(),
            self.config_data,
            self.optimization.state_dict(),
            input_spec=asdict(self.environment.spec),
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """预检配置与网络结构后恢复训练；optimizer/RNG 恢复异常时回滚原状态。

        Args:
            state: 待验证/恢复的 checkpoint 状态 Mapping；须来自兼容配置和 schema。

        Returns:
            None；仅支持完整 rollout 更新后的 checkpoint 恢复。
        """
        validate_trainer_checkpoint_state(
            state,
            self.config_data,
            algorithm_state=self.algorithm.state_dict(),
            input_spec=asdict(self.environment.spec),
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
        """导出当前 trainer 状态到本地文件；在完整算法更新边界调用。

        Args:
            path: 本地 checkpoint 文件路径；保存时自动创建父目录。

        Returns:
            None；父目录自动创建，同名文件会覆盖。
        """
        save_checkpoint_state(self.state_dict(), path)

    def load_checkpoint(
        self, path: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> None:
        """读取可信本地 checkpoint 并委派 load_state_dict 恢复训练状态。

        Args:
            path: 本地 checkpoint 文件路径；保存时自动创建父目录。
            map_location: torch.load 的张量映射设备；加载后的运行设备由已构造实验决定。

        Returns:
            None；当前实验需使用相同配置和兼容网络，外部环境/seed 进度由调用方管理。
        """
        self.load_state_dict(load_checkpoint_state(path, map_location=map_location))


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    horizon: int | None = None

    def __post_init__(self) -> None:
        """检查 horizon 为 None 或正数。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            None；非法配置在构造阶段抛 ValueError。
        """
        if self.horizon is not None and self.horizon < 1:
            raise ValueError("rollout horizon 必须大于 0 或为 None")
