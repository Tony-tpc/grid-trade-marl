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
    ) -> Tensor:
        """离策略采集器需要的批量动作接口；协议本身不实现网络。

        Args:
            observations: 浮点观测 [...,N,O]；循环单步为 [B,N,O]，序列为 [B,T,N,O]。
            deterministic: 是否采用确定性动作；具体算法的例外见本函数说明。
            action_mask: 可选 bool [...,N,A]，True 表示合法；连续动作必须为 None。

        Returns:
            离散动作 [B,N] 或连续动作 [B,N,A] Tensor。
        """
        ...


class OffPolicyAlgorithm(Protocol):
    """trainer 实际调用的最小能力，不是供算法继承的第二层基类。

    collector 单独依赖 OffPolicyActor；本协议不要求 act/parameters/target_pairs，
    因为 trainer 不负责动作选择、公式或 optimizer 分配。
    """

    def update(self, batch: MARLBatch, runtime: OptimizerRuntime) -> dict[str, float]:
        """声明离策略算法的一批经验更新能力；由具体算法实现 optimizer 顺序。

        Args:
            batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。
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


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    capacity: int = 100_000
    batch_size: int = 256

    def __post_init__(self) -> None:
        """检查 capacity/batch_size 为正且 batch_size 不超过容量。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            None；非法配置在构造阶段抛 ValueError。
        """
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
        """检查学习率、裁剪上限和可选 BF16 精度。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            None；非法配置在构造阶段抛 ValueError。
        """
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
        """先校验公共参数，再检查 actor/critic 专用覆盖值。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            None；非法配置在构造阶段抛 ValueError。
        """
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
            self.actor_max_grad_norm
            if self.actor_max_grad_norm is not None
            else self.max_grad_norm
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


@dataclass(frozen=True, slots=True)
class TemperatureActorCriticUpdateConfig(ActorCriticUpdateConfig):
    temperature_learning_rate: float | None = None
    temperature_max_grad_norm: float | None = None

    def __post_init__(self) -> None:
        """先校验 actor/critic 配置，再检查 temperature 覆盖值。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            None；非法配置在构造阶段抛 ValueError。
        """
        ActorCriticUpdateConfig.__post_init__(self)
        for name, value in (
            ("temperature_learning_rate", self.temperature_learning_rate),
            ("temperature_max_grad_norm", self.temperature_max_grad_norm),
        ):
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} 必须大于 0 或为 None")

    @property
    def resolved_temperature_learning_rate(self) -> float:
        """解析 temperature 的 learning_rate；专用值为 None 时回退公共 learning_rate。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            有效学习率 float。
        """
        return self.temperature_learning_rate or self.learning_rate

    @property
    def resolved_temperature_max_grad_norm(self) -> float | None:
        """解析 temperature 的 max_grad_norm；专用值为 None 时回退公共 max_grad_norm。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            有效裁剪上限 float 或 None；不会修改冻结配置。
        """
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
        """保存向量环境、动作算法和推理设备；不创建 replay 或 optimizer。

        Args:
            environment: 已构造的同步向量环境；每个 adapter 具有相同 EnvironmentSpec。
            algorithm: 满足本训练器/采集器能力协议的算法实例；复用实验中已构造的网络。
            device: 模型前向或张量迁移使用的设备，例如 cpu 或 cuda。

        Returns:
            None；初始化采集器。
        """
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
        """reset 全部环境后批量推理，逐时刻产出真实经验；可在环境执行前添加探索。

        Args:
            seeds: 各并行环境的 reset 种子，数量必须等于环境数量。
            deterministic: 是否采用确定性动作；具体算法的例外见本函数说明。
            action_transform: 可选动作后处理：接收并返回 NumPy
                联合动作批次；用于探索并保持形状/边界。

        Returns:
            迭代器，每次 yield 长度为 B 的 Transition 元组；不写入 replay，不更新参数。
        """
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
        """绑定算法、replay 与优化器状态，并复制实验配置快照。

        Args:
            spec: 环境规格；N/O/A/S、动作类型、奖励语义和 horizon 均以它为准。
            algorithm: 满足本训练器/采集器能力协议的算法实例；复用实验中已构造的网络。
            replay: 保存真实 transition 的 TensorReplayBuffer 实例。
            replay_config: 容量和采样 batch_size 的静态配置。
            optimization: 算法共享的 OptimizerRuntime，负责状态管理而不定义算法更新顺序。
            device: 模型前向或张量迁移使用的设备，例如 cpu 或 cuda。
            config_data: 实验构造时生成的规范化配置快照；None 只与同为 None 的快照匹配。
            rng: replay 抽样用 NumPy Generator；None 时创建独立生成器。

        Returns:
            None；初始化训练器和 replay 抽样 RNG。
        """
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
        """检查 replay 是否足够采样一批；不要求缓冲区已满。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            bool：len(replay) >= batch_size。
        """
        return len(self.replay) >= self.replay_config.batch_size

    def record(self, transition: Transition) -> None:
        """把单条真实经验加入 replay；满容量后的覆盖由 replay 管理。

        Args:
            transition: 一步经验：当前观测、实际执行动作和下一环境结果；奖励取自 next。

        Returns:
            None；修改 replay。
        """
        self.replay.add(transition)

    def record_many(self, transitions: Sequence[Transition]) -> None:
        """批量加入向量环境经验，复用 replay 的批量写入路径。

        Args:
            transitions: 一组真实 Transition，通常来自同一时刻的多个并行环境。

        Returns:
            None；修改 replay。
        """
        self.replay.record_many(transitions)

    def update(self) -> dict[str, float]:
        """检查样本量并从 replay 抽样，委派具体算法 update。

        从 replay 抽取一批旧经验；只委派算法 update，不重复实现优化循环。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            命名 float 指标；样本不足抛 RuntimeError，延迟同步时返回 {}。
        """

        if not self.ready:
            raise RuntimeError(
                f"replay 样本不足：{len(self.replay)}/{self.replay_config.batch_size}"
            )
        batch = self.replay.sample(
            self.replay_config.batch_size, self.rng, device=self.device
        )
        return self.algorithm.update(batch, self.optimization)

    def update_batch(self, batch: MARLBatch) -> dict[str, float]:
        """接收外部 MARLBatch 并迁移设备，跳过 replay 抽样直接委派更新。

        外部经验来源的入口：跳过 replay 抽样，但复用完全相同的算法 update。

        Args:
            batch: MARLBatch；观测 [...,N,O]，奖励/终止标记 [...,N]，字段要求见说明。

        Returns:
            命名 float 指标；修改网络/optimizer，不把外部 batch 写入 replay。
        """

        return self.algorithm.update(batch.to(self.device), self.optimization)

    def state_dict(self) -> dict[str, Any]:
        """创建完整离策略训练快照，包含模型、优化器、replay、NumPy/Torch RNG。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            schema 4 状态字典；不包含外部采样器和环境当前位置。
        """
        state = build_trainer_checkpoint_state(
            self.algorithm.state_dict(),
            self.config_data,
            self.optimization.state_dict(),
        )
        state["replay"] = self.replay.state_dict()
        state["rng"] = deepcopy(self.rng.bit_generator.state)
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """检查 schema/config，恢复网络、优化器、replay 及 NumPy/Torch RNG。

        Args:
            state: 待验证/恢复的 checkpoint 状态 Mapping；须来自兼容配置和 schema。

        Returns:
            None；直接修改训练状态，不承诺损坏载荷的完整事务回滚。
        """
        validate_trainer_checkpoint_state(state, self.config_data)
        restore_torch_rng_state(state)
        self.algorithm.load_state_dict(cast(Mapping[str, Any], state["algorithm"]))
        self.optimization.load_state_dict(cast(Mapping[str, Any], state["optimization"]))
        self.replay.load_state_dict(cast(dict[str, object], state["replay"]))
        self.rng.bit_generator.state = cast(dict[str, Any], deepcopy(state["rng"]))

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
