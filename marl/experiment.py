"""唯一装配入口；算法选择只发生在这里，trainer 不识别算法名称。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar, cast, overload

import numpy as np
import torch

from marl.algorithms.base import BaseMARLAlgorithm
from marl.algorithms.maac import MAAC, MAACConfig
from marl.algorithms.maddpg import MADDPG, MADDPGConfig
from marl.algorithms.mappo import MAPPO, MAPPOConfig
from marl.algorithms.masac import MASAC, MASACConfig
from marl.algorithms.qmix import QMIXConfig
from marl.algorithms.sn_mappo import SNMAPPO, SNMAPPOConfig
from marl.config import AlgorithmConfig, config_to_dict
from marl.envs.base import EnvironmentAdapter, EnvironmentSpec
from marl.envs.sequential import SequentialEnvironment
from marl.runtime import SyncVectorEnv, TensorReplayBuffer
from marl.training.off_policy import OffPolicyTrainer
from marl.training.on_policy import OnPolicyTrainer
from marl.training.optimization import OptimizerRuntime
from marl.training.sequential import SequentialTrainer

A = TypeVar("A", bound=BaseMARLAlgorithm, covariant=True)
T = TypeVar("T", OnPolicyTrainer, OffPolicyTrainer, SequentialTrainer, covariant=True)
Environment = SyncVectorEnv | EnvironmentAdapter | EnvironmentSpec
OffConfig = MAACConfig | MADDPGConfig | MASACConfig | QMIXConfig


@dataclass(frozen=True)
class Experiment(Generic[A, T]):
    """同一次装配得到的 config/algorithm/trainer；三者不再各自重复创建模型。"""

    config: AlgorithmConfig
    algorithm: A
    trainer: T


@overload
def build_experiment(
    environment: Environment,
    config: SNMAPPOConfig,
    *,
    device: str | torch.device = "cpu",
    seed: int = 42,
) -> Experiment[SNMAPPO, SequentialTrainer]: ...


@overload
def build_experiment(
    environment: Environment,
    config: MAPPOConfig,
    *,
    device: str | torch.device = "cpu",
    seed: int = 42,
) -> Experiment[MAPPO, OnPolicyTrainer]: ...


@overload
def build_experiment(
    environment: Environment,
    config: OffConfig,
    *,
    device: str | torch.device = "cpu",
    seed: int = 42,
) -> Experiment[BaseMARLAlgorithm, OffPolicyTrainer]: ...


@overload
def build_experiment(
    environment: Environment,
    config: AlgorithmConfig,
    *,
    device: str | torch.device = "cpu",
    seed: int = 42,
) -> (
    Experiment[BaseMARLAlgorithm, OnPolicyTrainer] | Experiment[BaseMARLAlgorithm, OffPolicyTrainer]
    | Experiment[SNMAPPO, SequentialTrainer]
): ...


def build_experiment(
    environment: Environment,
    config: AlgorithmConfig,
    *,
    device: str | torch.device = "cpu",
    seed: int = 42,
) -> (
    Experiment[BaseMARLAlgorithm, OnPolicyTrainer] | Experiment[BaseMARLAlgorithm, OffPolicyTrainer]
    | Experiment[SNMAPPO, SequentialTrainer]
):
    """先绑定 spec、构造网络并移动设备，再创建 optimizer。

    overload 仅用于静态类型推导，不是多套运行入口。内置类型的显式分支保留
    各 optimizer 的参数归属；不再抽象成 registry、工厂链或 UpdatePlan。
    """
    torch.manual_seed(seed)
    spec = environment if isinstance(environment, EnvironmentSpec) else environment.spec
    snapshot = config_to_dict(config)
    if isinstance(config, SNMAPPOConfig):
        if not isinstance(environment, SequentialEnvironment):
            raise TypeError("SN-MAPPO 需要明确的 sequential_spec/commit_leader/settle 环境")
        sequential = config.build(environment.sequential_spec).to(device)

        def role_runtime(role: MAPPO) -> OptimizerRuntime:
            """按角色 MAPPO 配置装配互不重叠的 actor/critic Adam、裁剪与 minibatch RNG。"""
            settings = role.config.update
            return OptimizerRuntime({
                "actor": torch.optim.Adam(role.policy.parameters(),
                                          lr=settings.resolved_actor_learning_rate),
                "critic": torch.optim.Adam(role.critic.parameters(),
                                           lr=settings.resolved_critic_learning_rate),
            }, {"actor": settings.resolved_actor_max_grad_norm,
                "critic": settings.resolved_critic_max_grad_norm},
                generator=torch.Generator(device=torch.device(device)).manual_seed(seed))

        sequential_trainer = SequentialTrainer(
            environment, sequential, config.leader.advantage.build(),
            (role_runtime(sequential.leader), role_runtime(sequential.coordinator),
             role_runtime(sequential.followers)), device=device, config_data=snapshot, seed=seed,
            leader_trajectories=config.response_trajectories,
        )
        return Experiment(config, sequential, sequential_trainer)
    if isinstance(config, MAPPOConfig):
        if isinstance(environment, EnvironmentSpec):
            raise TypeError("MAPPO 采集 rollout 需要 environment，不能只传 EnvironmentSpec")
        vector = (
            environment if isinstance(environment, SyncVectorEnv) else SyncVectorEnv([environment])
        )
        algorithm = config.build(spec).to(device)
        actor_optimizer = torch.optim.Adam(
            algorithm.policy.parameters(),
            lr=config.update.resolved_actor_learning_rate,
        )
        critic_optimizer = torch.optim.Adam(
            algorithm.critic.parameters(),
            lr=config.update.resolved_critic_learning_rate,
        )
        trainer = OnPolicyTrainer(
            vector,
            algorithm,
            config.advantage.build(),
            config.rollout.horizon or spec.horizon,
            OptimizerRuntime(
                {"actor": actor_optimizer, "critic": critic_optimizer},
                {
                    "actor": config.update.resolved_actor_max_grad_norm,
                    "critic": config.update.resolved_critic_max_grad_norm,
                },
                generator=torch.Generator(device=torch.device(device)).manual_seed(seed),
            ),
            device=device,
            config_data=snapshot,
        )
        return Experiment(config, algorithm, trainer)
    return _off_policy(spec, config, device, seed, snapshot)


def _off_policy(
    spec: EnvironmentSpec,
    config: OffConfig,
    device: str | torch.device,
    seed: int,
    snapshot: dict,
) -> Experiment[BaseMARLAlgorithm, OffPolicyTrainer]:
    # 每个 config.build 都直接调用其算法；无 registry 或无类型组件容器。
    algorithm = config.build(spec).to(device)
    target_update = config.target_update.build()
    if isinstance(config, MAACConfig):
        maac = cast(MAAC, algorithm)
        optimization = OptimizerRuntime(
            {
                "actor": torch.optim.Adam(
                    maac.policy.parameters(),
                    lr=config.update.resolved_actor_learning_rate,
                ),
                "critic": torch.optim.Adam(
                    maac.critic.parameters(),
                    lr=config.update.resolved_critic_learning_rate,
                ),
            },
            {
                "actor": config.update.resolved_actor_max_grad_norm,
                "critic": config.update.resolved_critic_max_grad_norm,
            },
            target_update=target_update,
            amp_dtype=config.update.amp_dtype,
        )
    elif isinstance(config, MADDPGConfig):
        maddpg = cast(MADDPG, algorithm)
        optimization = OptimizerRuntime(
            {
                "actor": torch.optim.Adam(
                    maddpg.policy.parameters(),
                    lr=config.update.resolved_actor_learning_rate,
                ),
                "critic": torch.optim.Adam(
                    maddpg.critics.parameters(),
                    lr=config.update.resolved_critic_learning_rate,
                ),
            },
            {
                "actor": config.update.resolved_actor_max_grad_norm,
                "critic": config.update.resolved_critic_max_grad_norm,
            },
            target_update=target_update,
            amp_dtype=config.update.amp_dtype,
        )
    elif isinstance(config, MASACConfig):
        masac = cast(MASAC, algorithm)
        optimization = OptimizerRuntime(
            {
                "actor": torch.optim.Adam(
                    masac.policy.parameters(),
                    lr=config.update.resolved_actor_learning_rate,
                ),
                "critic": torch.optim.Adam(
                    masac.critics.parameters(),
                    lr=config.update.resolved_critic_learning_rate,
                ),
                "temperature": torch.optim.Adam(
                    masac.entropy_objective.parameters(),
                    lr=config.update.resolved_temperature_learning_rate,
                ),
            },
            {
                "actor": config.update.resolved_actor_max_grad_norm,
                "critic": config.update.resolved_critic_max_grad_norm,
                "temperature": config.update.resolved_temperature_max_grad_norm,
            },
            target_update=target_update,
            amp_dtype=config.update.amp_dtype,
        )
    else:
        optimization = OptimizerRuntime(
            {
                "value": torch.optim.Adam(
                    (p for p in algorithm.parameters() if p.requires_grad),
                    lr=config.update.learning_rate,
                )
            },
            {"value": config.update.max_grad_norm},
            target_update=target_update,
            amp_dtype=config.update.amp_dtype,
        )
    trainer = OffPolicyTrainer(
        spec,
        algorithm,
        TensorReplayBuffer(
            spec,
            config.replay.capacity,
            pin_memory=torch.device(device).type == "cuda",
        ),
        config.replay,
        optimization,
        device=device,
        config_data=snapshot,
        rng=np.random.default_rng(seed),
    )
    return Experiment(config, algorithm, trainer)
