"""唯一装配入口；算法选择只发生在这里，trainer 不识别算法名称。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar, overload

import numpy as np
import torch

from marl.algorithms.base import BaseMARLAlgorithm
from marl.algorithms.maac import MAACConfig
from marl.algorithms.maddpg import MADDPGConfig
from marl.algorithms.mappo import MAPPO, MAPPOConfig
from marl.algorithms.masac import MASACConfig
from marl.algorithms.qmix import QMIXConfig
from marl.config import AlgorithmConfig, config_to_dict
from marl.envs.base import EnvironmentAdapter, EnvironmentSpec
from marl.runtime import SyncVectorEnv, TensorReplayBuffer
from marl.training.off_policy import OffPolicyTrainer, OffPolicyUpdatePlan
from marl.training.on_policy import OnPolicyTrainer, PPOUpdatePlan

A = TypeVar("A", bound=BaseMARLAlgorithm, covariant=True)
T = TypeVar("T", OnPolicyTrainer, OffPolicyTrainer, covariant=True)
Environment = SyncVectorEnv | EnvironmentAdapter | EnvironmentSpec
OffConfig = MAACConfig | MADDPGConfig | MASACConfig | QMIXConfig


@dataclass(frozen=True)
class Experiment(Generic[A, T]):
    config: AlgorithmConfig
    algorithm: A
    trainer: T


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
): ...


def build_experiment(
    environment: Environment,
    config: AlgorithmConfig,
    *,
    device: str | torch.device = "cpu",
    seed: int = 42,
) -> (
    Experiment[BaseMARLAlgorithm, OnPolicyTrainer] | Experiment[BaseMARLAlgorithm, OffPolicyTrainer]
):
    """先绑定 spec、构造网络并移动设备，再创建 optimizer。"""
    torch.manual_seed(seed)
    spec = environment if isinstance(environment, EnvironmentSpec) else environment.spec
    snapshot = config_to_dict(config)
    if isinstance(config, MAPPOConfig):
        if isinstance(environment, EnvironmentSpec):
            raise TypeError("MAPPO 采集 rollout 需要 environment，不能只传 EnvironmentSpec")
        vector = (
            environment if isinstance(environment, SyncVectorEnv) else SyncVectorEnv([environment])
        )
        algorithm = config.build(spec).to(device)
        optimizer = torch.optim.Adam(algorithm.parameters(), lr=config.update.learning_rate)
        trainer = OnPolicyTrainer(
            vector,
            algorithm,
            config.advantage.build(),
            config.rollout.horizon or spec.horizon,
            PPOUpdatePlan(optimizer, config.update, generator=torch.Generator().manual_seed(seed)),
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
    optimizer = torch.optim.Adam(
        (p for p in algorithm.parameters() if p.requires_grad),
        lr=config.update.learning_rate,
    )
    trainer = OffPolicyTrainer(
        spec,
        algorithm,
        TensorReplayBuffer(spec, config.replay.capacity),
        config.replay,
        OffPolicyUpdatePlan(optimizer, config.update, config.target_update.build()),
        device=device,
        config_data=snapshot,
        rng=np.random.default_rng(seed),
    )
    return Experiment(config, algorithm, trainer)
