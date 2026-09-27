"""分离 optimizer 的梯度边界与专用学习率回归。"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from marl.algorithms import MAAC, MADDPG, MASAC, MAACConfig, MADDPGConfig, MASACConfig
from marl.core import MARLBatch
from marl.envs import ActionKind, EnvironmentSpec, RewardStructure
from marl.experiment import build_experiment
from marl.training.off_policy import ActorCriticUpdateConfig, TemperatureActorCriticUpdateConfig


def _spec(kind: ActionKind) -> EnvironmentSpec:
    action_dim = 4 if kind == ActionKind.DISCRETE else 2
    return EnvironmentSpec(2, 3, action_dim, 6, kind, 8, RewardStructure.INDIVIDUAL)


def _batch(kind: ActionKind) -> MARLBatch:
    observations = torch.randn(5, 2, 3)
    actions = (
        torch.randint(0, 4, (5, 2))
        if kind == ActionKind.DISCRETE
        else torch.randn(5, 2, 2).tanh()
    )
    flags = torch.zeros(5, 2, dtype=torch.bool)
    return MARLBatch(
        observations=observations,
        actions=actions,
        rewards=torch.randn(5, 2),
        next_observations=torch.randn_like(observations),
        terminated=flags,
        truncated=flags,
    )


def _has_gradient(module: nn.Module) -> bool:
    return any(parameter.grad is not None for parameter in module.parameters())


def _has_nonzero_gradient(module: nn.Module) -> bool:
    return any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in module.parameters()
    )


def test_maac_critic_and_per_agent_actor_gradients_are_isolated() -> None:
    algorithm = MAAC(_spec(ActionKind.DISCRETE), MAACConfig())
    batch = _batch(ActionKind.DISCRETE)

    algorithm.zero_grad(set_to_none=True)
    algorithm.compute_critic_loss_bundle(batch).total.backward()
    assert _has_gradient(algorithm.get_submodule("critic"))
    assert not _has_gradient(algorithm.get_submodule("policy"))

    algorithm.zero_grad(set_to_none=True)
    algorithm.compute_actor_loss_bundle(batch, 0).total.backward()
    assert _has_nonzero_gradient(algorithm.get_submodule("policy.actors.0"))
    assert not _has_nonzero_gradient(algorithm.get_submodule("policy.actors.1"))
    assert not _has_gradient(algorithm.get_submodule("critic"))


def test_maddpg_and_masac_loss_groups_have_disjoint_gradients() -> None:
    batch = _batch(ActionKind.CONTINUOUS)
    for algorithm in (
        MADDPG(_spec(ActionKind.CONTINUOUS), MADDPGConfig()),
        MASAC(_spec(ActionKind.CONTINUOUS), MASACConfig()),
    ):
        algorithm.zero_grad(set_to_none=True)
        algorithm.compute_critic_loss_bundle(batch).total.backward()
        assert _has_gradient(algorithm.get_submodule("critics"))
        assert not _has_gradient(algorithm.get_submodule("policy"))

        algorithm.zero_grad(set_to_none=True)
        algorithm.compute_actor_loss_bundle(batch).total.backward()
        assert _has_gradient(algorithm.get_submodule("policy"))
        assert not _has_gradient(algorithm.get_submodule("critics"))

    masac = MASAC(_spec(ActionKind.CONTINUOUS), MASACConfig())
    masac.zero_grad(set_to_none=True)
    masac.compute_temperature_loss_bundle(batch).total.backward()
    assert masac.entropy_objective.log_alpha.grad is not None
    assert not _has_gradient(masac.get_submodule("policy"))
    assert not _has_gradient(masac.get_submodule("critics"))


def test_typed_update_configs_control_each_optimizer_and_step_count() -> None:
    cases = (
        replace(
            MAACConfig(),
            update=ActorCriticUpdateConfig(actor_learning_rate=1e-4, critic_learning_rate=2e-4),
        ),
        replace(
            MADDPGConfig(),
            update=ActorCriticUpdateConfig(actor_learning_rate=1e-4, critic_learning_rate=2e-4),
        ),
        replace(
            MASACConfig(),
            update=TemperatureActorCriticUpdateConfig(
                actor_learning_rate=1e-4,
                critic_learning_rate=2e-4,
                temperature_learning_rate=3e-4,
            ),
        ),
    )
    for config in cases:
        kind = ActionKind.DISCRETE if isinstance(config, MAACConfig) else ActionKind.CONTINUOUS
        trainer = build_experiment(_spec(kind), config).trainer
        runtime = trainer.optimization
        assert runtime.optimizer("actor").param_groups[0]["lr"] == 1e-4
        assert runtime.optimizer("critic").param_groups[0]["lr"] == 2e-4
        trainer.update_batch(_batch(kind))
        expected_steps = 3 if isinstance(config, (MAACConfig, MASACConfig)) else 2
        assert runtime.optimizer_step_count == expected_steps
        if isinstance(config, MASACConfig):
            assert runtime.optimizer("temperature").param_groups[0]["lr"] == 3e-4
