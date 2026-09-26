from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from marl.algorithms import MADDPG, MADDPGConfig
from marl.core import MARLBatch
from marl.envs import (
    ActionKind,
    EnvironmentSpec,
    EnvironmentStep,
    RewardStructure,
    Transition,
)
from marl.experiment import build_experiment
from marl.modules.critic import IndependentQConfig
from marl.modules.policy import IndependentDeterministicConfig
from marl.target_updates import SoftTargetConfig
from marl.training.off_policy import OffPolicyUpdateConfig, ReplayConfig


def spec() -> EnvironmentSpec:
    return EnvironmentSpec(
        2, 3, 2, 6, ActionKind.CONTINUOUS, 4, RewardStructure.INDIVIDUAL
    )


def recipe():
    return replace(
        MADDPGConfig(),
        policy=IndependentDeterministicConfig(hidden_dim= 16),
        critic=IndependentQConfig(hidden_dim= 16),
        replay=ReplayConfig(capacity= 4, batch_size= 2),
        update=OffPolicyUpdateConfig(learning_rate= 1e-3, max_grad_norm= 0.5),
        target_update=SoftTargetConfig(tau= 0.5),
    )


def batch() -> MARLBatch:
    flags = torch.zeros(4, 2, dtype=torch.bool)
    return MARLBatch(
        observations=torch.randn(4, 2, 3),
        actions=torch.randn(4, 2, 2).tanh(),
        rewards=torch.randn(4, 2),
        next_observations=torch.randn(4, 2, 3),
        terminated=flags,
        truncated=flags,
    )


def transition(value: float) -> Transition:
    current = EnvironmentStep(
        observations=np.full((2, 3), value, dtype=np.float32),
        state=np.full(6, value, dtype=np.float32),
        rewards=np.zeros(2, dtype=np.float32),
        terminated=False,
        truncated=False,
    )
    following = EnvironmentStep(
        observations=np.full((2, 3), value + 1.0, dtype=np.float32),
        state=np.full(6, value + 1.0, dtype=np.float32),
        rewards=np.asarray([value, -value], dtype=np.float32),
        terminated=False,
        truncated=False,
    )
    return Transition(
        current,
        np.zeros((2, 2), dtype=np.float32),
        following,
    )


def test_off_policy_update_changes_online_and_target_parameters() -> None:
    algorithm = MADDPG(spec(), recipe())
    trainer = build_experiment(spec(), recipe(), seed=8).trainer
    algorithm = trainer.algorithm
    optimized_ids = {
        id(parameter)
        for group in trainer.update_plan.optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimized_ids == {
        id(parameter) for parameter in algorithm.parameters() if parameter.requires_grad
    }
    assert not optimized_ids.intersection(
        id(parameter) for parameter in algorithm.target_policy.parameters()
    )
    online_before = next(algorithm.policy.parameters()).detach().clone()
    target_before = next(algorithm.target_policy.parameters()).detach().clone()

    metrics = trainer.update_batch(batch())

    assert not torch.equal(online_before, next(algorithm.policy.parameters()).detach())
    assert not torch.equal(target_before, next(algorithm.target_policy.parameters()).detach())
    assert {"loss", "actor_loss", "critic_loss", "gradient_norm"} <= metrics.keys()
    assert trainer.update_plan.update_count == 1


def test_replay_and_trainer_checkpoint_resume_exact_state(tmp_path: Path) -> None:
    selected_recipe = recipe()
    algorithm = MADDPG(spec(), selected_recipe)
    trainer = build_experiment(spec(), selected_recipe, seed=8).trainer
    algorithm = trainer.algorithm
    trainer.record(transition(1.0))
    trainer.record(transition(2.0))
    assert trainer.ready
    trainer.update()
    checkpoint = tmp_path / "off_policy.pt"
    trainer.save_checkpoint(checkpoint)
    expected = [parameter.detach().clone() for parameter in algorithm.parameters()]

    restored_algorithm = MADDPG(spec(), selected_recipe)
    restored = build_experiment(spec(), selected_recipe, seed=8).trainer
    restored_algorithm = restored.algorithm
    restored.load_checkpoint(checkpoint)
    assert len(restored.replay) == 2
    assert restored.update_plan.update_count == 1
    assert all(
        torch.equal(left, right.detach())
        for left, right in zip(
            expected, restored_algorithm.parameters(), strict=True
        )
    )

    mismatched = replace(
        selected_recipe,
        target_update=SoftTargetConfig(tau= 0.1),
    )
    other_trainer = build_experiment(spec(), mismatched, seed=8).trainer
    with pytest.raises(ValueError, match="config"):
        other_trainer.load_checkpoint(checkpoint)
