from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from marl.algorithms import MAPPO
from marl.algorithms.mappo import MAPPOConfig
from marl.core import MARLBatch
from marl.envs import (
    ActionKind,
    EnvironmentAdapter,
    EnvironmentSpec,
    EnvironmentStep,
    RewardStructure,
)
from marl.experiment import build_experiment
from marl.modules.critic import CentralizedValueConfig
from marl.modules.policy import IndependentDiscreteConfig
from marl.runtime import SyncVectorEnv
from marl.training.on_policy import PPOUpdateConfig, RolloutConfig


def environment_spec() -> EnvironmentSpec:
    return EnvironmentSpec(
        num_agents=2,
        observation_dim=3,
        action_dim=4,
        state_dim=6,
        action_kind=ActionKind.DISCRETE,
        horizon=3,
        reward_structure=RewardStructure.INDIVIDUAL,
    )


def training_config() -> MAPPOConfig:
    config = MAPPOConfig()
    return replace(
        config,
        policy=IndependentDiscreteConfig(hidden_dim=16),
        critic=CentralizedValueConfig(hidden_dim=16),
        rollout=RolloutConfig(horizon=3),
        update=PPOUpdateConfig(
            learning_rate=1e-3,
            epochs=2,
            mini_batch_size=2,
            max_grad_norm=0.5,
        ),
    )


def sample_batch() -> MARLBatch:
    return MARLBatch(
        observations=torch.randn(5, 2, 3),
        actions=torch.randint(0, 4, (5, 2)),
        state=torch.randn(5, 6),
        action_mask=torch.ones(5, 2, 4, dtype=torch.bool),
        extras={
            "old_log_prob": torch.randn(5, 2),
            "advantages": torch.randn(5, 2),
            "returns": torch.randn(5, 2),
        },
    )


def test_config_construction_is_reproducible_with_fixed_seed() -> None:
    config = training_config()
    torch.manual_seed(41)
    first = MAPPO(environment_spec(), config)
    torch.manual_seed(41)
    second = MAPPO(environment_spec(), config)
    batch = sample_batch()

    assert torch.allclose(
        first.policy.logits(batch.observations, batch.action_mask),
        second.policy.logits(batch.observations, batch.action_mask),
    )
    assert batch.state is not None
    assert torch.allclose(first.critic(batch.state), second.critic(batch.state))
    first_bundle = first.compute_loss_bundle(batch)
    second_bundle = second.compute_loss_bundle(batch)
    assert torch.allclose(first_bundle.total, second_bundle.total)
    for name in first_bundle.terms:
        assert torch.allclose(first_bundle.terms[name], second_bundle.terms[name])


def test_config_rejects_wrong_algorithm_and_invalid_shapes() -> None:
    config = replace(MAPPOConfig(), algorithm="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="不匹配"):
        MAPPO(environment_spec(), config)

    algorithm = MAPPO(environment_spec(), MAPPOConfig())
    with pytest.raises(ValueError, match="action_mask"):
        algorithm.sample(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 6),
            action_mask=torch.ones(1, 2, 3, dtype=torch.bool),
        )
    batch = sample_batch()
    batch.state = torch.zeros(5, 5)
    with pytest.raises(ValueError):
        algorithm.compute_loss_bundle(batch)


class TinyAdapter(EnvironmentAdapter):
    def __init__(self) -> None:
        self._spec = EnvironmentSpec(
            num_agents=2,
            observation_dim=3,
            action_dim=2,
            state_dim=6,
            action_kind=ActionKind.DISCRETE,
            horizon=3,
            reward_structure=RewardStructure.INDIVIDUAL,
        )
        self.time = 0

    @property
    def spec(self) -> EnvironmentSpec:
        return self._spec

    def _step(self, rewards: np.ndarray, terminated: bool) -> EnvironmentStep:
        observations = np.full((2, 3), self.time, dtype=np.float32)
        return self.validate_step(
            EnvironmentStep(
                observations=observations,
                state=observations.reshape(-1),
                rewards=rewards,
                terminated=terminated,
                truncated=False,
                action_mask=np.ones((2, 2), dtype=bool),
            )
        )

    def reset(self, seed: int | None = None) -> EnvironmentStep:
        del seed
        self.time = 0
        return self._step(np.zeros(2, dtype=np.float32), False)

    def step(self, actions: np.ndarray) -> EnvironmentStep:
        if actions.shape != (2,):
            raise ValueError("actions 必须是 [N]")
        self.time += 1
        rewards = np.asarray([1.0 + actions[0], 2.0 + actions[1]], dtype=np.float32)
        return self._step(rewards, self.time >= self.spec.horizon)


def test_on_policy_trainer_runs_end_to_end_and_restores_checkpoint(
    tmp_path: Path,
) -> None:
    environment = SyncVectorEnv([TinyAdapter(), TinyAdapter()])
    config = training_config()
    experiment = build_experiment(environment, config, seed=8)
    trainer = experiment.trainer
    algorithm = experiment.algorithm
    before = [parameter.detach().clone() for parameter in algorithm.parameters()]
    metrics = trainer.train_rollout([1, 2])
    after = list(algorithm.parameters())

    assert any(
        not torch.equal(left, right.detach()) for left, right in zip(before, after, strict=True)
    )
    assert {"policy_loss", "value_loss", "entropy", "approx_kl"} <= metrics.keys()
    assert trainer.update_plan.update_count == 1

    checkpoint = tmp_path / "trainer.pt"
    trainer.save_checkpoint(checkpoint)
    saved_parameters = [parameter.detach().clone() for parameter in algorithm.parameters()]
    with torch.no_grad():
        for parameter in algorithm.parameters():
            parameter.add_(10.0)
    restored = build_experiment(environment, config, seed=8).trainer
    restored_algorithm = restored.algorithm
    restored.load_checkpoint(checkpoint)
    assert restored.update_plan.update_count == 1
    assert restored.update_plan.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert all(
        torch.equal(expected, actual.detach())
        for expected, actual in zip(saved_parameters, restored_algorithm.parameters(), strict=True)
    )

    mismatched = replace(config, update=PPOUpdateConfig(epochs=1))
    other_trainer = build_experiment(environment, mismatched, seed=8).trainer
    with pytest.raises(ValueError, match="config"):
        other_trainer.load_checkpoint(checkpoint)
