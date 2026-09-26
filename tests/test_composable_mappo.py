from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from marl.algorithms import MAPPO
from marl.algorithms.mappo import default_mappo_recipe
from marl.core import MARLBatch
from marl.envs import (
    ActionKind,
    EnvironmentAdapter,
    EnvironmentSpec,
    EnvironmentStep,
    RewardStructure,
)
from marl.recipes import AlgorithmRecipe, ComponentRecipe
from marl.runtime import SyncVectorEnv
from marl.training import OnPolicyTrainer


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


def training_recipe() -> AlgorithmRecipe:
    recipe = default_mappo_recipe()
    return replace(
        recipe,
        policy=ComponentRecipe("independent_discrete", {"hidden_dim": 16}),
        critic=ComponentRecipe("centralized_value", {"hidden_dim": 16}),
        experience=ComponentRecipe("rollout", {"horizon": 3}),
        update=ComponentRecipe(
            "ppo_update",
            {
                "learning_rate": 1e-3,
                "epochs": 2,
                "mini_batch_size": 2,
                "max_grad_norm": 0.5,
            },
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


def test_recipe_construction_is_reproducible_with_fixed_seed() -> None:
    recipe = training_recipe()
    torch.manual_seed(41)
    first = MAPPO.from_recipe(environment_spec(), recipe)
    torch.manual_seed(41)
    second = MAPPO.from_recipe(environment_spec(), recipe)
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


def test_recipe_rejects_wrong_algorithm_and_invalid_shapes() -> None:
    recipe = replace(default_mappo_recipe(), algorithm="other")
    with pytest.raises(ValueError, match="不能从"):
        MAPPO.from_recipe(environment_spec(), recipe)

    algorithm = MAPPO.from_recipe(environment_spec(), default_mappo_recipe())
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
    recipe = training_recipe()
    algorithm = MAPPO.from_recipe(environment.spec, recipe)
    trainer = OnPolicyTrainer.from_recipe(
        environment,
        algorithm,
        recipe,
        generator=torch.Generator().manual_seed(8),
    )
    before = [parameter.detach().clone() for parameter in algorithm.parameters()]
    metrics = trainer.train_rollout([1, 2])
    after = list(algorithm.parameters())

    assert any(
        not torch.equal(left, right.detach())
        for left, right in zip(before, after, strict=True)
    )
    assert {"policy_loss", "value_loss", "entropy", "approx_kl"} <= metrics.keys()
    assert trainer.update_plan.update_count == 1

    checkpoint = tmp_path / "trainer.pt"
    trainer.save_checkpoint(checkpoint)
    saved_parameters = [parameter.detach().clone() for parameter in algorithm.parameters()]
    with torch.no_grad():
        for parameter in algorithm.parameters():
            parameter.add_(10.0)
    restored = OnPolicyTrainer.from_recipe(environment, algorithm, recipe)
    restored.load_checkpoint(checkpoint)
    assert restored.update_plan.update_count == 1
    assert restored.update_plan.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert all(
        torch.equal(expected, actual.detach())
        for expected, actual in zip(saved_parameters, algorithm.parameters(), strict=True)
    )

    mismatched = replace(recipe, update=ComponentRecipe("ppo_update", {"epochs": 1}))
    other = MAPPO.from_recipe(environment.spec, mismatched)
    other_trainer = OnPolicyTrainer.from_recipe(environment, other, mismatched)
    with pytest.raises(ValueError, match="recipe"):
        other_trainer.load_checkpoint(checkpoint)
