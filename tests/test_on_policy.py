from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from marl.algorithms import MAPPO, MAPPOConfig
from marl.envs import ActionKind, EnvironmentSpec, RewardStructure
from marl.returns import GAEEstimator
from marl.training import OptimizerRuntime, PreparedRollout, RolloutBuffer
from marl.training.on_policy import PPOUpdateConfig


def spec() -> EnvironmentSpec:
    return EnvironmentSpec(
        num_agents=2,
        observation_dim=3,
        action_dim=4,
        state_dim=6,
        action_kind=ActionKind.DISCRETE,
        horizon=3,
        reward_structure=RewardStructure.INDIVIDUAL,
    )


def filled_rollout() -> RolloutBuffer:
    buffer = RolloutBuffer(spec(), horizon=3, num_envs=2)
    mask = torch.ones(2, 2, 4, dtype=torch.bool)
    for step in range(3):
        observations = torch.full((2, 2, 3), float(step * 2))
        observations[:, :, 0] += torch.arange(2).view(2, 1)
        buffer.add(
            observations=observations,
            states=torch.full((2, 6), float(step)),
            actions=torch.full((2, 2), step % 4),
            rewards=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            old_log_prob=torch.full((2, 2), -0.5),
            old_values=torch.full((2, 2), 0.25),
            terminated=torch.zeros(2, 2, dtype=torch.bool),
            truncated=torch.zeros(2, 2, dtype=torch.bool),
            action_masks=mask,
        )
    return buffer


def prepare() -> PreparedRollout:
    return filled_rollout().finish(
        torch.zeros(2, 2), GAEEstimator(gamma=0.9, gae_lambda=0.95, normalize=False)
    )


def test_rollout_finishes_with_expected_shapes_and_detached_old_values() -> None:
    buffer = filled_rollout()
    source = torch.full((2, 2), 9.0, requires_grad=True)
    # 已存张量来自 add 时的 copy，不应与外部 source 或计算图共享。
    assert not buffer.old_log_prob.requires_grad
    source.data.zero_()
    rollout = buffer.finish(
        torch.zeros(2, 2), GAEEstimator(gamma=0.9, normalize=False)
    )
    batch = rollout.batch
    assert batch.observations.shape == (6, 2, 3)
    assert batch.actions is not None and batch.actions.shape == (6, 2)
    assert batch.action_mask is not None and batch.action_mask.shape == (6, 2, 4)
    assert batch.extras["old_log_prob"].shape == (6, 2)
    assert batch.extras["returns"].shape == (6, 2)
    with pytest.raises(RuntimeError, match="已经 finish"):
        buffer.finish(torch.zeros(2, 2), GAEEstimator())


def test_minibatches_cover_each_sample_once_per_epoch_and_are_single_use() -> None:
    rollout = prepare()
    generator = torch.Generator().manual_seed(123)
    batches = list(
        rollout.minibatches(epochs=2, mini_batch_size=2, generator=generator)
    )
    assert len(batches) == 6
    for epoch in range(2):
        observations = torch.cat(
            [batch.observations for batch in batches[epoch * 3 : (epoch + 1) * 3]]
        )
        identifiers = observations[:, 0, 0]
        assert sorted(identifiers.tolist()) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    with pytest.raises(RuntimeError, match="已消费"):
        list(rollout.minibatches(epochs=1, mini_batch_size=2))


def test_rollout_rejects_missing_masks_and_incomplete_finish() -> None:
    buffer = RolloutBuffer(spec(), horizon=1, num_envs=1)
    with pytest.raises(ValueError, match="必须提供 action_masks"):
        buffer.add(
            observations=torch.zeros(1, 2, 3),
            states=torch.zeros(1, 6),
            actions=torch.zeros(1, 2, dtype=torch.long),
            rewards=torch.zeros(1, 2),
            old_log_prob=torch.zeros(1, 2),
            old_values=torch.zeros(1, 2),
            terminated=torch.zeros(1, 2, dtype=torch.bool),
            truncated=torch.zeros(1, 2, dtype=torch.bool),
        )
    incomplete = RolloutBuffer(spec(), horizon=2, num_envs=1)
    with pytest.raises(RuntimeError, match="尚未填满"):
        incomplete.finish(torch.zeros(1, 2), GAEEstimator())


def test_mappo_update_reports_metrics_and_restores_optimizer_runtime() -> None:
    config = replace(
        MAPPOConfig(),
        update=PPOUpdateConfig(epochs=2, mini_batch_size=2, max_grad_norm=0.1),
    )
    algorithm = MAPPO(spec(), config)
    runtime = OptimizerRuntime(
        {
            "actor": torch.optim.Adam(algorithm.policy.parameters(), lr=0.01),
            "critic": torch.optim.Adam(algorithm.critic.parameters(), lr=0.02),
        },
        {"actor": 0.1, "critic": 0.1},
        generator=torch.Generator().manual_seed(5),
    )
    before = tuple(parameter.detach().clone() for parameter in algorithm.parameters())
    metrics = algorithm.update(prepare(), runtime)
    after = tuple(parameter.detach().clone() for parameter in algorithm.parameters())

    assert any(not torch.equal(left, right) for left, right in zip(before, after, strict=True))
    assert {
        "loss",
        "policy_loss",
        "value_loss",
        "actor_gradient_norm",
        "critic_gradient_norm",
        "explained_variance",
    } <= metrics.keys()
    assert runtime.update_count == 1
    assert runtime.optimizer_step_count == 12

    state = runtime.state_dict()
    restored = OptimizerRuntime(
        {
            "actor": torch.optim.Adam(algorithm.policy.parameters(), lr=0.5),
            "critic": torch.optim.Adam(algorithm.critic.parameters(), lr=0.6),
        },
        {"actor": 0.1, "critic": 0.1},
    )
    restored.load_state_dict(state)
    assert restored.update_count == 1
    assert restored.optimizer_step_count == 12
    assert restored.optimizer("actor").param_groups[0]["lr"] == pytest.approx(0.01)
    assert restored.optimizer("critic").param_groups[0]["lr"] == pytest.approx(0.02)

    with pytest.raises(ValueError, match="旧 schema"):
        restored.load_state_dict({"optimizer": {}, "update_count": 1})
