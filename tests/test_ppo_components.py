from __future__ import annotations

import torch
from torch.distributions import Categorical

from marl.algorithms import MAPPO, MAPPOConfig
from marl.core import MARLBatch
from marl.envs import ActionKind, EnvironmentSpec, RewardStructure
from marl.objectives import (
    EntropyObjective,
    LossBundle,
    PPOClipObjective,
    ValueMSEObjective,
)
from marl.returns import GAEEstimator, TD0Estimator


def test_ppo_clips_positive_and_negative_advantages() -> None:
    objective = PPOClipObjective(clip_ratio=0.2)
    old = torch.zeros(2)
    new = torch.log(torch.tensor([1.5, 0.5]))
    advantages = torch.tensor([2.0, -2.0])
    result = objective(new, old, advantages)

    expected = -torch.tensor([1.2 * 2.0, 0.8 * -2.0]).mean()
    assert torch.allclose(result.loss, expected)
    assert result.metrics["clip_fraction"].item() == 1.0


def test_loss_bundle_combines_value_and_entropy_coefficients() -> None:
    value = ValueMSEObjective(coefficient=0.5)(
        torch.tensor([[1.0, 2.0]]), torch.tensor([[0.0, 0.0]])
    )
    entropy = EntropyObjective(coefficient=0.1)(torch.tensor([[2.0, 4.0]]))
    bundle = LossBundle.combine((value, entropy))

    assert torch.allclose(bundle.terms["value_loss"], torch.tensor(2.5))
    assert torch.allclose(bundle.terms["entropy"], torch.tensor(3.0))
    assert torch.allclose(bundle.total, torch.tensor(0.95))


def test_gae_distinguishes_termination_from_time_limit_truncation() -> None:
    estimator = GAEEstimator(gamma=0.9, gae_lambda=1.0, normalize=False)
    rewards = torch.ones(1, 2)
    values = torch.full((1, 2), 0.5)
    next_value = torch.full((2,), 10.0)
    terminated = torch.tensor([[True, False]])
    truncated = torch.tensor([[False, True]])

    advantages, returns = estimator.estimate(
        rewards, values, next_value, terminated, truncated
    )
    assert torch.allclose(advantages, torch.tensor([[0.5, 9.5]]))
    assert torch.allclose(returns, torch.tensor([[1.0, 10.0]]))


def test_gae_keeps_individual_agent_returns_and_normalizes_advantage() -> None:
    estimator = GAEEstimator(gamma=0.0, gae_lambda=0.0, normalize=True)
    rewards = torch.tensor([[[1.0, 10.0]], [[3.0, 20.0]]])
    values = torch.zeros_like(rewards)
    flags = torch.zeros_like(rewards, dtype=torch.bool)

    advantages, returns = estimator.estimate(
        rewards, values, torch.zeros(1, 2), flags, flags
    )
    assert torch.equal(returns, rewards)
    assert torch.allclose(advantages.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(advantages.var(unbiased=False), torch.tensor(1.0), atol=1e-6)


def test_td0_bootstraps_truncation_but_not_termination() -> None:
    estimator = TD0Estimator(gamma=0.9)
    rewards = torch.ones(1, 2)
    next_values = torch.full((1, 2), 10.0)
    terminated = torch.tensor([[True, False]])
    assert torch.allclose(
        estimator.estimate(rewards, next_values, terminated),
        torch.tensor([[1.0, 10.0]]),
    )


def test_extracted_objectives_match_mappo_bundle() -> None:
    torch.manual_seed(7)
    spec = EnvironmentSpec(
        2, 3, 4, 6, ActionKind.DISCRETE, 8, RewardStructure.INDIVIDUAL
    )
    algorithm = MAPPO(spec, MAPPOConfig())
    observations = torch.randn(5, 2, 3)
    state = torch.randn(5, 6)
    actions = torch.randint(0, 4, (5, 2))
    old_log_prob = torch.randn(5, 2)
    advantages = torch.randn(5, 2)
    returns = torch.randn(5, 2)
    batch = MARLBatch(
        observations=observations,
        actions=actions,
        state=state,
        extras={
            "old_log_prob": old_log_prob,
            "advantages": advantages,
            "returns": returns,
        },
    )
    existing = algorithm.compute_loss_bundle(batch)

    distribution = Categorical(logits=algorithm.policy.logits(observations))
    values = algorithm.critic(state)
    bundle = LossBundle.combine(
        (
            PPOClipObjective(algorithm.policy_objective.clip_ratio)(
                distribution.log_prob(actions), old_log_prob, advantages
            ),
            ValueMSEObjective(algorithm.value_objective.coefficient)(values, returns),
            EntropyObjective(algorithm.entropy_objective.coefficient)(
                distribution.entropy()
            ),
        )
    )

    assert torch.allclose(bundle.total, existing.total)
    assert torch.allclose(bundle.terms["policy_loss"], existing.terms["policy_loss"])
    assert torch.allclose(bundle.terms["value_loss"], existing.terms["value_loss"])
    assert torch.allclose(bundle.terms["entropy"], existing.terms["entropy"])
