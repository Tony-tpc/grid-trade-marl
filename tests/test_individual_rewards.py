"""逐智能体奖励与共享团队奖励的语义回归。"""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from marl.algorithms import (
    MAAC,
    MADDPG,
    MAPPO,
    MASAC,
    QMIX,
    default_maac_recipe,
    default_maddpg_recipe,
    default_mappo_recipe,
    default_masac_recipe,
    default_qmix_recipe,
)
from marl.core import MARLBatch
from marl.envs import ActionKind, EnvironmentSpec, RewardStructure
from marl.recipes import ComponentRecipe


def spec(
    kind: ActionKind,
    reward: RewardStructure = RewardStructure.INDIVIDUAL,
) -> EnvironmentSpec:
    action_dim = 4 if kind == ActionKind.DISCRETE else 2
    return EnvironmentSpec(2, 3, action_dim, 6, kind, 8, reward)


def test_maddpg_and_masac_fit_individual_targets() -> None:
    torch.manual_seed(3)
    observations = torch.randn(4, 2, 3)
    actions = torch.randn(4, 2, 2).tanh()
    rewards = torch.tensor([[2.0, -1.0]]).expand(4, -1)
    flags = torch.zeros(4, 2, dtype=torch.bool)
    batch = MARLBatch(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=observations,
        terminated=flags,
        truncated=flags,
    )
    maddpg_recipe = replace(
        default_maddpg_recipe(), returns=ComponentRecipe("td0", {"gamma": 0.0})
    )
    maddpg = MADDPG.from_recipe(spec(ActionKind.CONTINUOUS), maddpg_recipe)
    assert maddpg.policy.actors[0] is not maddpg.policy.actors[1]
    q = maddpg.critics(maddpg._critic_input(observations, actions))
    assert torch.allclose(
        maddpg.compute_loss_bundle(batch).terms["critic_loss"],
        F.mse_loss(q, rewards),
    )
    masac_recipe = replace(
        default_masac_recipe(), returns=ComponentRecipe("td0", {"gamma": 0.0})
    )
    masac = MASAC.from_recipe(spec(ActionKind.CONTINUOUS), masac_recipe)
    q1, q2 = masac.critics(masac._critic_input(observations, actions))
    expected = F.mse_loss(q1, rewards) + F.mse_loss(q2, rewards)
    assert torch.allclose(
        masac.compute_loss_bundle(batch).terms["critic_loss"], expected
    )


def test_maac_counterfactual_value_ignores_own_replay_action() -> None:
    torch.manual_seed(4)
    algorithm = MAAC.from_recipe(spec(ActionKind.DISCRETE), default_maac_recipe())
    observations = torch.randn(5, 2, 3)
    actions = torch.zeros(5, 2, dtype=torch.long)
    changed = actions.clone()
    changed[:, 0] = 3
    q_before = algorithm.critic(observations, actions)
    q_after = algorithm.critic(observations, changed)
    assert torch.allclose(q_before[:, 0], q_after[:, 0])


def test_mappo_keeps_per_agent_returns() -> None:
    algorithm = MAPPO.from_recipe(spec(ActionKind.DISCRETE), default_mappo_recipe())
    returns = torch.tensor([[2.0, -1.0]]).expand(3, -1)
    batch = MARLBatch(
        observations=torch.randn(3, 2, 3),
        actions=torch.zeros(3, 2, dtype=torch.long),
        state=torch.randn(3, 6),
        extras={
            "old_log_prob": torch.zeros(3, 2),
            "advantages": torch.ones(3, 2),
            "returns": returns,
        },
    )
    assert algorithm.compute_loss_bundle(batch).terms["value_loss"].ndim == 0
    batch.extras["returns"] = returns.mean(dim=-1)
    with pytest.raises(ValueError, match="逐智能体"):
        algorithm.compute_loss_bundle(batch)


def test_qmix_rejects_general_sum_reward_batch() -> None:
    algorithm = QMIX.from_recipe(
        spec(ActionKind.DISCRETE, RewardStructure.SHARED),
        default_qmix_recipe(),
    )
    flags = torch.zeros(2, 2, dtype=torch.bool)
    batch = MARLBatch(
        observations=torch.randn(2, 2, 3),
        actions=torch.zeros(2, 2, dtype=torch.long),
        rewards=torch.tensor([[1.0, -1.0], [0.0, 2.0]]),
        next_observations=torch.randn(2, 2, 3),
        terminated=flags,
        truncated=flags,
        state=torch.randn(2, 6),
        next_state=torch.randn(2, 6),
    )
    with pytest.raises(ValueError, match="共享团队奖励"):
        algorithm.compute_loss_bundle(batch)
