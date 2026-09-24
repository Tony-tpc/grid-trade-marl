"""针对一般和博弈的关键回归测试：不允许把 r_i 偷偷变成团队平均。"""

import pytest
import torch
from torch.nn import functional as F

from marl.algorithms import (
    MAAC,
    MADDPG,
    MAPPO,
    MASAC,
    QMIX,
    MAACConfig,
    MADDPGConfig,
    MAPPOConfig,
    MASACConfig,
    QMIXConfig,
)
from marl.core import MARLBatch


def test_maddpg_and_masac_fit_individual_targets() -> None:
    torch.manual_seed(3)
    obs = torch.randn(4, 2, 3)
    actions = torch.randn(4, 2, 2).tanh()
    rewards = torch.tensor([[2.0, -1.0]]).expand(4, -1)
    batch = MARLBatch(obs, actions=actions, rewards=rewards, next_observations=obs)

    maddpg = MADDPG(MADDPGConfig(2, 3, 2, hidden_dim=16, gamma=0.0))
    assert maddpg.actors[0] is not maddpg.actors[1]
    current_input = maddpg._critic_input(obs, actions)
    q = torch.stack(
        [critic.forward(current_input).squeeze(-1) for critic in maddpg.critics], dim=-1
    )
    expected = F.mse_loss(q, rewards)
    assert torch.allclose(maddpg.compute_loss(batch)["critic_loss"], expected)

    masac = MASAC(MASACConfig(2, 3, 2, hidden_dim=16, gamma=0.0))
    assert masac.actors[0] is not masac.actors[1]
    current_input = masac._critic_input(obs, actions)
    q1 = torch.stack(
        [critic.forward(current_input).squeeze(-1) for critic in masac.critics1], dim=-1
    )
    q2 = torch.stack(
        [critic.forward(current_input).squeeze(-1) for critic in masac.critics2], dim=-1
    )
    expected = F.mse_loss(q1, rewards) + F.mse_loss(q2, rewards)
    assert torch.allclose(masac.compute_loss(batch)["critic_loss"], expected)


def test_maac_counterfactual_value_ignores_own_replay_action() -> None:
    torch.manual_seed(4)
    algorithm = MAAC(MAACConfig(2, 3, 4, hidden_dim=16, gamma=0.0))
    assert algorithm.actors[0] is not algorithm.actors[1]
    obs = torch.randn(5, 2, 3)
    actions = torch.zeros(5, 2, dtype=torch.long)
    changed_own_action = actions.clone()
    changed_own_action[:, 0] = 3

    # 只改变 a_0 时，Q_0 的全部候选值应保持不变，因为上下文只读 a_1。
    q_before = algorithm.critic(obs, actions)
    q_after = algorithm.critic(obs, changed_own_action)
    assert torch.allclose(q_before[:, 0], q_after[:, 0])

    rewards = torch.tensor([[1.0, -2.0]]).expand(5, -1)
    batch = MARLBatch(obs, actions=actions, rewards=rewards, next_observations=obs)
    chosen_q = q_before.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(
        algorithm.compute_loss(batch)["critic_loss"], F.mse_loss(chosen_q, rewards)
    )


def test_mappo_requires_and_fits_per_agent_returns() -> None:
    obs = torch.randn(3, 2, 3)
    algorithm = MAPPO(MAPPOConfig(2, 3, 4, state_dim=6, hidden_dim=16))
    state = torch.randn(3, 6)
    returns = torch.tensor([[2.0, -1.0]]).expand(3, -1)
    batch = MARLBatch(
        observations=obs,
        actions=torch.zeros(3, 2, dtype=torch.long),
        state=state,
        extras={
            "old_log_prob": torch.zeros(3, 2),
            "advantages": torch.ones(3, 2),
            "returns": returns,
        },
    )
    expected = F.mse_loss(algorithm.critic.forward(state), returns)
    assert torch.allclose(algorithm.compute_loss(batch)["value_loss"], expected)
    batch.extras["returns"] = returns.mean(dim=-1)
    with pytest.raises(ValueError, match="逐智能体"):
        algorithm.compute_loss(batch)


def test_qmix_rejects_general_sum_reward_batch() -> None:
    algorithm = QMIX(QMIXConfig(2, 3, 4, state_dim=6, hidden_dim=16))
    batch = MARLBatch(
        observations=torch.randn(2, 2, 3),
        actions=torch.zeros(2, 2, dtype=torch.long),
        rewards=torch.tensor([[1.0, -1.0], [0.0, 2.0]]),
        next_observations=torch.randn(2, 2, 3),
        state=torch.randn(2, 6),
        next_state=torch.randn(2, 6),
    )
    with pytest.raises(ValueError, match="共享团队奖励"):
        algorithm.compute_loss(batch)
