from __future__ import annotations

import torch
from torch import nn

from marl.algorithms import MAAC, MAACConfig
from marl.core import MARLBatch
from marl.envs import ActionKind, EnvironmentSpec, RewardStructure
from marl.experiment import build_experiment
from marl.modules import AttentionCritic
from marl.modules.policy import AgentLogitsPolicy
from marl.objectives import CounterfactualPolicyObjective


def _spec() -> EnvironmentSpec:
    return EnvironmentSpec(
        num_agents=2,
        observation_dim=3,
        action_dim=4,
        state_dim=6,
        action_kind=ActionKind.DISCRETE,
        horizon=8,
        reward_structure=RewardStructure.INDIVIDUAL,
    )


def _batch() -> MARLBatch:
    flags = torch.zeros(5, 2, dtype=torch.bool)
    return MARLBatch(
        observations=torch.randn(5, 2, 3),
        actions=torch.randint(0, 4, (5, 2)),
        rewards=torch.randn(5, 2),
        next_observations=torch.randn(5, 2, 3),
        terminated=flags,
        truncated=flags,
    )


def test_attention_critic_has_agent_specific_encoders_and_q_heads() -> None:
    critic = AttentionCritic(2, 3, 4, hidden_dim=8, attention_heads=2)

    assert critic.own_encoders[0] is not critic.own_encoders[1]
    assert critic.state_action_encoders[0] is not critic.state_action_encoders[1]
    assert critic.q_heads[0] is not critic.q_heads[1]

    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.zero_()
        first_head = critic.get_submodule("q_heads.0.2")
        second_head = critic.get_submodule("q_heads.1.2")
        assert isinstance(first_head, nn.Linear)
        assert isinstance(second_head, nn.Linear)
        first_head.bias.fill_(1.0)
        second_head.bias.fill_(-2.0)
    observations = torch.ones(3, 2, 3)
    actions = torch.zeros(3, 2, dtype=torch.long)
    values = critic(observations, actions)
    assert torch.equal(values[:, 0], torch.ones(3, 4))
    assert torch.equal(values[:, 1], torch.full((3, 4), -2.0))


def test_maac_actor_uses_current_policy_joint_actions_not_replay_actions() -> None:
    algorithm = MAAC(_spec(), MAACConfig())
    batch = _batch()
    changed = MARLBatch(
        observations=batch.observations,
        actions=(batch.actions + 1) % 4 if batch.actions is not None else None,
        rewards=batch.rewards,
        next_observations=batch.next_observations,
        terminated=batch.terminated,
        truncated=batch.truncated,
    )

    torch.manual_seed(23)
    original_loss = algorithm.compute_actor_loss_bundle(batch, 0).total
    torch.manual_seed(23)
    changed_loss = algorithm.compute_actor_loss_bundle(changed, 0).total

    torch.testing.assert_close(original_loss, changed_loss)


def test_maac_target_networks_are_frozen() -> None:
    algorithm = MAAC(_spec(), MAACConfig())

    assert not any(parameter.requires_grad for parameter in algorithm.target_policy.parameters())
    assert not any(parameter.requires_grad for parameter in algorithm.target_critic.parameters())
    assert all(parameter.grad is None for parameter in algorithm.target_policy.parameters())
    assert all(parameter.grad is None for parameter in algorithm.target_critic.parameters())


def test_counterfactual_baseline_matches_manual_small_tensor() -> None:
    logits = torch.tensor([[0.2, -0.4, 0.7]], requires_grad=True)
    q_values = torch.tensor([[2.0, -1.0, 0.5]])

    result = CounterfactualPolicyObjective()(logits, q_values)

    probabilities = logits.softmax(dim=-1).detach()
    baseline = (probabilities * q_values).sum(dim=-1, keepdim=True)
    advantage = q_values - baseline
    expected = -(probabilities * advantage * logits.log_softmax(dim=-1)).sum(dim=-1).mean()
    torch.testing.assert_close(result.loss, expected)


def test_attention_critic_validates_joint_shapes_early() -> None:
    critic = AttentionCritic(2, 3, 4, hidden_dim=8, attention_heads=2)

    try:
        critic(torch.zeros(2, 3, 3), torch.zeros(2, 3, dtype=torch.long))
    except ValueError as error:
        assert "observations" in str(error)
    else:
        raise AssertionError("错误智能体维应被拒绝")


def test_single_agent_logits_fast_path_matches_full_policy_logits() -> None:
    algorithm = MAAC(_spec(), MAACConfig())
    observations = torch.randn(5, 2, 3)
    action_mask = torch.ones(5, 2, 4, dtype=torch.bool)

    full = algorithm.policy.logits(observations, action_mask)
    assert isinstance(algorithm.policy, AgentLogitsPolicy)
    selected = torch.stack(
        [
            algorithm.policy.logits_for_agent(observations, index, action_mask)
            for index in range(2)
        ],
        dim=-2,
    )

    torch.testing.assert_close(selected, full)


def test_maac_update_reuses_one_actor_q_context() -> None:
    experiment = build_experiment(_spec(), MAACConfig())
    algorithm = experiment.algorithm
    assert isinstance(algorithm, MAAC)
    calls = 0

    def count_forward(_module, _inputs, _output) -> None:
        nonlocal calls
        calls += 1

    handle = algorithm.get_submodule("critic").register_forward_hook(count_forward)
    try:
        experiment.trainer.update_batch(_batch())
    finally:
        handle.remove()

    # 一次 replay critic loss + 一次全部 actor 共享的当前联合动作 Q。
    assert calls == 2
