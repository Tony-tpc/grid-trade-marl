import torch

from marl.algorithms import (
    MAAC,
    MADDPG,
    MAPPO,
    MASAC,
    QMIX,
    BaseMARLAlgorithm,
    MAACConfig,
    MADDPGConfig,
    MAPPOConfig,
    MASACConfig,
    QMIXConfig,
)
from marl.core import MARLBatch
from marl.envs import ActionKind, EnvironmentSpec, RewardStructure
from marl.models import (
    GNNBackbone,
    GRUBackbone,
    MLPBackbone,
    MLPBackboneConfig,
    TransformerBackbone,
)
from marl.modules import (
    Actor,
    DiscreteActionHead,
    IndependentDiscretePolicy,
    QMixer,
    VDNMixer,
)


def test_all_backbones_follow_shape_contract() -> None:
    observations = torch.randn(4, 3, 6)
    assert MLPBackbone(6, output_dim=16)(observations).features.shape == (4, 3, 16)
    assert TransformerBackbone(6, model_dim=16)(observations).features.shape == (4, 3, 16)
    adjacency = torch.ones(4, 3, 3)
    assert GNNBackbone(6, hidden_dim=16)(observations, adjacency=adjacency).features.shape == (
        4,
        3,
        16,
    )
    sequence = torch.randn(4, 3, 5, 6)
    gru_output = GRUBackbone(6, hidden_dim=16)(sequence)
    assert gru_output.features.shape == (4, 3, 5, 16)
    assert gru_output.hidden_state is not None


def test_actor_and_mask() -> None:
    actor = Actor(MLPBackbone(6, output_dim=16), DiscreteActionHead(16, 4))
    observations = torch.randn(8, 3, 6)
    mask = torch.tensor([True, False, False, False]).expand(8, 3, 4)
    result = actor(observations, deterministic=True, action_mask=mask)
    evaluated = actor.evaluate_actions(observations, result.actions, action_mask=mask)
    assert result.actions.shape == (8, 3)
    assert torch.equal(result.actions, torch.zeros_like(result.actions))
    assert torch.equal(evaluated.actions, result.actions)
    assert torch.allclose(evaluated.log_prob, result.log_prob)
    assert torch.allclose(
        evaluated.distribution_params["logits"],
        result.distribution_params["logits"],
    )


def test_independent_policy_only_stacks_actor_results() -> None:
    policy = IndependentDiscretePolicy(2, 3, 4, MLPBackboneConfig(output_dim=8))
    observations = torch.randn(5, 2, 3)
    mask = torch.ones(5, 2, 4, dtype=torch.bool)
    mask[..., 1, 3] = False

    sampled = policy.act(observations, deterministic=True, action_mask=mask)
    evaluated = policy.evaluate(observations, sampled.actions, action_mask=mask)

    assert sampled.actions.shape == (5, 2)
    assert sampled.log_prob is not None and sampled.log_prob.shape == (5, 2)
    assert sampled.entropy is not None and sampled.entropy.shape == (5, 2)
    assert sampled.logits is not None and sampled.logits.shape == (5, 2, 4)
    assert evaluated.log_prob is not None
    assert evaluated.logits is not None
    assert torch.equal(evaluated.actions, sampled.actions)
    assert torch.allclose(evaluated.log_prob, sampled.log_prob)
    assert torch.allclose(evaluated.logits, sampled.logits)


def test_mixers() -> None:
    agent_q = torch.randn(7, 3)
    state = torch.randn(7, 10)
    assert VDNMixer()(agent_q).shape == (7, 1)
    assert QMixer(3, 10)(agent_q, state).shape == (7, 1)


def test_all_algorithms_are_config_built_direct_base_subclasses() -> None:
    batch_size, agents, obs_dim = 4, 2, 3
    observations = torch.randn(batch_size, agents, obs_dim)
    next_observations = torch.randn_like(observations)
    rewards = torch.randn(batch_size, agents)
    flags = torch.zeros(batch_size, agents, dtype=torch.bool)
    discrete = EnvironmentSpec(
        agents, obs_dim, 4, 6, ActionKind.DISCRETE, 8, RewardStructure.INDIVIDUAL
    )
    continuous = EnvironmentSpec(
        agents, obs_dim, 2, 6, ActionKind.CONTINUOUS, 8, RewardStructure.INDIVIDUAL
    )
    shared = EnvironmentSpec(agents, obs_dim, 4, 6, ActionKind.DISCRETE, 8, RewardStructure.SHARED)
    maac = MAAC(discrete, MAACConfig())
    maac_batch = MARLBatch(
        observations=observations,
        actions=torch.randint(0, 4, (batch_size, agents)),
        rewards=rewards,
        next_observations=next_observations,
        terminated=flags,
        truncated=flags,
    )
    assert maac.compute_loss_bundle(maac_batch).total.ndim == 0
    continuous_batch = MARLBatch(
        observations=observations,
        actions=torch.randn(batch_size, agents, 2).tanh(),
        rewards=rewards,
        next_observations=next_observations,
        terminated=flags,
        truncated=flags,
    )
    maddpg = MADDPG(continuous, MADDPGConfig())
    masac = MASAC(continuous, MASACConfig())
    assert maddpg.compute_loss_bundle(continuous_batch).total.ndim == 0
    assert masac.compute_loss_bundle(continuous_batch).total.ndim == 0
    mappo = MAPPO(discrete, MAPPOConfig())
    mappo_batch = MARLBatch(
        observations=observations,
        actions=torch.randint(0, 4, (batch_size, agents)),
        state=torch.randn(batch_size, 6),
        extras={
            "old_log_prob": torch.zeros(batch_size, agents),
            "advantages": torch.randn(batch_size, agents),
            "returns": torch.randn(batch_size, agents),
        },
    )
    assert mappo.compute_loss_bundle(mappo_batch).total.ndim == 0
    qmix = QMIX(shared, QMIXConfig())
    qmix_batch = MARLBatch(
        observations=observations,
        actions=torch.randint(0, 4, (batch_size, agents)),
        rewards=rewards[:, :1].expand(-1, agents),
        next_observations=next_observations,
        terminated=flags,
        truncated=flags,
        state=torch.randn(batch_size, 6),
        next_state=torch.randn(batch_size, 6),
    )
    assert qmix.compute_loss_bundle(qmix_batch).total.ndim == 0
    for algorithm in (maac, maddpg, mappo, masac, qmix):
        assert isinstance(algorithm, BaseMARLAlgorithm)
