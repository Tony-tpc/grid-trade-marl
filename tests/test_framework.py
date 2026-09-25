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
from marl.models import GNNBackbone, GRUBackbone, MLPBackbone, TransformerBackbone
from marl.modules import Actor, DiscreteActionHead, QMixer, VDNMixer


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

    sequence = torch.randn(4, 3, 5, 6)  # [batch, agent, time, observation]
    gru_output = GRUBackbone(6, hidden_dim=16)(sequence)
    assert gru_output.features.shape == (4, 3, 5, 16)
    assert gru_output.hidden_state is not None


def test_actor_and_mask() -> None:
    actor = Actor(MLPBackbone(6, output_dim=16), DiscreteActionHead(16, 4))
    observations = torch.randn(8, 3, 6)
    mask = torch.tensor([True, False, False, False]).expand(8, 3, 4)
    result = actor(observations, deterministic=True, action_mask=mask)
    assert result.actions.shape == (8, 3)
    assert torch.equal(result.actions, torch.zeros_like(result.actions))
    assert torch.equal(
        actor.discrete_logits(observations, action_mask=mask),
        result.distribution_params["logits"],
    )


def test_mixers() -> None:
    agent_q = torch.randn(7, 3)
    state = torch.randn(7, 10)
    assert VDNMixer()(agent_q).shape == (7, 1)
    assert QMixer(3, 10)(agent_q, state).shape == (7, 1)


def test_concrete_algorithms_inherit_base_and_compute_loss() -> None:
    batch_size, agents, obs_dim = 4, 2, 3
    observations = torch.randn(batch_size, agents, obs_dim)
    next_observations = torch.randn_like(observations)
    rewards = torch.randn(batch_size, agents)
    dones = torch.zeros(batch_size, agents)

    maac = MAAC(MAACConfig(agents, obs_dim, 4, hidden_dim=16))
    maac_batch = MARLBatch(
        observations,
        torch.randint(0, 4, (batch_size, agents)),
        rewards,
        next_observations,
        dones,
    )
    assert maac.compute_loss(maac_batch)["loss"].ndim == 0

    continuous_actions = torch.randn(batch_size, agents, 2).tanh()
    maddpg = MADDPG(MADDPGConfig(agents, obs_dim, 2, hidden_dim=16))
    continuous_batch = MARLBatch(
        observations,
        continuous_actions,
        rewards,
        next_observations,
        dones,
    )
    assert maddpg.compute_loss(continuous_batch)["loss"].ndim == 0

    mappo = MAPPO(MAPPOConfig(agents, obs_dim, 4, state_dim=6, hidden_dim=16))
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
    assert mappo.compute_loss(mappo_batch)["loss"].ndim == 0

    masac = MASAC(MASACConfig(agents, obs_dim, 2, hidden_dim=16))
    assert masac.compute_loss(continuous_batch)["loss"].ndim == 0

    qmix = QMIX(QMIXConfig(agents, obs_dim, 4, state_dim=6, hidden_dim=16))
    qmix_batch = MARLBatch(
        observations=observations,
        actions=torch.randint(0, 4, (batch_size, agents)),
        rewards=rewards[:, :1].expand(-1, agents),
        next_observations=next_observations,
        dones=dones,
        state=torch.randn(batch_size, 6),
        next_state=torch.randn(batch_size, 6),
    )
    assert qmix.compute_loss(qmix_batch)["loss"].ndim == 0

    for algorithm in (maac, maddpg, mappo, masac, qmix):
        assert isinstance(algorithm, BaseMARLAlgorithm)
