import numpy as np
import torch

from marl.algorithms import MAAC, MAACConfig
from marl.envs import (
    ActionKind,
    EnergyTradingAdapter,
    EnergyTradingConfig,
    EnergyTradingEnv,
    Transition,
    transitions_to_batch,
)
from marl.runtime import TensorReplayBuffer


def test_tensor_replay_matches_transition_batch_after_wraparound() -> None:
    adapter = EnergyTradingAdapter(
        EnergyTradingEnv(EnergyTradingConfig(num_agents=2)), ActionKind.DISCRETE
    )
    replay = TensorReplayBuffer(adapter.spec, capacity=3)
    current = adapter.reset(seed=7)
    transitions = []
    for _ in range(4):
        assert current.action_mask is not None
        actions = current.action_mask.argmax(axis=-1).astype(np.int64)
        next_step = adapter.step(actions)
        transition = Transition(current, actions, next_step)
        replay.add(transition)
        transitions.append(transition)
        current = next_step

    rng = np.random.default_rng(11)
    sample = replay.sample(3, rng)
    # Ring slots contain transitions 3, 1 and 2 after the fourth write.
    slots = [transitions[3], transitions[1], transitions[2]]
    indices = np.random.default_rng(11).choice(3, size=3, replace=False)
    expected = transitions_to_batch([slots[int(index)] for index in indices])
    for name in (
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "dones",
        "state",
        "next_state",
        "action_mask",
        "next_action_mask",
    ):
        assert torch.equal(getattr(sample, name), getattr(expected, name))
    assert torch.equal(sample.extras["terminated"], expected.extras["terminated"])
    assert torch.equal(sample.extras["truncated"], expected.extras["truncated"])


def test_optimize_async_metrics_updates_parameters_and_reports_metrics() -> None:
    algorithm = MAAC(MAACConfig(2, 3, 4, hidden_dim=16))
    optimizer = torch.optim.Adam(algorithm.parameters(), lr=1e-3)
    from marl.core import MARLBatch

    observations = torch.randn(4, 2, 3)
    batch = MARLBatch(
        observations=observations,
        actions=torch.randint(0, 4, (4, 2)),
        rewards=torch.randn(4, 2),
        next_observations=torch.randn_like(observations),
    )
    parameter = next(algorithm.actors[0].parameters())
    before = parameter.detach().clone()
    metrics = algorithm.optimize(batch, optimizer, sync_metrics=False)
    after = parameter.detach()
    assert not torch.equal(before, after)
    assert np.isfinite(algorithm.metrics_to_cpu(metrics)["loss"])
