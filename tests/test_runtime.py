from dataclasses import replace

import numpy as np
import torch

from marl.algorithms import MAACConfig
from marl.algorithms.maac import MAACUpdateConfig
from marl.core import MARLBatch
from marl.envs import (
    ActionKind,
    EnergyTradingAdapter,
    EnergyTradingConfig,
    EnergyTradingEnv,
    EnvironmentSpec,
    RewardStructure,
    Transition,
    transitions_to_batch,
)
from marl.experiment import build_experiment
from marl.modules.critic import AttentionQConfig
from marl.modules.policy import IndependentDiscreteConfig
from marl.runtime import TensorReplayBuffer
from marl.training.off_policy import ReplayConfig


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
        "terminated",
        "truncated",
        "state",
        "next_state",
        "action_mask",
        "next_action_mask",
    ):
        assert torch.equal(getattr(sample, name), getattr(expected, name))


def test_off_policy_trainer_updates_parameters_and_reports_metrics() -> None:
    spec = EnvironmentSpec(
        2,
        3,
        4,
        6,
        ActionKind.DISCRETE,
        8,
        RewardStructure.INDIVIDUAL,
    )
    config = MAACConfig()
    config = replace(
        config,
        policy=IndependentDiscreteConfig(hidden_dim=16),
        critic=AttentionQConfig(hidden_dim=16, attention_heads=4),
        replay=ReplayConfig(capacity=8, batch_size=4),
        update=MAACUpdateConfig(learning_rate=1e-3, max_grad_norm=0.5),
    )
    experiment = build_experiment(spec, config, seed=8)
    trainer = experiment.trainer
    algorithm = experiment.algorithm
    observations = torch.randn(4, 2, 3)
    batch = MARLBatch(
        observations=observations,
        actions=torch.randint(0, 4, (4, 2)),
        rewards=torch.randn(4, 2),
        next_observations=torch.randn_like(observations),
        terminated=torch.zeros(4, 2, dtype=torch.bool),
        truncated=torch.zeros(4, 2, dtype=torch.bool),
    )
    parameter = next(algorithm.get_submodule("policy").parameters())
    before = parameter.detach().clone()
    metrics = trainer.update_batch(batch)
    after = parameter.detach()
    assert not torch.equal(before, after)
    assert np.isfinite(metrics["loss"])
