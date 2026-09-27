from dataclasses import replace

import numpy as np
import torch
from test_composable_mappo import TinyAdapter

from marl.algorithms import MAACConfig
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
from marl.runtime import SyncVectorEnv, TensorReplayBuffer
from marl.training.off_policy import ActorCriticUpdateConfig, OffPolicyCollector, ReplayConfig


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


def test_record_many_matches_sequential_add_across_wraparound() -> None:
    adapter = EnergyTradingAdapter(
        EnergyTradingEnv(EnergyTradingConfig(num_agents=2)), ActionKind.DISCRETE
    )
    current = adapter.reset(seed=13)
    transitions = []
    for _ in range(7):
        assert current.action_mask is not None
        actions = current.action_mask.argmax(axis=-1).astype(np.int64)
        following = adapter.step(actions)
        transitions.append(Transition(current, actions, following))
        current = following

    sequential = TensorReplayBuffer(adapter.spec, capacity=3)
    batched = TensorReplayBuffer(adapter.spec, capacity=3)
    for transition in transitions:
        sequential.add(transition)
    batched.record_many(transitions)

    sequential_state = sequential.state_dict()
    batched_state = batched.state_dict()
    assert sequential.position == batched.position
    assert len(sequential) == len(batched)
    for name in (
        "observations",
        "next_observations",
        "states",
        "next_states",
        "actions",
        "rewards",
        "terminated",
        "truncated",
        "action_masks",
        "next_action_masks",
    ):
        sequential_value = sequential_state[name]
        batched_value = batched_state[name]
        assert isinstance(sequential_value, torch.Tensor)
        assert isinstance(batched_value, torch.Tensor)
        assert torch.equal(sequential_value, batched_value)


class _CountingPolicy:
    def __init__(self, spec: EnvironmentSpec) -> None:
        self.spec = spec
        self.calls = 0

    def act(
        self,
        observations: torch.Tensor,
        *,
        deterministic: bool = False,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del deterministic, action_mask
        self.calls += 1
        return torch.zeros(observations.shape[:-1], dtype=torch.long)


def test_off_policy_collector_uses_one_actor_call_per_vector_step() -> None:
    environment = SyncVectorEnv([TinyAdapter(), TinyAdapter()])
    algorithm = _CountingPolicy(environment.spec)
    collector = OffPolicyCollector(environment, algorithm)

    batches = list(collector.rollout([3, 5]))

    assert len(batches) == environment.spec.horizon
    assert algorithm.calls == environment.spec.horizon
    assert all(len(transitions) == 2 for transitions in batches)


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
        update=ActorCriticUpdateConfig(learning_rate=1e-3, max_grad_norm=0.5),
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
    cached_actor_parameters = trainer.optimization.parameters("actor")
    assert cached_actor_parameters is trainer.optimization.parameters("actor")

    trainer.optimization.sync_metrics = False
    assert trainer.update_batch(batch) == {}
    deferred = trainer.optimization.flush_metrics()
    assert {"loss", "actor_gradient_norm", "critic_gradient_norm"} <= deferred.keys()
    assert trainer.optimization.flush_metrics() == {}

    assert trainer.update_batch(batch) == {}
    saved_runtime = trainer.optimization.state_dict()
    restored = build_experiment(spec, config, seed=9).trainer.optimization
    restored.load_state_dict(saved_runtime)
    assert restored.sync_metrics is False
    assert restored.flush_metrics() == trainer.optimization.flush_metrics()
