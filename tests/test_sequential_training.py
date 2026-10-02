"""阶段冻结、新鲜版本、二阶更新及完整周期恢复。"""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from marl.algorithms import MAPPOConfig, SNMAPPOConfig
from marl.config import algorithm_config_from_dict, config_to_dict
from marl.envs import ActionKind, EnvironmentAdapter, EnvironmentSpec, EnvironmentStep
from marl.envs.sequential import SequentialSpec
from marl.experiment import build_experiment
from marl.models import GRUBackboneConfig, LSTMBackboneConfig, MLPBackboneConfig
from marl.modules.critic import CentralizedValueConfig
from marl.modules.policy import IndependentGaussianConfig
from marl.training.implicit import ImplicitResponseConfig, flat_gradient, implicit_response_gradient
from marl.training.on_policy import PPOUpdateConfig, PreparedRollout
from marl.training.optimization import OptimizerRuntime


class SequentialToy(EnvironmentAdapter):
    @property
    def checkpoint_context(self):
        return {'task': 'sequential-toy-v1'}

    @property
    def spec(self):
        return EnvironmentSpec(3, 3, 2, 9, ActionKind.CONTINUOUS, 4)

    @property
    def sequential_spec(self):
        role = replace(self.spec, num_agents=1)
        return SequentialSpec(self.spec, role, role,
                              replace(self.spec, num_agents=2, observation_dim=5))

    def reset(self, seed=None):
        self.time = 0
        self.observations = np.zeros((3, 3), dtype=np.float32)
        return self._view(np.zeros(3, dtype=np.float32))

    def _view(self, rewards):
        return self.validate_step(EnvironmentStep(self.observations.copy(),
                                                 self.observations.flatten(), rewards,
                                                 self.time == 4, False))

    def commit_leader(self, action):
        self.leader_action = action
        self.observations[1:, :2] = action
        return self._view(np.zeros(3, dtype=np.float32))

    def settle(self, actions):
        consumer = -np.square(actions-self.leader_action).sum(-1)
        leader = -np.square(actions.mean(0)-.5).sum()
        self.time += 1
        self.observations[:, 2] = self.time/4
        return self._view(np.r_[leader, consumer].astype(np.float32))

    def step(self, actions):
        self.commit_leader(actions[0])
        return self.settle(actions[1:])


def configuration(recurrent=False, implicit=False):
    network = GRUBackboneConfig(hidden_dim=4) if recurrent else MLPBackboneConfig(
        hidden_dims=(4,), output_dim=4)
    role = MAPPOConfig(
        policy=IndependentGaussianConfig(backbone=network),
        critic=CentralizedValueConfig(backbone=network),
        update=PPOUpdateConfig(epochs=2, mini_batch_size=4,
                               sequence_length=2 if recurrent else None),
    )
    return SNMAPPOConfig(leader=role, coordinator=role, followers=role, implicit=implicit,
                         response=ImplicitResponseConfig(damping=.1, max_iterations=100))


@pytest.mark.parametrize('recurrent', [False, True])
def test_stage_ownership_fresh_versions_and_checkpoint(recurrent):
    torch.set_num_threads(1)
    config = configuration(recurrent)
    assert algorithm_config_from_dict(config_to_dict(config)) == config
    experiment = build_experiment(SequentialToy(), config, seed=3)
    trainer, algorithm = experiment.trainer, experiment.algorithm
    ds = trainer.collect('leader', 2)
    stale = trainer.collect('leader', 3)
    before = deepcopy(algorithm.state_dict())
    algorithm.update(ds, trainer.runtimes)
    after = deepcopy(algorithm.state_dict())
    assert any(not torch.equal(v, before[k]) for k, v in after.items()
               if k.startswith('leader.policy.'))
    assert all(torch.equal(v, before[k]) for k, v in after.items()
               if k.startswith('followers.'))
    with pytest.raises(RuntimeError, match='阶段|版本'):
        algorithm.update(stale, trainer.runtimes)
    with pytest.raises(RuntimeError, match='边界'):
        trainer.state_dict()
    dn = trainer.collect('followers', 5)
    algorithm.update(dn, trainer.runtimes)
    assert all(torch.equal(v, after[k]) for k, v in algorithm.state_dict().items()
               if k.startswith(('leader.', 'coordinator.')))
    assert algorithm.versions == (1, 1, 1)
    checkpoint = trainer.state_dict()
    metrics = trainer.train_cycle()
    expected = deepcopy(algorithm.state_dict())
    trainer.load_state_dict(checkpoint)
    assert trainer.train_cycle() == metrics
    for key, value in algorithm.state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=0, rtol=0)


def test_implicit_end_to_end_and_config_errors():
    torch.set_num_threads(1)
    experiment = build_experiment(SequentialToy(), configuration(implicit=True), seed=7)
    metrics = experiment.trainer.train_cycle()
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics['leader/response_success'] == 1
    assert metrics['leader/response_residual'] < 1e-4
    assert metrics['physical_steps'] == 4*(experiment.config.response_trajectories+1)
    with pytest.raises(ValueError, match='未知'):
        algorithm_config_from_dict({'schema_version': 2, 'algorithm': 'sn_mappo', 'replay': {}})


def test_leader_optimizer_uses_complete_consistent_implicit_gradient(monkeypatch):
    import marl.algorithms.sn_mappo as module
    experiment = build_experiment(SequentialToy(), configuration(implicit=True), seed=7)
    algorithm, trainer = experiment.algorithm, experiment.trainer
    rollout = trainer.collect('leader', 1)
    # PPO advantage 的值不应进入 DS 目标；coordinator 的下降方向使用构造 H 的同一个 Lc。
    batches = (rollout.leader.batch, rollout.coordinator.batch, rollout.followers.batch)
    upper, lower = algorithm.response_objectives(batches)
    cp = tuple(algorithm.coordinator.policy.parameters())
    lp = tuple(algorithm.leader.policy.parameters())
    expected_lower = flat_gradient(lower.total, cp).detach()
    expected_upper = flat_gradient(upper.total, lp).detach()
    for batch in batches:
        batch.extras['advantages'].fill_(12345)
    again, _ = algorithm.response_objectives(batches)
    torch.testing.assert_close(flat_gradient(again.total, lp), expected_upper)
    before = torch.cat([p.detach().flatten() for p in lp]).clone()
    captured = []

    def capture(upper, lower, leader, coordinator, settings):
        torch.testing.assert_close(flat_gradient(lower, coordinator), expected_lower)
        result = implicit_response_gradient(upper, lower, leader, coordinator, settings)
        captured.append(result)
        return result
    monkeypatch.setattr(module, 'implicit_response_gradient', capture)
    runtime = OptimizerRuntime({
        'actor': torch.optim.SGD(lp, lr=.1),
        'critic': torch.optim.SGD(algorithm.leader.critic.parameters(), lr=.001),
    }, {'actor': None, 'critic': None})
    metrics = algorithm.update(rollout, (runtime, *trainer.runtimes[1:]))
    assert metrics['response_success'] == 1
    after = torch.cat([p.detach().flatten() for p in lp])
    torch.testing.assert_close(after, before-.1*captured[0].gradient)


@pytest.mark.parametrize('kind', ['mlp', 'gru', 'lstm'])
def test_multiple_episodes_preserve_histories_log_prob_and_consumption(kind):
    config = configuration(recurrent=kind != 'mlp')
    if kind == 'lstm':
        network = LSTMBackboneConfig(hidden_dim=4)
        role = replace(config.leader, policy=IndependentGaussianConfig(backbone=network),
                       critic=CentralizedValueConfig(backbone=network))
        config = replace(config, leader=role, coordinator=role, followers=role)
    experiment = build_experiment(SequentialToy(), config, seed=3)
    trainer, policy = experiment.trainer, experiment.algorithm.leader.policy
    episodes = [trainer._collect_episode('leader', 2).leader for _ in range(2)]
    logs = []
    for episode in episodes:
        batch = episode.batch
        output = policy.evaluate(batch.observations, batch.actions,
                                 hidden_state=batch.policy_state,
                                 raw_actions=batch.extras['raw_actions'])
        logs.append(output.log_prob)
    old_returns = torch.cat([e.batch.extras['returns'] for e in episodes])
    combined = PreparedRollout.concatenate(episodes)
    assert all(e.consumed for e in episodes)
    assert combined.size == 8
    batch = combined.batch
    output = policy.evaluate(batch.observations, batch.actions, hidden_state=batch.policy_state,
                             raw_actions=batch.extras['raw_actions'])
    torch.testing.assert_close(output.log_prob, torch.cat(logs))
    torch.testing.assert_close(batch.extras['returns'], old_returns)
    if kind == 'lstm':
        assert isinstance(combined.policy_history, tuple)
        for history in combined.policy_history:
            assert history.shape == (4, 1, 2, 1, 4)
            assert torch.count_nonzero(history[0]) == 0
        assert isinstance(combined.value_history, tuple)
        assert all(s.shape == (4, 1, 2, 4) for s in combined.value_history)
    with pytest.raises(RuntimeError, match='消费'):
        PreparedRollout.concatenate(episodes)
    with pytest.raises(ValueError, match='重复'):
        PreparedRollout.concatenate([combined, combined])
    assert not combined.consumed


def test_implicit_recurrent_checkpoint_continues_exactly_and_rejects_old_configuration():
    config = replace(configuration(recurrent=True, implicit=True), response_trajectories=3)
    experiment = build_experiment(SequentialToy(), config, seed=7)
    assert experiment.trainer.train_cycle()['leader/response_success'] == 1
    saved = experiment.trainer.state_dict()
    metrics = experiment.trainer.train_cycle()
    expected = deepcopy(experiment.algorithm.state_dict())
    experiment.trainer.load_state_dict(saved)
    assert experiment.trainer.train_cycle() == metrics
    for k, p in experiment.algorithm.state_dict().items():
        torch.testing.assert_close(p, expected[k], atol=0, rtol=0)
    old = deepcopy(saved)
    old['config'].pop('response_trajectories')
    with pytest.raises(ValueError, match='config'):
        experiment.trainer.load_state_dict(old)
