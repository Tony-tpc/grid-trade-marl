"""固定动作密度、饱和、循环状态、个体训练和完整恢复。"""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.distributions import Normal, TanhTransform, TransformedDistribution

from marl.algorithms import MAPPOConfig, MASACConfig
from marl.config import algorithm_config_from_dict, config_to_dict
from marl.envs import ActionKind, EnvironmentAdapter, EnvironmentSpec, EnvironmentStep
from marl.experiment import build_experiment
from marl.models import GRUBackboneConfig, LSTMBackboneConfig, MLPBackboneConfig
from marl.modules.action_head import GaussianActionHead
from marl.modules.critic import CentralizedValueConfig
from marl.modules.policy import IndependentGaussianConfig
from marl.training import PPOUpdateConfig


class ContinuousTarget(EnvironmentAdapter):
    """仅用于测试的双 agent 连续目标任务；不是论文数据。"""
    @property
    def spec(self):
        return EnvironmentSpec(2, 3, 2, 6, ActionKind.CONTINUOUS, 5)

    def reset(self, seed=None):
        self.time = 0
        self.observation = np.random.default_rng(seed).normal(size=(2, 3)).astype(np.float32)
        return self._step(np.zeros(2, dtype=np.float32))

    def _step(self, reward):
        return self.validate_step(EnvironmentStep(
            self.observation.copy(), self.observation.flatten(), reward,
            False, self.time == self.spec.horizon,
        ))

    def step(self, actions):
        self.time += 1
        target = np.array([[-.4, .2], [.3, -.5]], dtype=np.float32)
        return self._step(-np.square(actions - target).sum(-1))


def test_density_and_saturation_preserve_the_sample():
    torch.manual_seed(5)
    head = GaussianActionHead(3, 2).double()
    x = torch.randn(4, 3, dtype=torch.float64)
    sampled = head(x)
    params = sampled.distribution_params
    expected = TransformedDistribution(
        Normal(params['mean'], params['log_std'].exp()), [TanhTransform()]
    ).log_prob(sampled.actions).sum(-1)
    torch.testing.assert_close(sampled.log_prob, expected)
    evaluated = head.evaluate_actions(
        x, sampled.actions.detach(), raw_actions=params['raw_actions'].detach()
    )
    torch.testing.assert_close(evaluated.log_prob, sampled.log_prob)
    with torch.no_grad():
        head.mean_layer.weight.zero_()
        head.mean_layer.bias.fill_(30)
    sampled = head(x)
    assert (sampled.actions == 1).all()
    evaluated = head.evaluate_actions(
        x, sampled.actions.detach(), raw_actions=sampled.distribution_params['raw_actions'].detach()
    )
    torch.testing.assert_close(evaluated.log_prob, sampled.log_prob)
    assert torch.isfinite(evaluated.entropy).all()
    with pytest.raises(ValueError, match='raw_actions'):
        head.evaluate_actions(x, sampled.actions)
    with pytest.raises(ValueError, match='不一致'):
        head.evaluate_actions(x, torch.zeros_like(sampled.actions), raw_actions=torch.ones(4, 2))


@pytest.mark.parametrize('kind', ['mlp', 'gru', 'lstm'])
def test_continuous_rollout_update_and_exact_resume(kind):
    backbone = {
        'mlp': MLPBackboneConfig(hidden_dims=(8,), output_dim=8),
        'gru': GRUBackboneConfig(hidden_dim=8),
        'lstm': LSTMBackboneConfig(hidden_dim=8),
    }[kind]
    config = MAPPOConfig(
        policy=IndependentGaussianConfig(backbone=backbone),
        critic=CentralizedValueConfig(backbone=backbone),
        update=PPOUpdateConfig(epochs=2, mini_batch_size=6,
                               sequence_length=3 if kind != 'mlp' else None),
    )
    assert algorithm_config_from_dict(config_to_dict(config)) == config
    experiment = build_experiment(ContinuousTarget(), config, seed=4)
    trainer = experiment.trainer
    rollout = trainer.collect([8])
    batch = rollout.batch
    assert batch.actions is not None
    assert not batch.extras['raw_actions'].requires_grad
    assert batch.extras['raw_actions'].shape == batch.actions.shape
    out = experiment.algorithm.policy.evaluate(
        batch.observations, batch.actions, hidden_state=batch.policy_state,
        raw_actions=batch.extras['raw_actions'],
    )
    torch.testing.assert_close(out.log_prob, batch.extras['old_log_prob'], atol=1e-5, rtol=1e-5)
    old = {k: p.clone() for k, p in experiment.algorithm.named_parameters()}
    metrics = experiment.algorithm.update(rollout, trainer.optimization)
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(p, old[k]) for k, p in experiment.algorithm.named_parameters())
    with pytest.raises(RuntimeError, match='消费'):
        experiment.algorithm.update(rollout, trainer.optimization)
    checkpoint = deepcopy(trainer.state_dict())
    first = trainer.train_rollout([9])
    expected = deepcopy(experiment.algorithm.state_dict())
    trainer.load_state_dict(checkpoint)
    second = trainer.train_rollout([9])
    assert first == second
    for key, value in expected.items():
        torch.testing.assert_close(experiment.algorithm.state_dict()[key], value, rtol=0, atol=0)


def test_off_policy_rejects_streaming_continuous_networks():
    config = replace(MASACConfig(), policy=IndependentGaussianConfig(
        backbone=GRUBackboneConfig(hidden_dim=8)
    ))
    with pytest.raises(ValueError, match='离策略'):
        build_experiment(ContinuousTarget(), config)
