"""新入口、配置错误、外部组件和 checkpoint 的行为测试。"""
from dataclasses import dataclass, replace

import pytest
import torch
import yaml  # type: ignore[import-untyped]
from test_composable_mappo import TinyAdapter
from torch import nn

from marl.algorithms.maac import MAACConfig
from marl.algorithms.maddpg import MADDPGConfig
from marl.algorithms.mappo import MAPPOConfig
from marl.algorithms.masac import MASACConfig
from marl.algorithms.qmix import QMIXConfig
from marl.config import algorithm_config_from_dict, config_to_dict, load_algorithm_config
from marl.envs import ActionKind, EnvironmentSpec, RewardStructure
from marl.experiment import build_experiment
from marl.extensions import ExtensionCatalog
from marl.modules.policy import IndependentDiscretePolicy
from marl.runtime import SyncVectorEnv
from marl.training import OffPolicyTrainer, OnPolicyTrainer
from marl.training.on_policy import PPOUpdateConfig


@pytest.mark.parametrize("config", [
    MAPPOConfig(), MAACConfig(), MADDPGConfig(), MASACConfig(), QMIXConfig(),
])
def test_config_roundtrip_and_experiment(config, tmp_path):
    path = tmp_path / "algorithm.yaml"
    path.write_text(yaml.safe_dump(config_to_dict(config)), encoding="utf-8")
    loaded = load_algorithm_config(path)
    assert loaded == config
    if isinstance(config, MAPPOConfig):
        env = SyncVectorEnv([TinyAdapter()])
        run = build_experiment(env, loaded, seed=9)
        assert isinstance(run.trainer, OnPolicyTrainer)
        assert run.trainer.train_rollout([9])["gradient_norm"] >= 0
    else:
        continuous = config.algorithm in ("maddpg", "masac")
        spec = EnvironmentSpec(
            2, 3, 2 if continuous else 4, 6,
            ActionKind.CONTINUOUS if continuous else ActionKind.DISCRETE,
            3, RewardStructure.SHARED if config.algorithm == "qmix"
            else RewardStructure.INDIVIDUAL,
        )
        run = build_experiment(spec, loaded)
        assert isinstance(run.trainer, OffPolicyTrainer)
    assert run.trainer.config_data == config_to_dict(config)


@pytest.mark.parametrize("change", [
    {"schema_version": True}, {"schema_version": 1.1}, {"schema_version": 2},
    {"algorithm": "unknown"}, {"critic": {"hidden_dim": True}},
    {"critic": {"hidden_dim": -1}}, {"policy": {"observation_dim": 10}},
    {"policy": {"kind": "independent_gaussian"}},
    {"update": {"learning_rate": float("nan")}}, {"group": [1, 2]},
    {"target_update": {"kind": "none"}},
])
def test_yaml_rejects_invalid_config(change):
    data = {"schema_version": 1, "algorithm": "mappo", **change}
    with pytest.raises((ValueError, TypeError)):
        algorithm_config_from_dict(data)


def test_python_config_incompatible_environment():
    with pytest.raises(ValueError, match="动作"):
        MADDPGConfig().build(TinyAdapter().spec)
    with pytest.raises(ValueError, match="共享"):
        QMIXConfig().build(TinyAdapter().spec)


@dataclass(frozen=True)
class ExternalSettings:
    hidden_dim: int = 8


class ExternalPolicy(nn.Module):
    """独立 nn.Module，不继承内置策略类。"""
    def __init__(self, spec, settings):
        super().__init__()
        self.network = IndependentDiscretePolicy(
            spec.num_agents, spec.observation_dim, spec.action_dim, settings.hidden_dim,
        )

    def act(self, observations, *, deterministic=False, action_mask=None):
        return self.network.act(
            observations, deterministic=deterministic, action_mask=action_mask,
        )

    def logits(self, observations, action_mask=None):
        return self.network.logits(observations, action_mask)

    def evaluate(self, observations, actions, *, action_mask=None):
        return self.network.evaluate(observations, actions, action_mask=action_mask)


def test_catalog_builds_external_modules_once_and_restores(tmp_path):
    counts = {"policy": 0, "critic": 0}
    catalog = ExtensionCatalog()

    def parse(data):
        return ExternalSettings(**data)

    def policy(settings, spec):
        counts["policy"] += 1
        return ExternalPolicy(spec, settings)

    def critic(settings, spec):
        counts["critic"] += 1
        return nn.Sequential(nn.Linear(spec.state_dim, settings.hidden_dim),
                             nn.Tanh(), nn.Linear(settings.hidden_dim, spec.num_agents))

    for category, builder in (("policy", policy), ("critic", critic)):
        catalog.register(category, "custom", parse, builder,
                         action_kinds=frozenset({ActionKind.DISCRETE}))
    data = {"schema_version": 1, "algorithm": "mappo",
            "policy": {"kind": "custom", "hidden_dim": 8},
            "critic": {"kind": "custom", "hidden_dim": 8}}
    config = algorithm_config_from_dict(data, catalog=catalog)
    assert counts == {"policy": 0, "critic": 0}
    env = SyncVectorEnv([TinyAdapter()])
    experiment = build_experiment(env, config)
    assert counts == {"policy": 1, "critic": 1}
    assert isinstance(experiment.trainer, OnPolicyTrainer)
    experiment.trainer.train_rollout([1])
    checkpoint = tmp_path / "custom.pt"
    experiment.trainer.save_checkpoint(checkpoint)
    restored = build_experiment(env, algorithm_config_from_dict(
        config_to_dict(config), catalog=catalog,
    ))
    restored.trainer.load_checkpoint(checkpoint)
    assert restored.trainer.optimization.update_count == 1
    assert counts == {"policy": 2, "critic": 2}
    with pytest.raises(ValueError, match="已注册"):
        catalog.register("policy", "custom", parse, policy,
                         action_kinds=frozenset({ActionKind.DISCRETE}))


def test_new_checkpoint_rejects_different_config(tmp_path):
    env = SyncVectorEnv([TinyAdapter()])
    run = build_experiment(env, MAPPOConfig())
    checkpoint = tmp_path / "new.pt"
    run.trainer.save_checkpoint(checkpoint)
    changed = replace(MAPPOConfig(), update=PPOUpdateConfig(epochs=1))
    with pytest.raises(ValueError, match="config"):
        build_experiment(env, changed).trainer.load_checkpoint(checkpoint)


def test_on_policy_checkpoint_repeats_next_update_exactly(tmp_path):
    env = SyncVectorEnv([TinyAdapter()])
    config = replace(MAPPOConfig(), update=PPOUpdateConfig(epochs=2, mini_batch_size=2))
    run = build_experiment(env, config, seed=11)
    run.trainer.train_rollout([1])
    checkpoint = tmp_path / "resume.pt"
    run.trainer.save_checkpoint(checkpoint)
    expected = run.trainer.train_rollout([2])
    parameters = {k: v.clone() for k, v in run.algorithm.state_dict().items()}
    restored = build_experiment(SyncVectorEnv([TinyAdapter()]), config, seed=999)
    restored.trainer.load_checkpoint(checkpoint)
    actual = restored.trainer.train_rollout([2])
    assert actual == expected
    for key, value in restored.algorithm.state_dict().items():
        torch.testing.assert_close(value, parameters[key], rtol=0, atol=0)
