"""Fixed-input migration regression: both construction paths must be identical."""
import json
from pathlib import Path

import pytest
import torch

from marl.algorithms.maac import MAACConfig
from marl.algorithms.maddpg import MADDPGConfig
from marl.algorithms.mappo import MAPPOConfig
from marl.algorithms.masac import MASACConfig
from marl.algorithms.qmix import QMIXConfig
from marl.core import MARLBatch
from marl.envs import ActionKind, EnvironmentSpec, RewardStructure
from marl.modules import centralized_critic_input

CASES = [MAPPOConfig(), MAACConfig(), MADDPGConfig(), MASACConfig(), QMIXConfig()]


def migration_batch(config):
    continuous = config.algorithm in ("maddpg", "masac")
    spec = EnvironmentSpec(
        2, 3, 2 if continuous else 4, 6,
        ActionKind.CONTINUOUS if continuous else ActionKind.DISCRETE,
        4, RewardStructure.SHARED if config.algorithm == "qmix" else RewardStructure.INDIVIDUAL,
    )
    torch.manual_seed(13)
    rewards = torch.randn(3, 2)
    if config.algorithm == "qmix":
        rewards = rewards[:, :1].expand(-1, 2).clone()
    flags = torch.zeros(3, 2, dtype=torch.bool)
    flags[0, 0] = True
    batch = MARLBatch(
        observations=torch.randn(3, 2, 3),
        actions=torch.randn(3, 2, 2).tanh() if continuous else torch.randint(0, 4, (3, 2)),
        rewards=rewards, next_observations=torch.randn(3, 2, 3),
        state=torch.randn(3, 6), next_state=torch.randn(3, 6),
        terminated=flags, truncated=torch.zeros_like(flags),
        action_mask=None if continuous else torch.ones(3, 2, 4, dtype=torch.bool),
        extras={"old_log_prob": torch.zeros(3, 2),
                "advantages": torch.randn(3, 2), "returns": torch.randn(3, 2)},
    )
    if config.algorithm == "qmix":
        batch.terminated = torch.zeros_like(flags)
    return spec, batch


@pytest.mark.parametrize("config", CASES)
def test_matches_pre_migration_numerical_fixture(config):
    # Captured before removing the recipe implementation (commit 0f6fc53).
    expected = json.loads(
        (Path(__file__).parent / "fixtures" / "algorithm_baseline.json").read_text()
    )[config.algorithm]
    spec, batch = migration_batch(config)
    torch.manual_seed(91)
    model = config.build(spec)
    torch.manual_seed(21)
    actions = model.act(batch.observations, action_mask=batch.action_mask)
    torch.testing.assert_close(actions, torch.tensor(expected["actions"]))
    torch.manual_seed(31)
    bundle = model.compute_loss_bundle(batch)
    if config.algorithm == "masac":
        # 旧 fixture 使用有 epsilon 偏差的 tanh Jacobian；数学修复不是等价迁移。
        # 动作与 Q 仍严格对照旧 fixture，loss 单独保存修复后的确定性回归值。
        expected["loss"] = -2.5014781951904297
        expected["terms"].update(
            loss=-2.5014781951904297, critic_loss=3.091871738433838,
            actor_loss=-0.20708514750003815, alpha_loss=-5.386264801025391,
        )
    assert bundle.total.item() == pytest.approx(expected["loss"], rel=1e-6, abs=1e-6)
    for key, value in expected["terms"].items():
        assert bundle.terms[key].item() == pytest.approx(value, rel=1e-6, abs=1e-6)
    if config.algorithm == "mappo":
        values = model.values(batch.observations, batch.state)
    elif config.algorithm == "maac":
        values = model.critic(batch.observations, batch.actions)
    elif config.algorithm == "qmix":
        values = model.policy.q_values(batch.observations)
    else:
        values = model.critics(centralized_critic_input(batch.observations, batch.actions))
    if isinstance(values, tuple):
        values = torch.stack(values)
    torch.testing.assert_close(values, torch.tensor(expected["values"]), rtol=1e-6, atol=1e-6)
