"""Fixed-input migration regression: both construction paths must be identical."""
import pytest
import torch

from marl.algorithms import MAAC, MADDPG, MAPPO, MASAC, QMIX
from marl.algorithms.maac import MAACConfig, default_maac_recipe
from marl.algorithms.maddpg import MADDPGConfig, default_maddpg_recipe
from marl.algorithms.mappo import MAPPOConfig, default_mappo_recipe
from marl.algorithms.masac import MASACConfig, default_masac_recipe
from marl.algorithms.qmix import QMIXConfig, default_qmix_recipe
from marl.core import MARLBatch
from marl.envs import ActionKind, EnvironmentSpec, RewardStructure

CASES = [
    (MAPPOConfig(), MAPPO, default_mappo_recipe),
    (MAACConfig(), MAAC, default_maac_recipe),
    (MADDPGConfig(), MADDPG, default_maddpg_recipe),
    (MASACConfig(), MASAC, default_masac_recipe),
    (QMIXConfig(), QMIX, default_qmix_recipe),
]


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


@pytest.mark.parametrize("config,algorithm,recipe", CASES)
def test_direct_build_matches_legacy(config, algorithm, recipe):
    spec, batch = migration_batch(config)
    torch.manual_seed(91)
    old = algorithm.from_recipe(spec, recipe())
    torch.manual_seed(91)
    new = config.build(spec)
    assert old.state_dict().keys() == new.state_dict().keys()
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
    torch.manual_seed(21)
    old_actions = old.act(batch.observations, action_mask=batch.action_mask)
    torch.manual_seed(21)
    torch.testing.assert_close(old_actions, new.act(
        batch.observations, action_mask=batch.action_mask,
    ), rtol=0, atol=0)
    torch.manual_seed(31)
    old_loss = old.compute_loss_bundle(batch)
    torch.manual_seed(31)
    new_loss = new.compute_loss_bundle(batch)
    torch.testing.assert_close(old_loss.total, new_loss.total, rtol=0, atol=0)
    for key, value in old_loss.terms.items():
        torch.testing.assert_close(value, new_loss.terms[key], rtol=0, atol=0)
