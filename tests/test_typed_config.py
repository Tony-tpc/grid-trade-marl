"""配置 build 与迁移前组件初始化的数值等价验证。"""
import pytest
import torch

from marl.envs import ActionKind, EnvironmentSpec, RewardStructure
from marl.modules.critic import CentralizedValueConfig
from marl.modules.policy import IndependentDiscreteConfig, IndependentDiscretePolicy
from marl.returns import GAEConfig, TD0Config
from marl.target_updates import HardTargetConfig, SoftTargetConfig


def test_config_build_preserves_policy_parameters() -> None:
    spec = EnvironmentSpec(2, 3, 4, 6, ActionKind.DISCRETE, 8, RewardStructure.INDIVIDUAL)
    torch.manual_seed(123)
    expected = IndependentDiscretePolicy(2, 3, 4, 16)
    torch.manual_seed(123)
    actual = IndependentDiscreteConfig(hidden_dim=16).build(spec)
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], value)
    assert CentralizedValueConfig(hidden_dim=16).build(spec)(
        torch.zeros(5, 6)
    ).shape == (5, 2)


def test_return_configs_keep_distinct_interfaces() -> None:
    rewards = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    flags = torch.zeros_like(rewards, dtype=torch.bool)
    values = torch.zeros_like(rewards)
    advantage, returns = GAEConfig(gamma=0.0, normalize=False).build().estimate(
        rewards, values, torch.zeros(2), flags, flags,
    )
    torch.testing.assert_close(advantage, rewards)
    torch.testing.assert_close(returns, rewards)
    torch.testing.assert_close(TD0Config(gamma=0.0).build().estimate(
        rewards, values, flags,
    ), rewards)


@pytest.mark.parametrize("make", [
    lambda: IndependentDiscreteConfig(hidden_dim=0),
    lambda: GAEConfig(gamma=1.1),
    lambda: TD0Config(gamma=-0.1),
    lambda: SoftTargetConfig(tau=0),
    lambda: HardTargetConfig(interval=0),
])
def test_config_rejects_invalid_ranges(make) -> None:
    with pytest.raises(ValueError):
        make()
