from dataclasses import fields
from pathlib import Path

import pytest

from marl.envs import (
    ActionKind,
    EnergyTradingConfig,
    build_energy_trading_adapter,
    energy_trading_config_from_environment,
    environment_config_from_dict,
    load_environment_config,
)


def test_environment_yaml_is_independent_and_builds_adapter() -> None:
    path = (
        Path(__file__).parents[1] / "examples" / "configs" / "environments" / "energy_trading.yaml"
    )
    selected = load_environment_config(path)
    config = energy_trading_config_from_environment(selected)
    adapter = build_energy_trading_adapter(selected)

    assert selected.environment == "energy_trading"
    assert selected.action_kind == ActionKind.DISCRETE
    assert config.num_agents == 3
    assert config.sa_cycle_kw == (1.5, 1.0)
    assert adapter.spec.num_agents == config.num_agents
    assert adapter.spec.horizon == config.horizon
    assert adapter.spec.action_kind == ActionKind.DISCRETE
    assert set(selected.options) == {item.name for item in fields(EnergyTradingConfig)}


def test_environment_config_rejects_unknown_fields_and_options() -> None:
    base: dict[str, object] = {
        "schema_version": 1,
        "environment": "energy_trading",
        "action_kind": "discrete",
        "options": {},
    }
    with pytest.raises(ValueError, match="未知字段"):
        environment_config_from_dict({**base, "algorithm": "mappo"})

    selected = environment_config_from_dict({**base, "options": {"learning_rate": 0.001}})
    with pytest.raises(ValueError, match="未知字段"):
        energy_trading_config_from_environment(selected)
