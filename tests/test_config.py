"""每个算法 YAML 的默认值与 Python 默认配置相同。"""
from pathlib import Path

import pytest

from marl.algorithms import MAACConfig, MADDPGConfig, MAPPOConfig, MASACConfig, QMIXConfig
from marl.config import load_algorithm_config


@pytest.mark.parametrize("config", [
    MAPPOConfig(), MAACConfig(), MADDPGConfig(), MASACConfig(), QMIXConfig(),
])
def test_documented_yaml_matches_python_defaults(config):
    path = Path(__file__).parents[1] / "examples/configs/algorithms" / (config.algorithm + ".yaml")
    assert load_algorithm_config(path) == config
