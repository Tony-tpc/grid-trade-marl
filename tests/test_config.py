"""每个算法 YAML 的默认值与 Python 默认配置相同。"""
from pathlib import Path

import pytest

from marl.algorithms import (
    MAACConfig,
    MADDPGConfig,
    MAPPOConfig,
    MASACConfig,
    QMIXConfig,
    SNMAPPOConfig,
)
from marl.config import load_algorithm_config


@pytest.mark.parametrize("config, relative_path", [
    (MAPPOConfig(), "examples/configs/algorithms/mappo.yaml"),
    (MAACConfig(), "examples/configs/algorithms/maac.yaml"),
    (MADDPGConfig(), "examples/configs/algorithms/maddpg.yaml"),
    (MASACConfig(), "examples/configs/algorithms/masac.yaml"),
    (QMIXConfig(), "examples/configs/algorithms/qmix.yaml"),
    (SNMAPPOConfig(), "reproduction/sn_mappo/configs/sn_mappo.yaml"),
])
def test_documented_yaml_matches_python_defaults(config, relative_path):
    path = Path(__file__).parents[1] / relative_path
    assert load_algorithm_config(path) == config
