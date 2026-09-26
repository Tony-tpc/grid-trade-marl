"""通过可选 ExtensionCatalog 接入库外 critic，并完成一次 MAPPO 更新。

内置组件不需要此目录。Python-only 用法可直接 MAPPOConfig(critic=TanhValueConfig())。
此例从字典模拟 YAML 数据；文件加载时把同一个 catalog 传给 load_algorithm_config。
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from torch import nn

from marl.algorithms import MAPPOConfig
from marl.config import algorithm_config_from_dict
from marl.envs import (
    ActionKind,
    EnvironmentSpec,
    build_energy_trading_adapter,
    load_environment_config,
)
from marl.experiment import build_experiment
from marl.extensions import ExtensionCatalog


@dataclass(frozen=True)
class TanhValueConfig:
    """输入全局 state [*B,S]，输出各 agent value [*B,N]。"""

    hidden_dim: int = 32
    kind: str = "tanh_value"

    def __post_init__(self) -> None:
        if type(self.hidden_dim) is not int or self.hidden_dim < 1:
            raise ValueError("hidden_dim 必须是正整数")

    def build(self, spec: EnvironmentSpec) -> nn.Sequential:
        if spec.state_dim is None:
            raise ValueError("tanh_value 需要全局 state")
        return nn.Sequential(
            nn.Linear(spec.state_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, spec.num_agents),
        )


def read_settings(data: Mapping[str, object]) -> TanhValueConfig:
    """外部组件负责自己的字段校验；不编写通用 schema 框架。"""
    unknown = set(data) - {"hidden_dim"}
    if unknown:
        raise ValueError(f"tanh_value 未知字段: {sorted(unknown)}")
    hidden_dim = data.get("hidden_dim", 32)
    if not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool):
        raise TypeError("hidden_dim 必须是整数")
    return TanhValueConfig(hidden_dim=hidden_dim)


def main() -> None:
    catalog = ExtensionCatalog()
    catalog.register(
        "critic",
        "tanh_value",
        parse=read_settings,
        build=TanhValueConfig.build,
        action_kinds=frozenset({ActionKind.DISCRETE}),
    )
    config = algorithm_config_from_dict(
        {
            "schema_version": 1,
            "algorithm": "mappo",
            "critic": {"kind": "tanh_value", "hidden_dim": 32},
            "rollout": {"horizon": 4},
            "update": {"epochs": 1, "mini_batch_size": 4},
        },
        catalog=catalog,
    )
    assert isinstance(config, MAPPOConfig)
    env_path = Path(__file__).parent / "configs/environments/energy_trading.yaml"
    env = build_energy_trading_adapter(load_environment_config(env_path))
    experiment = build_experiment(env, config, device="cpu", seed=42)
    metrics = experiment.trainer.train_rollout([42])
    print(f"custom critic: loss={metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
