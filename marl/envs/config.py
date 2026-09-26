"""安全、不可变的环境 YAML config。

算法 config 只描述训练数学；环境 config 只描述环境动力学和动作编码。环境尺寸由
构建后的 ``EnvironmentAdapter.spec`` 推导，绝不能复制到算法 YAML 中。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias, cast

import yaml  # type: ignore[import-untyped]

from marl.envs.base import ActionKind
from marl.envs.energy_trading import EnergyProfiles, EnergyTradingConfig, EnergyTradingEnv
from marl.envs.energy_trading_adapter import EnergyTradingAdapter
from marl.envs.mpe2_simple_adversary_adapter import (
    MPE2SimpleAdversaryAdapter,
    MPE2SimpleAdversaryConfig,
    build_mpe2_simple_adversary,
)

EnvironmentScalar: TypeAlias = str | int | float | bool | None
EnvironmentValue: TypeAlias = (
    EnvironmentScalar
    | tuple["EnvironmentValue", ...]
    | Mapping[str, "EnvironmentValue"]
)


def _freeze(value: object) -> EnvironmentValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    raise TypeError(f"环境配置值不支持 {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """一份环境 YAML 的不可变表示。

    ``environment`` 选择环境实现，``action_kind`` 选择适配器向算法公开的动作编码，
    ``options`` 只包含环境物理、市场和设备参数。
    """

    schema_version: int
    environment: str
    action_kind: ActionKind
    options: Mapping[str, EnvironmentValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"不支持 environment schema_version={self.schema_version}；当前仅支持 1"
            )
        normalized = self.environment.strip().lower()
        if not normalized:
            raise ValueError("environment 不能为空")
        object.__setattr__(self, "environment", normalized)
        object.__setattr__(self, "action_kind", ActionKind(self.action_kind))
        object.__setattr__(
            self,
            "options",
            cast(Mapping[str, EnvironmentValue], _freeze(self.options)),
        )


_TOP_LEVEL_FIELDS = frozenset(
    {"schema_version", "environment", "action_kind", "options"}
)


def environment_config_from_dict(data: Mapping[str, object]) -> EnvironmentConfig:
    """严格解析环境 config；未知或缺失的顶层字段立即报错。"""

    unknown = set(data) - _TOP_LEVEL_FIELDS
    missing = _TOP_LEVEL_FIELDS - set(data)
    if unknown:
        raise ValueError(f"environment config 含未知字段：{sorted(unknown)}")
    if missing:
        raise ValueError(f"environment config 缺少字段：{sorted(missing)}")
    options = data["options"]
    if not isinstance(options, Mapping):
        raise TypeError("environment config options 必须是 mapping")
    return EnvironmentConfig(
        schema_version=int(cast(int, data["schema_version"])),
        environment=str(data["environment"]),
        action_kind=ActionKind(str(data["action_kind"])),
        options=cast(Mapping[str, EnvironmentValue], options),
    )


def load_environment_config(path: str | Path) -> EnvironmentConfig:
    """使用 ``yaml.safe_load`` 读取环境配置，不执行任何动态 Python 导入。"""

    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, Mapping):
        raise TypeError("environment YAML 顶层必须是 mapping")
    return environment_config_from_dict(cast(Mapping[str, object], data))


_ENERGY_TRADING_FIELDS = frozenset(
    item.name for item in fields(EnergyTradingConfig)
)
_TUPLE_FIELDS = frozenset(
    {"sa_cycle_kw", "has_ev", "has_storage", "has_hvac", "has_sa"}
)


def energy_trading_config_from_environment(
    config: EnvironmentConfig,
) -> EnergyTradingConfig:
    """把通用环境 config 严格绑定为 ``EnergyTradingConfig``。"""

    if config.environment != "energy_trading":
        raise ValueError(
            "energy_trading_config_from_environment 只接受 environment='energy_trading'"
        )
    unknown = set(config.options) - _ENERGY_TRADING_FIELDS
    if unknown:
        raise ValueError(f"energy_trading options 含未知字段：{sorted(unknown)}")
    options: dict[str, object] = dict(config.options)
    for name in _TUPLE_FIELDS:
        value = options.get(name)
        if value is not None:
            if not isinstance(value, tuple):
                raise TypeError(f"{name} 必须是 YAML 列表或 null")
            options[name] = value
    return EnergyTradingConfig(**options)  # type: ignore[arg-type]


def build_energy_trading_adapter(
    config: EnvironmentConfig,
    *,
    profiles: EnergyProfiles | None = None,
) -> EnergyTradingAdapter:
    """根据独立环境 YAML 创建一个全新的环境和适配器实例。"""

    settings = energy_trading_config_from_environment(config)
    return EnergyTradingAdapter(
        EnergyTradingEnv(settings, profiles),
        config.action_kind,
    )


_MPE2_SIMPLE_ADVERSARY_FIELDS = frozenset(
    item.name for item in fields(MPE2SimpleAdversaryConfig)
)


def mpe2_simple_adversary_config_from_environment(
    config: EnvironmentConfig,
) -> MPE2SimpleAdversaryConfig:
    """严格绑定 Farama MPE2 simple_adversary 的场景参数。"""

    if config.environment != "mpe2_simple_adversary":
        raise ValueError(
            "该构造函数只接受 environment='mpe2_simple_adversary'"
        )
    unknown = set(config.options) - _MPE2_SIMPLE_ADVERSARY_FIELDS
    if unknown:
        raise ValueError(f"MPE2 simple_adversary options 含未知字段：{sorted(unknown)}")
    return MPE2SimpleAdversaryConfig(**dict(config.options))  # type: ignore[arg-type]


def build_mpe2_simple_adversary_adapter(
    config: EnvironmentConfig,
) -> MPE2SimpleAdversaryAdapter:
    """根据独立环境 YAML 创建 MPE2 对抗场景适配器。"""

    settings = mpe2_simple_adversary_config_from_environment(config)
    return build_mpe2_simple_adversary(settings, action_kind=config.action_kind)
