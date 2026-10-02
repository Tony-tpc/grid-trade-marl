"""安全 YAML 边界：显式选择算法配置，所有默认值定义在配置 dataclass。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypeVar, get_args, get_origin, get_type_hints

import torch
import yaml  # type: ignore[import-untyped]

from marl.algorithms.maac import MAACConfig, MAACLossConfig
from marl.algorithms.maddpg import MADDPGConfig, MADDPGLossConfig
from marl.algorithms.mappo import MAPPOConfig, MAPPOLossConfig
from marl.algorithms.masac import MASACConfig, MASACLossConfig
from marl.algorithms.qmix import QMIXConfig, QMIXLossConfig
from marl.algorithms.sn_mappo import SNMAPPOConfig
from marl.extensions import Buildable, Category, ExtensionCatalog, ExternalConfig
from marl.models import (
    BackboneConfig,
    EncoderConfig,
    GNNBackboneConfig,
    GRUBackboneConfig,
    HistoryEncoderConfig,
    IdentityEncoderConfig,
    LSTMBackboneConfig,
    MLPBackboneConfig,
    TransformerBackboneConfig,
)
from marl.modules.critic import (
    AttentionQConfig,
    AttentionQNetwork,
    CentralizedValueConfig,
    IndependentQConfig,
    QEnsemble,
    TwinQConfig,
    TwinQEnsemble,
    ValueNetwork,
)
from marl.modules.mixer import MixingNetwork, QMixerConfig
from marl.modules.policy import (
    DiscretePolicy,
    IndependentDeterministicConfig,
    IndependentDiscreteConfig,
    IndependentGaussianConfig,
    LocalQPolicy,
    PolicyTopology,
    SharedDiscreteQConfig,
    StochasticPolicy,
)
from marl.returns import GAEConfig, TD0Config
from marl.target_updates import HardTargetConfig, SoftTargetConfig
from marl.training.implicit import ImplicitResponseConfig
from marl.training.off_policy import (
    ActorCriticUpdateConfig,
    OffPolicyUpdateConfig,
    ReplayConfig,
    TemperatureActorCriticUpdateConfig,
)
from marl.training.on_policy import PPOUpdateConfig, RolloutConfig

AlgorithmConfig: TypeAlias = (
    MAPPOConfig | MAACConfig | MADDPGConfig | MASACConfig | QMIXConfig | SNMAPPOConfig
)
T = TypeVar("T")


def _mapping(data: object, location: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping) or not all(isinstance(k, str) for k in data):
        raise TypeError(f"{location} 必须是字符串键 mapping")
    return data


def _flat(cls: type[T], data: object, location: str) -> T:
    """只解析叶子配置；算法组合在下方显式列出。Any 仅限 YAML 边界。"""
    values = dict(_mapping(data, location))
    if cls is GAEConfig and "lambda" in values:
        if "gae_lambda" in values:
            raise ValueError(f"{location} 不能同时填写 lambda 和 gae_lambda")
        values["gae_lambda"] = values.pop("lambda")
    hints = get_type_hints(cls)
    unknown = set(values) - set(hints)
    if unknown:
        raise ValueError(f"{location} 含未知字段: {sorted(unknown)}")
    for name, value in values.items():
        hint = hints[name]
        allowed = get_args(hint) if get_origin(hint) is not None else (hint,)
        if get_origin(hint) is Literal:
            if not any(type(value) is type(v) and value == v for v in allowed):
                raise ValueError(f"{location}.{name} 必须是 {allowed}")
        elif hint == tuple[int, ...]:
            if not isinstance(value, (list, tuple)) or any(
                type(item) is not int or item < 1 for item in value
            ):
                raise TypeError(f"{location}.{name} 必须为正整数列表，允许空列表")
            values[name] = tuple(value)
        elif is_dataclass(value) and type(value) in allowed:
            pass  # 已由下方显式网络分支解析的冻结配置。
        elif hint is torch.dtype or torch.dtype in allowed:
            if value not in (None, "bf16"):
                raise ValueError(f"{location}.{name} 只支持 null 或 bf16")
            values[name] = torch.bfloat16 if value == "bf16" else None
        else:
            valid = (
                (value is None and type(None) in allowed)
                or (type(value) is bool and bool in allowed)
                or (type(value) is int and (int in allowed or float in allowed))
                or (type(value) is float and float in allowed)
                or (type(value) is str and str in allowed)
            )
            if not valid:
                raise TypeError(f"{location}.{name} 类型错误")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{location}.{name} 必须是有限值")
    return cls(**values)


def _backbone(data: object, location: str) -> BackboneConfig:
    values = _mapping(data, location)
    kind = values.get("kind", "mlp")
    if kind == "mlp":
        return _flat(MLPBackboneConfig, values, location)
    if kind == "gru":
        return _flat(GRUBackboneConfig, values, location)
    if kind == "lstm":
        return _flat(LSTMBackboneConfig, values, location)
    raise ValueError(f"{location}.kind 只支持 mlp/gru/lstm")


def _component(
    data: object,
    builtin: type[Buildable[T]],
    capability: type,
    category: Category,
    catalog: ExtensionCatalog | None,
) -> Buildable[T]:
    values = dict(_mapping(data, category))
    default = builtin()
    kind = values.get("kind", default.kind)
    if kind == default.kind:
        if unknown := set(values) - set(get_type_hints(builtin)):
            raise ValueError(
                f"{category} 含旧/未知字段 {sorted(unknown)}；"
                "hidden_dim/layer_norm 请迁移到 backbone.output_dim/layer_norm"
            )
        if "encoder" in values:
            values["encoder"] = _encoder(values["encoder"], f"{category}.encoder")
        if values.get("graph") is not None:
            values["graph"] = _flat(GNNBackboneConfig, values["graph"], f"{category}.graph")
        if "backbone" in values:
            values["backbone"] = (
                _backbone(values["backbone"], f"{category}.backbone")
                if builtin in (
                    IndependentDiscreteConfig, IndependentGaussianConfig, CentralizedValueConfig
                )
                else _flat(MLPBackboneConfig, values["backbone"], f"{category}.backbone")
            )
        for field in ("embedding", "value_backbone"):
            if field in values:
                values[field] = _flat(MLPBackboneConfig, values[field], f"{category}.{field}")
        return _flat(builtin, values, category)
    if not isinstance(kind, str) or catalog is None:
        raise ValueError(f"未知或不兼容的 {category} kind: {kind}")
    values.pop("kind")
    return catalog.configure(category, kind, values, capability)


def _encoder(data: object, location: str) -> EncoderConfig:
    """历史窗口尺寸只能来自 EnvironmentSpec；YAML 只声明网络超参数。"""
    values = dict(_mapping(data, location))
    kind = values.get("kind", "identity")
    if kind == "identity":
        return _flat(IdentityEncoderConfig, values, location)
    if kind != "history" or set(values) - {"kind", "temporal"}:
        raise ValueError(f"{location} 只支持 identity/history 及 temporal 网络配置")
    temporal = _mapping(values.get("temporal", {}), f"{location}.temporal")
    temporal_kind = temporal.get("kind", "lstm")
    if temporal_kind == "gru":
        return HistoryEncoderConfig(temporal=_flat(GRUBackboneConfig, temporal, location))
    if temporal_kind == "lstm":
        return HistoryEncoderConfig(temporal=_flat(LSTMBackboneConfig, temporal, location))
    if temporal_kind == "transformer":
        return HistoryEncoderConfig(temporal=_flat(TransformerBackboneConfig, temporal, location))
    raise ValueError(f"{location}.temporal.kind 只支持 gru/lstm/transformer")


def algorithm_config_from_dict(
    data: Mapping[str, object],
    *,
    catalog: ExtensionCatalog | None = None,
) -> AlgorithmConfig:
    if type(data.get("schema_version")) is not int or data["schema_version"] != 2:
        raise ValueError("schema_version 必须是整数 2；旧网络字段需迁移到 backbone 配置")
    name = data.get("algorithm")
    if name == "sn_mappo":
        if unknown := set(data) - set(get_type_hints(SNMAPPOConfig)):
            raise ValueError(f"sn_mappo 含未知字段: {sorted(unknown)}")
        defaults = SNMAPPOConfig()
        roles = {}
        for role in ("leader", "coordinator", "followers"):
            if role not in data:
                roles[role] = getattr(defaults, role)
            else:
                parsed = algorithm_config_from_dict({
                    "schema_version": 2, "algorithm": "mappo", **_mapping(data[role], role)
                }, catalog=catalog)
                if not isinstance(parsed, MAPPOConfig):
                    raise ValueError("顺序角色必须使用 MAPPO 配置")
                roles[role] = parsed
        values = dict(data)
        values.update(roles)
        values["response"] = _flat(ImplicitResponseConfig, data.get("response", {}), "response")
        return _flat(SNMAPPOConfig, values, "sn_mappo")
    common = {"schema_version", "algorithm", "policy", "loss", "update"}
    if name == "mappo":
        allowed = common | {"critic", "advantage", "rollout"}
    elif name in ("maac", "maddpg", "masac", "qmix"):
        allowed = common | {
            "mixer" if name == "qmix" else "critic",
            "value_target",
            "replay",
            "target_update",
        }
    else:
        raise ValueError(f"未知 algorithm: {name}")
    if unknown := set(data) - allowed:
        raise ValueError(f"{name} 含未知字段: {sorted(unknown)}")
    policy = data.get("policy", {})
    critic = data.get("critic", {})
    loss = data.get("loss", {})
    if name == "mappo":
        return MAPPOConfig(
            policy=_component(
                policy,
                IndependentGaussianConfig
                if _mapping(policy, "policy").get("kind") == "independent_gaussian"
                else IndependentDiscreteConfig,
                StochasticPolicy, "policy", catalog,
            ),
            critic=_component(critic, CentralizedValueConfig, ValueNetwork, "critic", catalog),
            loss=_flat(MAPPOLossConfig, loss, "loss"),
            advantage=_flat(GAEConfig, data.get("advantage", {}), "advantage"),
            rollout=_flat(RolloutConfig, data.get("rollout", {}), "rollout"),
            update=_flat(PPOUpdateConfig, data.get("update", {}), "update"),
        )
    target_data = _mapping(data.get("target_update", {}), "target_update")
    target: SoftTargetConfig | HardTargetConfig
    if target_data.get("kind", "soft") == "soft":
        target = _flat(SoftTargetConfig, target_data, "target_update")
    elif target_data.get("kind") == "hard":
        target = _flat(HardTargetConfig, target_data, "target_update")
    else:
        raise ValueError("target_update.kind 必须是 soft 或 hard")
    # 两类 target 共用一个明确的 union；以下参数是离策略算法的公共配置。
    update_cls = {
        "maac": ActorCriticUpdateConfig,
        "maddpg": ActorCriticUpdateConfig,
        "masac": TemperatureActorCriticUpdateConfig,
        "qmix": OffPolicyUpdateConfig,
    }[name]
    shared: dict[str, Any] = dict(
        value_target=_flat(TD0Config, data.get("value_target", {}), "value_target"),
        replay=_flat(ReplayConfig, data.get("replay", {}), "replay"),
        update=_flat(update_cls, data.get("update", {}), "update"),
        target_update=target,
    )
    if name == "maac":
        return MAACConfig(
            policy=_component(policy, IndependentDiscreteConfig, DiscretePolicy, "policy", catalog),
            critic=_component(critic, AttentionQConfig, AttentionQNetwork, "critic", catalog),
            loss=_flat(MAACLossConfig, loss, "loss"),
            **shared,
        )
    if name == "maddpg":
        return MADDPGConfig(
            policy=_component(
                policy, IndependentDeterministicConfig, PolicyTopology, "policy", catalog
            ),
            critic=_component(critic, IndependentQConfig, QEnsemble, "critic", catalog),
            loss=_flat(MADDPGLossConfig, loss, "loss"),
            **shared,
        )
    if name == "masac":
        parsed_policy = _component(
            policy, IndependentGaussianConfig, PolicyTopology, "policy", catalog
        )
        if isinstance(parsed_policy, IndependentGaussianConfig) and not isinstance(
            parsed_policy.backbone, MLPBackboneConfig
        ):
            raise ValueError("离策略 policy 不支持跨环境步循环 backbone；请使用 history encoder")
        return MASACConfig(
            policy=parsed_policy,
            critic=_component(critic, TwinQConfig, TwinQEnsemble, "critic", catalog),
            loss=_flat(MASACLossConfig, loss, "loss"),
            **shared,
        )
    return QMIXConfig(
        policy=_component(policy, SharedDiscreteQConfig, LocalQPolicy, "policy", catalog),
        mixer=_component(data.get("mixer", {}), QMixerConfig, MixingNetwork, "mixer", catalog),
        loss=_flat(QMIXLossConfig, loss, "loss"),
        **shared,
    )


def load_algorithm_config(
    path: str | Path,
    *,
    catalog: ExtensionCatalog | None = None,
) -> AlgorithmConfig:
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return algorithm_config_from_dict(_mapping(data, "algorithm YAML"), catalog=catalog)


def config_to_dict(config: AlgorithmConfig) -> dict[str, Any]:
    """Checkpoint/YAML 数据快照；外部组件只保存名称和配置，不序列化构造器。"""

    def encode(value: Any) -> Any:
        if isinstance(value, ExternalConfig):
            return {"kind": value.kind, **deep_settings(value.settings)}
        if is_dataclass(value) and not isinstance(value, type):
            return {
                ("lambda" if item.name == "gae_lambda" else item.name): encode(
                    getattr(value, item.name)
                )
                for item in fields(value)
            }
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, torch.dtype):
            return "bf16" if value == torch.bfloat16 else str(value)
        if isinstance(value, Mapping):
            return {k: encode(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(v) for v in value]
        if value is None or type(value) in (str, bool, int, float):
            return value
        raise TypeError(f"配置必须是 dataclass 或 JSON 数据: {type(value).__name__}")

    def deep_settings(settings: Mapping[str, object]) -> dict[str, Any]:
        return {k: encode(v) for k, v in settings.items()}

    result: dict[str, Any] = encode(config)
    return result
