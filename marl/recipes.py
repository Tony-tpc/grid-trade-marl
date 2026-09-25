"""Python/YAML 算法配方及环境尺寸编译。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias, cast

import yaml  # type: ignore[import-untyped]

from marl.envs.base import EnvironmentSpec
from marl.registry import DEFAULT_COMPONENT_REGISTRY, ComponentKind, ComponentRegistry

ConfigScalar: TypeAlias = str | int | float | bool | None
ConfigValue: TypeAlias = ConfigScalar | tuple["ConfigValue", ...] | Mapping[str, "ConfigValue"]

_FORBIDDEN_ENVIRONMENT_OPTIONS = frozenset(
    {"num_agents", "observation_dim", "action_dim", "state_dim", "action_kind"}
)


def _freeze(value: object) -> ConfigValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, Mapping):
        frozen = {str(key): _freeze(item) for key, item in value.items()}
        return MappingProxyType(frozen)
    raise TypeError(f"配置值不支持 {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class ComponentRecipe:
    type: str
    options: Mapping[str, ConfigValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = self.type.strip().lower()
        if not normalized:
            raise ValueError("组件 type 不能为空")
        object.__setattr__(self, "type", normalized)
        object.__setattr__(self, "options", cast(Mapping[str, ConfigValue], _freeze(self.options)))
        forbidden = _FORBIDDEN_ENVIRONMENT_OPTIONS & set(self.options)
        if forbidden:
            raise ValueError(f"环境尺寸不能由 recipe 覆盖：{sorted(forbidden)}")


@dataclass(frozen=True, slots=True)
class AlgorithmRecipe:
    schema_version: int
    algorithm: str
    policy: ComponentRecipe
    critic: ComponentRecipe
    objectives: tuple[ComponentRecipe, ...]
    returns: ComponentRecipe
    experience: ComponentRecipe
    update: ComponentRecipe
    target_update: ComponentRecipe

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"不支持 recipe schema_version={self.schema_version}；当前仅支持 1")
        normalized = self.algorithm.strip().lower()
        if not normalized:
            raise ValueError("algorithm 不能为空")
        if not self.objectives:
            raise ValueError("recipe 至少需要一个 objective")
        object.__setattr__(self, "algorithm", normalized)
        object.__setattr__(self, "objectives", tuple(self.objectives))


@dataclass(frozen=True, slots=True)
class CompiledRecipe:
    """已验证组件名称并绑定环境规格的不可变配方。"""

    recipe: AlgorithmRecipe
    environment: EnvironmentSpec


_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "algorithm",
        "policy",
        "critic",
        "objectives",
        "returns",
        "experience",
        "update",
        "target_update",
    }
)


def _component_from_data(data: object, location: str) -> ComponentRecipe:
    if not isinstance(data, Mapping):
        raise TypeError(f"{location} 必须是 mapping")
    unknown = set(data) - {"type", "options"}
    if unknown:
        raise ValueError(f"{location} 含未知字段：{sorted(unknown)}")
    if "type" not in data:
        raise ValueError(f"{location} 缺少 type")
    options = data.get("options", {})
    if not isinstance(options, Mapping):
        raise TypeError(f"{location}.options 必须是 mapping")
    return ComponentRecipe(
        type=str(data["type"]),
        options=cast(Mapping[str, ConfigValue], options),
    )


def recipe_from_dict(data: Mapping[str, object]) -> AlgorithmRecipe:
    unknown = set(data) - _TOP_LEVEL_FIELDS
    missing = _TOP_LEVEL_FIELDS - set(data)
    if unknown:
        raise ValueError(f"recipe 含未知字段：{sorted(unknown)}")
    if missing:
        raise ValueError(f"recipe 缺少字段：{sorted(missing)}")
    objective_data = data["objectives"]
    if not isinstance(objective_data, list | tuple):
        raise TypeError("objectives 必须是列表")
    return AlgorithmRecipe(
        schema_version=int(cast(int, data["schema_version"])),
        algorithm=str(data["algorithm"]),
        policy=_component_from_data(data["policy"], "policy"),
        critic=_component_from_data(data["critic"], "critic"),
        objectives=tuple(
            _component_from_data(item, f"objectives[{index}]")
            for index, item in enumerate(objective_data)
        ),
        returns=_component_from_data(data["returns"], "returns"),
        experience=_component_from_data(data["experience"], "experience"),
        update=_component_from_data(data["update"], "update"),
        target_update=_component_from_data(data["target_update"], "target_update"),
    )


def load_algorithm_recipe(path: str | Path) -> AlgorithmRecipe:
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, Mapping):
        raise TypeError("recipe YAML 顶层必须是 mapping")
    return recipe_from_dict(cast(Mapping[str, object], data))


def _thaw(value: ConfigValue) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def recipe_to_dict(recipe: AlgorithmRecipe) -> dict[str, object]:
    def component(item: ComponentRecipe) -> dict[str, object]:
        result: dict[str, object] = {"type": item.type}
        if item.options:
            result["options"] = {
                key: _thaw(value) for key, value in item.options.items()
            }
        return result

    return {
        "schema_version": recipe.schema_version,
        "algorithm": recipe.algorithm,
        "policy": component(recipe.policy),
        "critic": component(recipe.critic),
        "objectives": [component(item) for item in recipe.objectives],
        "returns": component(recipe.returns),
        "experience": component(recipe.experience),
        "update": component(recipe.update),
        "target_update": component(recipe.target_update),
    }


def compile_recipe(
    recipe: AlgorithmRecipe,
    environment: EnvironmentSpec,
    *,
    registry: ComponentRegistry = DEFAULT_COMPONENT_REGISTRY,
) -> CompiledRecipe:
    components: tuple[tuple[ComponentKind, ComponentRecipe], ...] = (
        (ComponentKind.POLICY, recipe.policy),
        (ComponentKind.CRITIC, recipe.critic),
        *((ComponentKind.OBJECTIVE, item) for item in recipe.objectives),
        (ComponentKind.RETURN_ESTIMATOR, recipe.returns),
        (ComponentKind.EXPERIENCE_SOURCE, recipe.experience),
        (ComponentKind.UPDATE_PLAN, recipe.update),
        (ComponentKind.TARGET_UPDATE, recipe.target_update),
    )
    for kind, component in components:
        registration = registry.resolve(kind, component.type)
        registration.validate(component.type, environment)
    return CompiledRecipe(recipe=recipe, environment=environment)


# 注册发生在 recipe 模块完成定义之后，避免组件工厂反向依赖配置解析。
from marl.builtins import register_builtin_components  # noqa: E402

register_builtin_components(DEFAULT_COMPONENT_REGISTRY)
