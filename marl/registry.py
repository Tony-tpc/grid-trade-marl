"""安全、显式的算法组件注册表。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias

from marl.envs.base import ActionKind, EnvironmentSpec, RewardStructure

ComponentFactory: TypeAlias = Callable[[Mapping[str, object], EnvironmentSpec], object]


class ComponentKind(str, Enum):
    POLICY = "policy"
    CRITIC = "critic"
    OBJECTIVE = "objective"
    RETURN_ESTIMATOR = "return_estimator"
    EXPERIENCE_SOURCE = "experience_source"
    UPDATE_PLAN = "update_plan"
    TARGET_UPDATE = "target_update"


@dataclass(frozen=True, slots=True)
class ComponentRegistration:
    """一个组件构造器及其环境兼容条件。"""

    factory: ComponentFactory
    action_kinds: frozenset[ActionKind] | None = None
    reward_structures: frozenset[RewardStructure] | None = None

    def validate(self, name: str, spec: EnvironmentSpec) -> None:
        if self.action_kinds is not None and spec.action_kind not in self.action_kinds:
            expected = ", ".join(sorted(kind.value for kind in self.action_kinds))
            raise ValueError(
                f"组件 {name!r} 不支持 {spec.action_kind.value} 动作；支持：{expected}"
            )
        if (
            self.reward_structures is not None
            and spec.reward_structure not in self.reward_structures
        ):
            expected = ", ".join(
                sorted(structure.value for structure in self.reward_structures)
            )
            raise ValueError(
                f"组件 {name!r} 不支持 {spec.reward_structure.value} 奖励；支持：{expected}"
            )


class ComponentRegistry:
    """按类别保存受控组件构造器，不执行配置中的导入路径。"""

    def __init__(self) -> None:
        self._registrations: dict[
            ComponentKind, dict[str, ComponentRegistration]
        ] = {kind: {} for kind in ComponentKind}

    def register(
        self,
        kind: ComponentKind,
        name: str,
        factory: ComponentFactory,
        *,
        action_kinds: frozenset[ActionKind] | None = None,
        reward_structures: frozenset[RewardStructure] | None = None,
    ) -> None:
        normalized = name.strip().lower()
        if not normalized or normalized != name:
            raise ValueError("组件名称必须是非空的小写字符串且不能包含首尾空格")
        category = self._registrations[kind]
        if normalized in category:
            raise ValueError(f"{kind.value} 组件 {normalized!r} 已注册")
        category[normalized] = ComponentRegistration(
            factory=factory,
            action_kinds=action_kinds,
            reward_structures=reward_structures,
        )

    def resolve(self, kind: ComponentKind, name: str) -> ComponentRegistration:
        normalized = name.strip().lower()
        try:
            return self._registrations[kind][normalized]
        except KeyError as error:
            available = ", ".join(sorted(self._registrations[kind])) or "<none>"
            raise KeyError(
                f"未知 {kind.value} 组件 {name!r}；已注册：{available}"
            ) from error

    def build(
        self,
        kind: ComponentKind,
        name: str,
        options: Mapping[str, object],
        spec: EnvironmentSpec,
    ) -> object:
        registration = self.resolve(kind, name)
        registration.validate(name, spec)
        return registration.factory(options, spec)

    def names(self, kind: ComponentKind) -> tuple[str, ...]:
        return tuple(sorted(self._registrations[kind]))


DEFAULT_COMPONENT_REGISTRY = ComponentRegistry()


def _register(
    kind: ComponentKind,
    name: str,
    factory: ComponentFactory,
    **compatibility: Any,
) -> None:
    DEFAULT_COMPONENT_REGISTRY.register(kind, name, factory, **compatibility)


def register_policy(name: str, factory: ComponentFactory, **compatibility: Any) -> None:
    _register(ComponentKind.POLICY, name, factory, **compatibility)


def register_critic(name: str, factory: ComponentFactory, **compatibility: Any) -> None:
    _register(ComponentKind.CRITIC, name, factory, **compatibility)


def register_objective(name: str, factory: ComponentFactory, **compatibility: Any) -> None:
    _register(ComponentKind.OBJECTIVE, name, factory, **compatibility)


def register_return_estimator(
    name: str, factory: ComponentFactory, **compatibility: Any
) -> None:
    _register(ComponentKind.RETURN_ESTIMATOR, name, factory, **compatibility)


def register_experience_source(
    name: str, factory: ComponentFactory, **compatibility: Any
) -> None:
    _register(ComponentKind.EXPERIENCE_SOURCE, name, factory, **compatibility)


def register_update_plan(name: str, factory: ComponentFactory, **compatibility: Any) -> None:
    _register(ComponentKind.UPDATE_PLAN, name, factory, **compatibility)


def register_target_update(
    name: str, factory: ComponentFactory, **compatibility: Any
) -> None:
    _register(ComponentKind.TARGET_UPDATE, name, factory, **compatibility)
