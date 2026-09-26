"""可选外部组件目录。内置组件直接 build，不经过此目录。"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Generic, Literal, Protocol, TypeVar

from torch import nn

from marl.envs.base import ActionKind, EnvironmentSpec, RewardStructure

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
C = TypeVar("C")
Category = Literal["policy", "critic", "mixer"]


class Buildable(Protocol[T_co]):
    """配置只需暴露 build；内置与外部实现共享这个很小的接口。"""

    @property
    def kind(self) -> str: ...

    def build(self, spec: EnvironmentSpec) -> T_co: ...


@dataclass(frozen=True)
class ExternalConfig(Generic[T]):
    kind: str
    settings: Mapping[str, object]
    create: Callable[[EnvironmentSpec], T] = field(repr=False, compare=False)

    def build(self, spec: EnvironmentSpec) -> T:
        return self.create(spec)


@dataclass(frozen=True)
class _Entry:
    configure: Callable[[Mapping[str, object]], Callable[[EnvironmentSpec], nn.Module]]
    action_kinds: frozenset[ActionKind]
    reward_structures: frozenset[RewardStructure]


class ExtensionCatalog:
    """仅注册外部 policy/critic/mixer；调用者显式传给 YAML loader。"""

    def __init__(self) -> None:
        self._entries: dict[tuple[Category, str], _Entry] = {}

    def register(
        self, category: Category, name: str,
        parse: Callable[[Mapping[str, object]], C],
        build: Callable[[C, EnvironmentSpec], nn.Module],
        *, action_kinds: frozenset[ActionKind],
        reward_structures: frozenset[RewardStructure] = frozenset(RewardStructure),
    ) -> None:
        if category not in ("policy", "critic", "mixer"):
            raise ValueError(f"未知扩展类别: {category}")
        if not name or name != name.strip().lower():
            raise ValueError("扩展名称必须是非空小写名称")
        key = (category, name)
        if key in self._entries:
            raise ValueError(f"扩展已注册: {category}/{name}")

        def configure(data: Mapping[str, object]) -> Callable[[EnvironmentSpec], nn.Module]:
            parsed = parse(data)
            return lambda spec: build(parsed, spec)

        self._entries[key] = _Entry(configure, action_kinds, reward_structures)

    def configure(
        self, category: Category, name: str, settings: Mapping[str, object],
        capability: type[T],
    ) -> ExternalConfig[T]:
        try:
            entry = self._entries[category, name]
        except KeyError as error:
            raise ValueError(f"未知 {category} 扩展: {name}") from error
        saved = deepcopy(dict(settings))
        create = entry.configure(deepcopy(saved))

        def build(spec: EnvironmentSpec) -> T:
            if spec.action_kind not in entry.action_kinds:
                raise ValueError(f"{name} 不支持 {spec.action_kind.value} 动作")
            if spec.reward_structure not in entry.reward_structures:
                raise ValueError(f"{name} 不支持 {spec.reward_structure.value} 奖励")
            module = create(spec)
            if not isinstance(module, nn.Module) or not isinstance(module, capability):
                raise TypeError(f"{name} 必须是 nn.Module 并满足 {capability.__name__}")
            return module

        return ExternalConfig(name, saved, build)
