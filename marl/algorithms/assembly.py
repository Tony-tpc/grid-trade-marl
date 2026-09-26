"""具体算法共享的 recipe 组件装配。"""

from __future__ import annotations

from dataclasses import dataclass

from marl.envs.base import EnvironmentSpec
from marl.recipes import AlgorithmRecipe, CompiledRecipe, compile_recipe
from marl.registry import DEFAULT_COMPONENT_REGISTRY, ComponentKind, ComponentRegistry


@dataclass(frozen=True, slots=True)
class AlgorithmComponents:
    compiled: CompiledRecipe
    policy: object
    critic: object
    objectives: tuple[object, ...]
    returns: object


def assemble_algorithm_components(
    algorithm: str,
    spec: EnvironmentSpec,
    recipe: AlgorithmRecipe,
    registry: ComponentRegistry = DEFAULT_COMPONENT_REGISTRY,
) -> AlgorithmComponents:
    if recipe.algorithm != algorithm:
        raise ValueError(
            f"{algorithm.upper()} 不能从 algorithm={recipe.algorithm!r} 的 recipe 构造"
        )
    compiled = compile_recipe(recipe, spec, registry=registry)
    return AlgorithmComponents(
        compiled=compiled,
        policy=registry.build(
            ComponentKind.POLICY, recipe.policy.type, recipe.policy.options, spec
        ),
        critic=registry.build(
            ComponentKind.CRITIC, recipe.critic.type, recipe.critic.options, spec
        ),
        objectives=tuple(
            registry.build(ComponentKind.OBJECTIVE, item.type, item.options, spec)
            for item in recipe.objectives
        ),
        returns=registry.build(
            ComponentKind.RETURN_ESTIMATOR,
            recipe.returns.type,
            recipe.returns.options,
            spec,
        ),
    )


def require_one(
    values: tuple[object, ...], expected: type[object], algorithm: str
) -> object:
    matches = [value for value in values if isinstance(value, expected)]
    if len(matches) != 1:
        raise ValueError(
            f"{algorithm} recipe 必须且只能包含一个 {expected.__name__}"
        )
    return matches[0]
