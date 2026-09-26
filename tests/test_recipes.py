from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from marl.algorithms import (
    default_maac_recipe,
    default_maddpg_recipe,
    default_masac_recipe,
    default_qmix_recipe,
)
from marl.envs import ActionKind, EnvironmentSpec, RewardStructure
from marl.recipes import (
    AlgorithmRecipe,
    ComponentRecipe,
    ConfigValue,
    compile_recipe,
    load_algorithm_recipe,
    recipe_from_dict,
)
from marl.registry import ComponentKind, ComponentRegistry


def component(name: str, **options: object) -> ComponentRecipe:
    return ComponentRecipe(name, cast(Mapping[str, ConfigValue], options))


def recipe() -> AlgorithmRecipe:
    return AlgorithmRecipe(
        schema_version=1,
        algorithm="mappo",
        policy=component("policy", hidden_dim=16),
        critic=component("critic"),
        objectives=(component("policy_loss"), component("value_loss")),
        returns=component("gae"),
        experience=component("rollout"),
        update=component("ppo_update"),
        target_update=component("none"),
    )


def environment(action_kind: ActionKind = ActionKind.DISCRETE) -> EnvironmentSpec:
    return EnvironmentSpec(
        num_agents=2,
        observation_dim=3,
        action_dim=4,
        state_dim=6,
        action_kind=action_kind,
        horizon=8,
        reward_structure=RewardStructure.INDIVIDUAL,
    )


def populated_registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    for kind, name in (
        (ComponentKind.POLICY, "policy"),
        (ComponentKind.CRITIC, "critic"),
        (ComponentKind.OBJECTIVE, "policy_loss"),
        (ComponentKind.OBJECTIVE, "value_loss"),
        (ComponentKind.RETURN_ESTIMATOR, "gae"),
        (ComponentKind.EXPERIENCE_SOURCE, "rollout"),
        (ComponentKind.UPDATE_PLAN, "ppo_update"),
        (ComponentKind.TARGET_UPDATE, "none"),
    ):
        registry.register(kind, name, lambda options, spec: (options, spec))
    return registry


def test_python_and_yaml_recipes_are_equal(tmp_path: Path) -> None:
    path = tmp_path / "mappo.yaml"
    path.write_text(
        """schema_version: 1
algorithm: mappo
policy: {type: policy, options: {hidden_dim: 16}}
critic: {type: critic}
objectives:
  - {type: policy_loss}
  - {type: value_loss}
returns: {type: gae}
experience: {type: rollout}
update: {type: ppo_update}
target_update: {type: none}
""",
        encoding="utf-8",
    )
    assert load_algorithm_recipe(path) == recipe()


@pytest.mark.parametrize(
    ("name", "factory"),
    (
        ("maac", default_maac_recipe),
        ("maddpg", default_maddpg_recipe),
        ("masac", default_masac_recipe),
        ("qmix", default_qmix_recipe),
    ),
)
def test_off_policy_python_and_yaml_defaults_are_equal(
    name: str, factory: object
) -> None:
    assert callable(factory)
    path = Path(__file__).parents[1] / "examples" / "configs" / f"{name}.yaml"
    assert load_algorithm_recipe(path) == factory()


def test_recipe_rejects_unknown_and_missing_fields() -> None:
    data = {
        "schema_version": 1,
        "algorithm": "mappo",
        "unexpected": True,
    }
    with pytest.raises(ValueError, match="未知字段"):
        recipe_from_dict(data)

    data.pop("unexpected")
    with pytest.raises(ValueError, match="缺少字段"):
        recipe_from_dict(data)


def test_recipe_is_immutable_and_rejects_environment_dimensions() -> None:
    item = component("policy", layers=[16, 16])
    assert item.options["layers"] == (16, 16)
    with pytest.raises(TypeError):
        item.options["layers"] = (32,)  # type: ignore[index]
    with pytest.raises(ValueError, match="环境尺寸"):
        component("policy", observation_dim=3)


def test_registry_rejects_duplicates_and_reports_unknown_names() -> None:
    registry = ComponentRegistry()

    def factory(
        options: Mapping[str, object], spec: EnvironmentSpec
    ) -> tuple[Mapping[str, object], EnvironmentSpec]:
        return options, spec

    registry.register(ComponentKind.POLICY, "shared", factory)
    with pytest.raises(ValueError, match="已注册"):
        registry.register(ComponentKind.POLICY, "shared", factory)
    with pytest.raises(KeyError, match="已注册：shared"):
        registry.resolve(ComponentKind.POLICY, "missing")


def test_compile_validates_action_and_reward_compatibility() -> None:
    registry = populated_registry()
    discrete = frozenset({ActionKind.DISCRETE})
    shared = frozenset({RewardStructure.SHARED})
    incompatible = ComponentRegistry()
    for kind in ComponentKind:
        name = {
            ComponentKind.POLICY: "policy",
            ComponentKind.CRITIC: "critic",
            ComponentKind.OBJECTIVE: "policy_loss",
            ComponentKind.RETURN_ESTIMATOR: "gae",
            ComponentKind.EXPERIENCE_SOURCE: "rollout",
            ComponentKind.UPDATE_PLAN: "ppo_update",
            ComponentKind.TARGET_UPDATE: "none",
        }[kind]
        incompatible.register(
            kind,
            name,
            lambda options, spec: None,
            action_kinds=discrete if kind == ComponentKind.POLICY else None,
            reward_structures=shared if kind == ComponentKind.CRITIC else None,
        )
        if kind == ComponentKind.OBJECTIVE:
            incompatible.register(kind, "value_loss", lambda options, spec: None)

    assert compile_recipe(recipe(), environment(), registry=registry).environment.num_agents == 2
    with pytest.raises(ValueError, match="不支持 continuous 动作"):
        compile_recipe(recipe(), environment(ActionKind.CONTINUOUS), registry=incompatible)
    with pytest.raises(ValueError, match="不支持 individual 奖励"):
        compile_recipe(recipe(), environment(), registry=incompatible)


def test_schema_version_and_component_fields_are_strict() -> None:
    with pytest.raises(ValueError, match="schema_version=2"):
        AlgorithmRecipe(
            schema_version=2,
            algorithm="mappo",
            policy=component("policy"),
            critic=component("critic"),
            objectives=(component("loss"),),
            returns=component("gae"),
            experience=component("rollout"),
            update=component("update"),
            target_update=component("none"),
        )
    with pytest.raises(ValueError, match="未知字段"):
        recipe_from_dict(
            {
                "schema_version": 1,
                "algorithm": "mappo",
                "policy": {"type": "policy", "class_path": "unsafe.module"},
                "critic": {"type": "critic"},
                "objectives": [{"type": "loss"}],
                "returns": {"type": "gae"},
                "experience": {"type": "rollout"},
                "update": {"type": "update"},
                "target_update": {"type": "none"},
            }
        )
