"""由通用策略、价值函数、目标和训练组件组装的 MAPPO。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from torch import Tensor, nn

from marl.algorithms.base import AlgorithmConfig, BaseMARLAlgorithm
from marl.builtins import NoTargetUpdate, OptimizerConfig, RolloutConfig
from marl.core import MARLBatch, MARLModelOutput
from marl.envs.base import EnvironmentSpec
from marl.models.mlp import MLPBackbone
from marl.modules.critic import CentralizedCritic
from marl.objectives import (
    EntropyObjective,
    LossBundle,
    PPOClipObjective,
    ValueMSEObjective,
)
from marl.policies import IndependentDiscretePolicy
from marl.recipes import (
    AlgorithmRecipe,
    CompiledRecipe,
    ComponentRecipe,
    compile_recipe,
)
from marl.registry import DEFAULT_COMPONENT_REGISTRY, ComponentKind, ComponentRegistry
from marl.returns import GAEEstimator


@dataclass(frozen=True, slots=True)
class MAPPOConfig(AlgorithmConfig):
    num_agents: int = 2
    observation_dim: int = 16
    action_dim: int = 5
    state_dim: int | None = None
    hidden_dim: int = 128
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01


def default_mappo_recipe(config: MAPPOConfig | None = None) -> AlgorithmRecipe:
    """返回与旧 MAPPO 默认数学语义一致的 Python recipe。"""

    hidden_dim = config.hidden_dim if config is not None else 128
    clip_ratio = config.clip_ratio if config is not None else 0.2
    value_coef = config.value_coef if config is not None else 0.5
    entropy_coef = config.entropy_coef if config is not None else 0.01
    gamma = config.gamma if config is not None else 0.99
    return AlgorithmRecipe(
        schema_version=1,
        algorithm="mappo",
        policy=ComponentRecipe("independent_discrete", {"hidden_dim": hidden_dim}),
        critic=ComponentRecipe("centralized_value", {"hidden_dim": hidden_dim}),
        objectives=(
            ComponentRecipe("ppo_clip", {"clip_ratio": clip_ratio}),
            ComponentRecipe("value_mse", {"coefficient": value_coef}),
            ComponentRecipe("entropy", {"coefficient": entropy_coef}),
        ),
        returns=ComponentRecipe(
            "gae",
            {"gamma": gamma, "lambda": 0.95, "normalize_advantage": True},
        ),
        experience=ComponentRecipe("rollout", {}),
        update=ComponentRecipe(
            "ppo_update",
            {
                "learning_rate": 3e-4,
                "epochs": 4,
                "mini_batch_size": 256,
                "max_grad_norm": config.grad_clip_norm if config is not None else 10.0,
            },
        ),
        target_update=ComponentRecipe("none", {}),
    )


class MAPPO(BaseMARLAlgorithm):
    """使用平级组件组装的离散、同构智能体 MAPPO 薄类。"""

    def __init__(
        self,
        config: MAPPOConfig,
        *,
        policy: IndependentDiscretePolicy | None = None,
        critic: nn.Module | None = None,
        objectives: tuple[object, ...] | None = None,
        return_estimator: GAEEstimator | None = None,
        rollout_config: RolloutConfig | None = None,
        optimizer_config: OptimizerConfig | None = None,
        target_update: NoTargetUpdate | None = None,
        compiled_recipe: CompiledRecipe | None = None,
    ) -> None:
        super().__init__(config)
        critic_input = config.state_dim or config.num_agents * config.observation_dim
        self.policy = policy or IndependentDiscretePolicy(
            config.num_agents,
            config.observation_dim,
            config.action_dim,
            config.hidden_dim,
        )
        self.critic = critic or CentralizedCritic(
            MLPBackbone(critic_input, output_dim=config.hidden_dim),
            output_dim=config.num_agents,
        )
        selected = objectives or (
            PPOClipObjective(config.clip_ratio),
            ValueMSEObjective(config.value_coef),
            EntropyObjective(config.entropy_coef),
        )
        self.policy_objective = self._require_objective(selected, PPOClipObjective)
        self.value_objective = self._require_objective(selected, ValueMSEObjective)
        self.entropy_objective = self._require_objective(selected, EntropyObjective)
        self.return_estimator = return_estimator or GAEEstimator(gamma=config.gamma)
        self.rollout_config = rollout_config or RolloutConfig()
        self.optimizer_config = optimizer_config or OptimizerConfig()
        self.target_update_rule = target_update or NoTargetUpdate()
        self.compiled_recipe = compiled_recipe

    @staticmethod
    def _require_objective(
        objectives: tuple[object, ...], expected: type[object]
    ) -> object:
        matches = [objective for objective in objectives if isinstance(objective, expected)]
        if len(matches) != 1:
            raise ValueError(f"MAPPO recipe 必须且只能包含一个 {expected.__name__}")
        return matches[0]

    @classmethod
    def from_recipe(
        cls,
        spec: EnvironmentSpec,
        recipe: AlgorithmRecipe,
        *,
        registry: ComponentRegistry = DEFAULT_COMPONENT_REGISTRY,
    ) -> MAPPO:
        if recipe.algorithm != "mappo":
            raise ValueError(f"MAPPO 不能从 algorithm={recipe.algorithm!r} 的 recipe 构造")
        compiled = compile_recipe(recipe, spec, registry=registry)
        policy = cast(
            IndependentDiscretePolicy,
            registry.build(
                ComponentKind.POLICY, recipe.policy.type, recipe.policy.options, spec
            ),
        )
        critic = cast(
            nn.Module,
            registry.build(
                ComponentKind.CRITIC, recipe.critic.type, recipe.critic.options, spec
            ),
        )
        objectives = tuple(
            registry.build(ComponentKind.OBJECTIVE, item.type, item.options, spec)
            for item in recipe.objectives
        )
        estimator = cast(
            GAEEstimator,
            registry.build(
                ComponentKind.RETURN_ESTIMATOR,
                recipe.returns.type,
                recipe.returns.options,
                spec,
            ),
        )
        rollout = cast(
            RolloutConfig,
            registry.build(
                ComponentKind.EXPERIENCE_SOURCE,
                recipe.experience.type,
                recipe.experience.options,
                spec,
            ),
        )
        optimizer = cast(
            OptimizerConfig,
            registry.build(
                ComponentKind.UPDATE_PLAN,
                recipe.update.type,
                recipe.update.options,
                spec,
            ),
        )
        target = cast(
            NoTargetUpdate,
            registry.build(
                ComponentKind.TARGET_UPDATE,
                recipe.target_update.type,
                recipe.target_update.options,
                spec,
            ),
        )
        hidden_option = recipe.policy.options.get("hidden_dim", 128)
        if not isinstance(hidden_option, int) or isinstance(hidden_option, bool):
            raise TypeError("policy hidden_dim 必须是整数")
        hidden_dim = hidden_option
        policy_objective = cast(
            PPOClipObjective,
            cls._require_objective(objectives, PPOClipObjective),
        )
        value_objective = cast(
            ValueMSEObjective,
            cls._require_objective(objectives, ValueMSEObjective),
        )
        entropy_objective = cast(
            EntropyObjective,
            cls._require_objective(objectives, EntropyObjective),
        )
        config = MAPPOConfig(
            num_agents=spec.num_agents,
            observation_dim=spec.observation_dim,
            action_dim=spec.action_dim,
            state_dim=spec.state_dim,
            hidden_dim=hidden_dim,
            gamma=estimator.gamma,
            clip_ratio=policy_objective.clip_ratio,
            value_coef=value_objective.coefficient,
            entropy_coef=entropy_objective.coefficient,
            grad_clip_norm=optimizer.update.max_grad_norm,
        )
        return cls(
            config,
            policy=policy,
            critic=critic,
            objectives=objectives,
            return_estimator=estimator,
            rollout_config=rollout,
            optimizer_config=optimizer,
            target_update=target,
            compiled_recipe=compiled,
        )

    @property
    def cfg(self) -> MAPPOConfig:
        return cast(MAPPOConfig, self.config)

    @property
    def actors(self) -> nn.ModuleList:
        """兼容旧的只读网络检查；策略所有权仍属于 policy 组件。"""

        return self.policy.actors

    def _policy_logits(self, observations: Tensor, action_mask: Tensor | None) -> Tensor:
        return self.policy.logits(observations, action_mask)

    def _critic_input(self, batch: MARLBatch) -> Tensor:
        if batch.state is not None:
            return batch.state
        return batch.observations.flatten(-2)

    def values(self, observations: Tensor, state: Tensor | None = None) -> Tensor:
        critic_input = state if state is not None else observations.flatten(-2)
        return cast(Tensor, self.critic(critic_input))

    def sample(
        self,
        observations: Tensor,
        state: Tensor | None = None,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
    ) -> MARLModelOutput:
        policy_output = self.policy.act(
            observations,
            deterministic=deterministic,
            action_mask=action_mask,
        )
        policy_output.values = self.values(observations, state)
        return policy_output

    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
        **kwargs: Tensor,
    ) -> Tensor:
        return self.policy.act(
            observations,
            deterministic=deterministic,
            action_mask=action_mask,
        ).actions

    def compute_loss_bundle(self, batch: MARLBatch) -> LossBundle:
        if batch.actions is None:
            raise ValueError("MAPPO 训练需要 actions")
        required = ("old_log_prob", "advantages", "returns")
        if any(name not in batch.extras for name in required):
            raise ValueError(f"MAPPO batch.extras 必须包含 {required}")
        expected_shape = batch.observations.shape[:-1]
        for name in required:
            if batch.extras[name].shape != expected_shape:
                raise ValueError(f"MAPPO {name} 必须具有逐智能体形状 {expected_shape}")

        output = self.policy.evaluate(
            batch.observations,
            batch.actions,
            action_mask=batch.action_mask,
        )
        assert output.log_prob is not None and output.entropy is not None
        values = self.critic(self._critic_input(batch))
        return LossBundle.combine(
            (
                cast(PPOClipObjective, self.policy_objective)(
                    output.log_prob,
                    batch.extras["old_log_prob"],
                    batch.extras["advantages"],
                ),
                cast(ValueMSEObjective, self.value_objective)(
                    values, batch.extras["returns"]
                ),
                cast(EntropyObjective, self.entropy_objective)(output.entropy),
            )
        )

    def compute_loss(self, batch: MARLBatch) -> dict[str, Tensor]:
        """临时兼容旧算法测试；新 trainer 使用 compute_loss_bundle。"""

        bundle = self.compute_loss_bundle(batch)
        return {
            "loss": bundle.total,
            "actor_loss": bundle.terms["policy_loss"],
            "value_loss": bundle.terms["value_loss"],
            "entropy": bundle.terms["entropy"],
        }
