"""组合式离散动作 MAAC，保留逐智能体收益。"""

from __future__ import annotations

from copy import deepcopy
from typing import cast

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from marl.algorithms.assembly import assemble_algorithm_components, require_one
from marl.algorithms.base import BaseMARLAlgorithm
from marl.core import MARLBatch
from marl.envs.base import EnvironmentSpec
from marl.modules.critic import AttentionCritic
from marl.objectives import (
    CounterfactualPolicyObjective,
    EntropyObjective,
    LossBundle,
    TDLossObjective,
)
from marl.policies import IndependentDiscretePolicy
from marl.recipes import AlgorithmRecipe, CompiledRecipe, ComponentRecipe
from marl.registry import DEFAULT_COMPONENT_REGISTRY, ComponentRegistry
from marl.returns import TD0Estimator


def default_maac_recipe() -> AlgorithmRecipe:
    return AlgorithmRecipe(
        schema_version=1,
        algorithm="maac",
        policy=ComponentRecipe("independent_discrete", {"hidden_dim": 128}),
        critic=ComponentRecipe(
            "attention_q", {"hidden_dim": 128, "attention_heads": 4}
        ),
        objectives=(
            ComponentRecipe("td_mse", {"coefficient": 1.0}),
            ComponentRecipe("counterfactual", {}),
            ComponentRecipe("entropy", {"coefficient": 0.01}),
        ),
        returns=ComponentRecipe("td0", {"gamma": 0.99}),
        experience=ComponentRecipe("replay", {"capacity": 100_000, "batch_size": 256}),
        update=ComponentRecipe(
            "off_policy_update",
            {"learning_rate": 3e-4, "max_grad_norm": 10.0, "amp_dtype": None},
        ),
        target_update=ComponentRecipe("soft", {"tau": 0.005}),
    )


class MAAC(BaseMARLAlgorithm):
    def __init__(
        self,
        spec: EnvironmentSpec,
        policy: IndependentDiscretePolicy,
        critic: AttentionCritic,
        td_loss: TDLossObjective,
        policy_objective: CounterfactualPolicyObjective,
        entropy_objective: EntropyObjective,
        return_estimator: TD0Estimator,
        compiled_recipe: CompiledRecipe,
    ) -> None:
        super().__init__(spec)
        self.policy = policy
        self.critic = critic
        self.target_policy = deepcopy(policy).requires_grad_(False)
        self.target_critic = deepcopy(critic).requires_grad_(False)
        self.td_loss = td_loss
        self.policy_objective = policy_objective
        self.entropy_objective = entropy_objective
        self.return_estimator = return_estimator
        self.compiled_recipe = compiled_recipe

    @classmethod
    def from_recipe(
        cls,
        spec: EnvironmentSpec,
        recipe: AlgorithmRecipe,
        *,
        registry: ComponentRegistry = DEFAULT_COMPONENT_REGISTRY,
    ) -> MAAC:
        parts = assemble_algorithm_components("maac", spec, recipe, registry)
        if not isinstance(parts.policy, IndependentDiscretePolicy):
            raise TypeError("MAAC policy 必须是 IndependentDiscretePolicy")
        if not isinstance(parts.critic, AttentionCritic):
            raise TypeError("MAAC critic 必须是 AttentionCritic")
        if not isinstance(parts.returns, TD0Estimator):
            raise TypeError("MAAC returns 必须是 TD0Estimator")
        return cls(
            spec,
            parts.policy,
            parts.critic,
            cast(TDLossObjective, require_one(parts.objectives, TDLossObjective, "MAAC")),
            cast(
                CounterfactualPolicyObjective,
                require_one(parts.objectives, CounterfactualPolicyObjective, "MAAC"),
            ),
            cast(EntropyObjective, require_one(parts.objectives, EntropyObjective, "MAAC")),
            parts.returns,
            parts.compiled,
        )

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
        batch.validate(self.spec.num_agents, self.spec.observation_dim)
        if batch.actions is None or batch.rewards is None or batch.next_observations is None:
            raise ValueError("MAAC 训练需要 actions/rewards/next_observations")
        terminated, _ = batch.terminal_flags()
        q_all = self.critic(batch.observations, batch.actions)
        chosen_q = q_all.gather(
            -1, batch.actions.long().unsqueeze(-1)
        ).squeeze(-1)
        with torch.no_grad():
            next_output = self.target_policy.act(
                batch.next_observations,
                action_mask=batch.next_action_mask,
            )
            assert next_output.log_prob is not None
            next_q_all = self.target_critic(
                batch.next_observations, next_output.actions
            )
            next_q = next_q_all.gather(
                -1, next_output.actions.unsqueeze(-1)
            ).squeeze(-1)
            soft_next_q = (
                next_q
                - self.entropy_objective.coefficient * next_output.log_prob
            )
            target_q = self.return_estimator.estimate(
                batch.rewards, soft_next_q, terminated
            )
        critic_result = self.td_loss(chosen_q, target_q)

        logits = self.policy.logits(batch.observations, batch.action_mask)
        actor_result = self.policy_objective(logits, q_all)
        entropy = Categorical(logits=logits, validate_args=False).entropy()
        return LossBundle.combine(
            (critic_result, actor_result, self.entropy_objective(entropy))
        )

    def target_pairs(self) -> tuple[tuple[nn.Module, nn.Module], ...]:
        return (
            (self.target_policy, self.policy),
            (self.target_critic, self.critic),
        )
