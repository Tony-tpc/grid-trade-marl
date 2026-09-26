"""组合式一般和博弈多智能体 SAC。"""

from __future__ import annotations

from copy import deepcopy
from typing import cast

import torch
from torch import Tensor, nn

from marl.algorithms.assembly import assemble_algorithm_components, require_one
from marl.algorithms.base import BaseMARLAlgorithm
from marl.core import MARLBatch
from marl.envs.base import EnvironmentSpec
from marl.modules.critic import TwinIndependentCentralizedCritics
from marl.objectives import (
    LossBundle,
    ObjectiveResult,
    SACEntropyObjective,
    TDLossObjective,
)
from marl.policies import IndependentGaussianPolicy
from marl.recipes import AlgorithmRecipe, CompiledRecipe, ComponentRecipe
from marl.registry import DEFAULT_COMPONENT_REGISTRY, ComponentRegistry
from marl.returns import TD0Estimator
from marl.training.gradients import frozen_parameters


def default_masac_recipe() -> AlgorithmRecipe:
    return AlgorithmRecipe(
        schema_version=1,
        algorithm="masac",
        policy=ComponentRecipe("independent_gaussian", {"hidden_dim": 128}),
        critic=ComponentRecipe(
            "twin_independent_centralized_q", {"hidden_dim": 128}
        ),
        objectives=(
            ComponentRecipe("td_mse", {"coefficient": 1.0}),
            ComponentRecipe(
                "sac_entropy", {"initial_alpha": 0.2, "target_entropy": None}
            ),
        ),
        returns=ComponentRecipe("td0", {"gamma": 0.99}),
        experience=ComponentRecipe("replay", {"capacity": 100_000, "batch_size": 256}),
        update=ComponentRecipe(
            "off_policy_update",
            {"learning_rate": 3e-4, "max_grad_norm": 10.0, "amp_dtype": None},
        ),
        target_update=ComponentRecipe("soft", {"tau": 0.005}),
    )


class MASAC(BaseMARLAlgorithm):
    def __init__(
        self,
        spec: EnvironmentSpec,
        policy: IndependentGaussianPolicy,
        critics: TwinIndependentCentralizedCritics,
        td_loss: TDLossObjective,
        entropy_objective: SACEntropyObjective,
        return_estimator: TD0Estimator,
        compiled_recipe: CompiledRecipe,
    ) -> None:
        super().__init__(spec)
        self.policy = policy
        self.critics = critics
        self.target_critics = deepcopy(critics).requires_grad_(False)
        self.td_loss = td_loss
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
    ) -> MASAC:
        parts = assemble_algorithm_components("masac", spec, recipe, registry)
        if not isinstance(parts.policy, IndependentGaussianPolicy):
            raise TypeError("MASAC policy 必须是 IndependentGaussianPolicy")
        if not isinstance(parts.critic, TwinIndependentCentralizedCritics):
            raise TypeError("MASAC critic 必须是 TwinIndependentCentralizedCritics")
        if not isinstance(parts.returns, TD0Estimator):
            raise TypeError("MASAC returns 必须是 TD0Estimator")
        return cls(
            spec,
            parts.policy,
            parts.critic,
            cast(TDLossObjective, require_one(parts.objectives, TDLossObjective, "MASAC")),
            cast(
                SACEntropyObjective,
                require_one(parts.objectives, SACEntropyObjective, "MASAC"),
            ),
            parts.returns,
            parts.compiled,
        )

    @staticmethod
    def _critic_input(observations: Tensor, actions: Tensor) -> Tensor:
        return torch.cat((observations.flatten(-2), actions.flatten(-2)), dim=-1)

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
            raise ValueError("MASAC 训练需要 actions/rewards/next_observations")
        terminated, _ = batch.terminal_flags()
        q1, q2 = self.critics(
            self._critic_input(batch.observations, batch.actions)
        )
        with torch.no_grad():
            next_output = self.policy.act(batch.next_observations)
            assert next_output.log_prob is not None
            next_q1, next_q2 = self.target_critics(
                self._critic_input(batch.next_observations, next_output.actions)
            )
            soft_next = (
                torch.minimum(next_q1, next_q2)
                - self.entropy_objective.alpha * next_output.log_prob
            )
            target_q = self.return_estimator.estimate(
                batch.rewards, soft_next, terminated
            )
        first_result = self.td_loss(q1, target_q)
        second_result = self.td_loss(q2, target_q)
        critic_loss = first_result.loss + second_result.loss
        critic_result = ObjectiveResult(
            critic_loss,
            {"critic_loss": critic_loss.detach()},
        )

        policy_output = self.policy.act(batch.observations)
        assert policy_output.log_prob is not None
        q_terms = []
        for index in range(self.spec.num_agents):
            joint_actions = torch.stack(
                [
                    policy_output.actions[..., other, :]
                    if other == index
                    else policy_output.actions[..., other, :].detach()
                    for other in range(self.spec.num_agents)
                ],
                dim=-2,
            )
            policy_input = self._critic_input(batch.observations, joint_actions)
            first = self.critics.first.critics[index]
            second = self.critics.second.critics[index]
            with frozen_parameters(first), frozen_parameters(second):
                q_terms.append(
                    torch.minimum(first(policy_input), second(policy_input)).squeeze(-1)
                )
        min_q = torch.stack(q_terms, dim=-1)
        actor_result = self.entropy_objective.actor(
            policy_output.log_prob, min_q
        )
        alpha_result = self.entropy_objective.temperature(
            policy_output.log_prob
        )
        return LossBundle.combine((critic_result, actor_result, alpha_result))

    def target_pairs(self) -> tuple[tuple[nn.Module, nn.Module], ...]:
        return ((self.target_critics, self.critics),)
