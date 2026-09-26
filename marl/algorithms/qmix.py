"""组合式离散合作任务 QMIX。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import Tensor, nn

from marl.algorithms.assembly import assemble_algorithm_components, require_one
from marl.algorithms.base import BaseMARLAlgorithm
from marl.core import MARLBatch
from marl.envs.base import ActionKind, EnvironmentSpec, RewardStructure
from marl.extensions import Buildable
from marl.modules.mixer import MixingNetwork, QMixer, QMixerConfig
from marl.modules.policy import LocalQPolicy, SharedDiscreteQConfig, SharedDiscreteQPolicy
from marl.objectives import LossBundle, ObjectiveResult, TDLossObjective
from marl.recipes import AlgorithmRecipe, CompiledRecipe, ComponentRecipe
from marl.registry import DEFAULT_COMPONENT_REGISTRY, ComponentRegistry
from marl.returns import TD0Config, TD0Estimator, ValueTargetEstimator
from marl.target_updates import HardTargetConfig, SoftTargetConfig
from marl.training.off_policy import OffPolicyUpdateConfig, ReplayConfig


@dataclass(frozen=True, slots=True)
class QMIXLossConfig:
    td_coefficient: float = 1.0

    def __post_init__(self) -> None:
        TDLossObjective(self.td_coefficient)


@dataclass(frozen=True, slots=True)
class QMIXConfig:
    """QMIX 配置；环境尺寸由 spec 注入。"""

    schema_version: Literal[1] = 1
    algorithm: Literal["qmix"] = "qmix"
    policy: Buildable[LocalQPolicy] = SharedDiscreteQConfig()
    mixer: Buildable[MixingNetwork] = QMixerConfig()
    loss: QMIXLossConfig = QMIXLossConfig()
    value_target: TD0Config = TD0Config()
    replay: ReplayConfig = ReplayConfig()
    update: OffPolicyUpdateConfig = OffPolicyUpdateConfig()
    target_update: SoftTargetConfig | HardTargetConfig = SoftTargetConfig()


    def build(self, spec: EnvironmentSpec) -> QMIX:
        """在此处直接查看 QMIX 的完整组件组合。"""
        if spec.action_kind != ActionKind.DISCRETE:
            raise ValueError("QMIX 不支持当前动作类型")
        if spec.reward_structure != RewardStructure.SHARED:
            raise ValueError("QMIX 要求共享团队奖励")
        return QMIX(
            spec, self.policy.build(spec), self.mixer.build(spec),
            TDLossObjective(self.loss.td_coefficient),
            self.value_target.build(),
        )


def default_qmix_recipe() -> AlgorithmRecipe:
    return AlgorithmRecipe(
        schema_version=1,
        algorithm="qmix",
        policy=ComponentRecipe("shared_discrete_q", {"hidden_dim": 128}),
        critic=ComponentRecipe("qmix_mixer", {"mixing_dim": 32}),
        objectives=(ComponentRecipe("td_mse", {"coefficient": 1.0}),),
        returns=ComponentRecipe("td0", {"gamma": 0.99}),
        experience=ComponentRecipe("replay", {"capacity": 100_000, "batch_size": 256}),
        update=ComponentRecipe(
            "off_policy_update",
            {"learning_rate": 3e-4, "max_grad_norm": 10.0, "amp_dtype": None},
        ),
        target_update=ComponentRecipe("soft", {"tau": 0.005}),
    )


class QMIX(BaseMARLAlgorithm):
    def __init__(
        self,
        spec: EnvironmentSpec,
        policy: LocalQPolicy,
        mixer: MixingNetwork,
        td_loss: TDLossObjective,
        return_estimator: ValueTargetEstimator,
        compiled_recipe: CompiledRecipe | None = None,
    ) -> None:
        super().__init__(spec)
        self.policy = policy
        self.mixer = mixer
        self.target_policy = deepcopy(policy)
        self.get_submodule("target_policy").requires_grad_(False)
        self.target_mixer = deepcopy(mixer)
        self.get_submodule("target_mixer").requires_grad_(False)
        self.td_loss = td_loss
        self.return_estimator = return_estimator
        self.compiled_recipe = compiled_recipe

    @classmethod
    def from_recipe(
        cls,
        spec: EnvironmentSpec,
        recipe: AlgorithmRecipe,
        *,
        registry: ComponentRegistry = DEFAULT_COMPONENT_REGISTRY,
    ) -> QMIX:
        parts = assemble_algorithm_components("qmix", spec, recipe, registry)
        if not isinstance(parts.policy, SharedDiscreteQPolicy):
            raise TypeError("QMIX policy 必须是 SharedDiscreteQPolicy")
        if not isinstance(parts.critic, QMixer):
            raise TypeError("QMIX critic 必须是 QMixer")
        if not isinstance(parts.returns, TD0Estimator):
            raise TypeError("QMIX returns 必须是 TD0Estimator")
        return cls(
            spec,
            parts.policy,
            parts.critic,
            cast(TDLossObjective, require_one(parts.objectives, TDLossObjective, "QMIX")),
            parts.returns,
            parts.compiled,
        )

    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool = True,
        action_mask: Tensor | None = None,
        **kwargs: Tensor,
    ) -> Tensor:
        return self.policy.act(
            observations,
            deterministic=True,
            action_mask=action_mask,
        ).actions

    def compute_loss_bundle(self, batch: MARLBatch) -> LossBundle:
        batch.validate(self.spec.num_agents, self.spec.observation_dim)
        if any(
            value is None
            for value in (
                batch.actions,
                batch.rewards,
                batch.next_observations,
                batch.state,
                batch.next_state,
            )
        ):
            raise ValueError("QMIX 需要 actions/rewards/next_observations/state/next_state")
        assert batch.actions is not None and batch.rewards is not None
        assert batch.next_observations is not None and batch.state is not None
        assert batch.next_state is not None
        if not torch.allclose(
            batch.rewards, batch.rewards[..., :1].expand_as(batch.rewards)
        ):
            raise ValueError("QMIX 仅适用于共享团队奖励")
        all_q = self.policy.q_values(batch.observations)
        chosen_q = all_q.gather(
            -1, batch.actions.long().unsqueeze(-1)
        ).squeeze(-1)
        total_q = self.mixer(chosen_q, batch.state).squeeze(-1)
        terminated, _ = batch.terminal_flags()
        with torch.no_grad():
            online_next_q = self.policy.q_values(batch.next_observations)
            if batch.next_action_mask is not None:
                online_next_q = online_next_q.masked_fill(
                    ~batch.next_action_mask.bool(),
                    torch.finfo(online_next_q.dtype).min,
                )
            next_actions = online_next_q.argmax(dim=-1, keepdim=True)
            target_next_q = self.target_policy.q_values(batch.next_observations)
            chosen_next_q = target_next_q.gather(
                -1, next_actions
            ).squeeze(-1)
            total_next_q = self.target_mixer(
                chosen_next_q, batch.next_state
            ).squeeze(-1)
            reward = batch.rewards[..., 0]
            team_terminated = terminated.any(dim=-1)
            td_target = self.return_estimator.estimate(
                reward, total_next_q, team_terminated
            )
        base = self.td_loss(total_q, td_target, name="td_loss")
        result = ObjectiveResult(
            base.loss,
            {
                "td_loss": base.metrics["td_loss"],
                "mean_abs_td_error": (total_q - td_target).abs().mean().detach(),
            },
        )
        return LossBundle.combine((result,))

    def target_pairs(self) -> tuple[tuple[nn.Module, nn.Module], ...]:
        return (
            (self.get_submodule("target_policy"), self.get_submodule("policy")),
            (self.get_submodule("target_mixer"), self.get_submodule("mixer")),
        )
