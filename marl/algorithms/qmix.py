"""组合式离散合作任务 QMIX。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from marl.algorithms.base import BaseMARLAlgorithm
from marl.core import MARLBatch
from marl.envs.base import ActionKind, EnvironmentSpec, RewardStructure
from marl.extensions import Buildable
from marl.modules.mixer import MixingNetwork, QMixerConfig
from marl.modules.policy import LocalQPolicy, SharedDiscreteQConfig
from marl.objectives import LossBundle, ObjectiveResult, TDLossObjective
from marl.returns import TD0Config, ValueTargetEstimator
from marl.target_updates import HardTargetConfig, SoftTargetConfig, frozen_target
from marl.training.off_policy import (
    OffPolicyUpdateConfig,
    ReplayConfig,
)
from marl.training.optimization import OptimizerRuntime


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


    def validate(self, spec: EnvironmentSpec) -> None:
        if self.schema_version != 1 or self.algorithm != "qmix":
            raise ValueError("QMIX config 的 algorithm/schema_version 不匹配")
        if spec.action_kind != ActionKind.DISCRETE:
            raise ValueError("QMIX 不支持当前动作类型")
        if spec.reward_structure != RewardStructure.SHARED:
            raise ValueError("QMIX 要求共享团队奖励")

    def build(self, spec: EnvironmentSpec) -> QMIX:
        return QMIX(spec, self)


class QMIX(BaseMARLAlgorithm):
    def __init__(self, spec: EnvironmentSpec, config: QMIXConfig) -> None:
        super().__init__(spec)
        config.validate(spec)
        self.config = config
        self.policy = config.policy.build(spec)
        self.mixer = config.mixer.build(spec)
        self.target_policy = frozen_target(self.policy)
        self.target_mixer = frozen_target(self.mixer)
        self.td_loss = TDLossObjective(config.loss.td_coefficient)
        self.return_estimator: ValueTargetEstimator = config.value_target.build()

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

    def update(
        self, batch: MARLBatch, runtime: OptimizerRuntime
    ) -> dict[str, float]:
        """QMIX 仅执行一次 value optimizer，再更新 target。"""

        with runtime.autocast(batch.observations.device):
            bundle = self.compute_loss_bundle(batch)
        value_norm = self.optimize(
            runtime.optimizer("value"),
            bundle,
            runtime.max_grad_norm("value"),
            parameters=runtime.parameters("value"),
        )
        runtime.record_optimizer_step()
        runtime.finish(self.target_pairs())
        metrics = dict(bundle.terms)
        metrics["gradient_norm"] = value_norm
        return runtime.export_metrics(metrics)

    def target_pairs(self) -> tuple[tuple[nn.Module, nn.Module], ...]:
        return (
            (self.get_submodule("target_policy"), self.get_submodule("policy")),
            (self.get_submodule("target_mixer"), self.get_submodule("mixer")),
        )
