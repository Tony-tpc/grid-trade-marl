"""由通用策略、价值函数、目标和训练组件组装的 MAPPO。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from marl.algorithms.base import BaseMARLAlgorithm
from marl.core import MARLBatch, MARLModelOutput
from marl.envs.base import ActionKind, EnvironmentSpec
from marl.extensions import Buildable
from marl.modules.critic import CentralizedValueConfig, ValueNetwork
from marl.modules.policy import DiscretePolicy, IndependentDiscreteConfig
from marl.objectives import (
    ClippedValueObjective,
    EntropyObjective,
    LossBundle,
    PPOClipObjective,
    ValueMSEObjective,
)
from marl.returns import GAEConfig
from marl.training.on_policy import PPOUpdateConfig, PreparedRollout, RolloutConfig
from marl.training.optimization import OptimizerRuntime


@dataclass(frozen=True, slots=True)
class MAPPOLossConfig:
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01

    def __post_init__(self) -> None:
        PPOClipObjective(self.clip_ratio)
        ValueMSEObjective(self.value_coefficient)
        EntropyObjective(self.entropy_coefficient)


@dataclass(frozen=True, slots=True)
class MAPPOConfig:
    """MAPPO 配置；环境尺寸由 spec 注入。"""

    schema_version: Literal[1] = 1
    algorithm: Literal["mappo"] = "mappo"
    policy: Buildable[DiscretePolicy] = IndependentDiscreteConfig()
    critic: Buildable[ValueNetwork] = CentralizedValueConfig()
    loss: MAPPOLossConfig = MAPPOLossConfig()
    advantage: GAEConfig = GAEConfig()
    rollout: RolloutConfig = RolloutConfig()
    update: PPOUpdateConfig = PPOUpdateConfig()


    def validate(self, spec: EnvironmentSpec) -> None:
        if self.schema_version != 1 or self.algorithm != "mappo":
            raise ValueError("MAPPO config 的 algorithm/schema_version 不匹配")
        if spec.action_kind != ActionKind.DISCRETE:
            raise ValueError("MAPPO 不支持当前动作类型")

    def build(self, spec: EnvironmentSpec) -> MAPPO:
        return MAPPO(spec, self)


class MAPPO(BaseMARLAlgorithm):
    """只负责 MAPPO 前向语义的薄装配类。"""

    def __init__(self, spec: EnvironmentSpec, config: MAPPOConfig) -> None:
        super().__init__(spec)
        config.validate(spec)
        self.config = config
        self.policy = config.policy.build(spec)
        self.critic = config.critic.build(spec)
        self.policy_objective = PPOClipObjective(config.loss.clip_ratio)
        self.value_objective = ValueMSEObjective(config.loss.value_coefficient)
        self.entropy_objective = EntropyObjective(config.loss.entropy_coefficient)

    def _critic_input(self, batch: MARLBatch) -> Tensor:
        return batch.state if batch.state is not None else batch.observations.flatten(-2)

    def values(self, observations: Tensor, state: Tensor | None = None) -> Tensor:
        critic_input = state if state is not None else observations.flatten(-2)
        return self.critic(critic_input)

    def sample(
        self,
        observations: Tensor,
        state: Tensor | None = None,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
    ) -> MARLModelOutput:
        output = self.policy.act(
            observations,
            deterministic=deterministic,
            action_mask=action_mask,
        )
        output.values = self.values(observations, state)
        return output

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
        policy = self.compute_policy_loss_bundle(batch)
        value = self.compute_value_loss_bundle(batch)
        terms = {**policy.terms, **value.terms}
        terms["loss"] = policy.total + value.total
        return LossBundle(total=terms["loss"], terms=terms)

    def _validate_training_batch(self, batch: MARLBatch) -> None:
        batch.validate(self.spec.num_agents, self.spec.observation_dim)
        if batch.actions is None:
            raise ValueError("MAPPO 训练需要 actions")
        required = ("old_log_prob", "advantages", "returns")
        if any(name not in batch.extras for name in required):
            raise ValueError(f"MAPPO batch.extras 必须包含 {required}")
        expected_shape = batch.observations.shape[:-1]
        for name in required:
            if batch.extras[name].shape != expected_shape:
                raise ValueError(f"MAPPO {name} 必须具有逐智能体形状 {expected_shape}")

    def compute_policy_loss_bundle(self, batch: MARLBatch) -> LossBundle:
        """只构建 policy+entropy 计算图，供 actor optimizer 独立更新。"""

        self._validate_training_batch(batch)
        assert batch.actions is not None
        output = self.policy.evaluate(
            batch.observations,
            batch.actions,
            action_mask=batch.action_mask,
        )
        assert output.log_prob is not None and output.entropy is not None
        return LossBundle.combine(
            (
                self.policy_objective(
                    output.log_prob,
                    batch.extras["old_log_prob"],
                    batch.extras["advantages"],
                ),
                self.entropy_objective(output.entropy),
            )
        )

    def compute_value_loss_bundle(self, batch: MARLBatch) -> LossBundle:
        """只构建 critic 计算图；可选使用 rollout 保存的 old value 做 clipping。"""

        self._validate_training_batch(batch)
        values = self.critic(self._critic_input(batch))
        clip_ratio = self.config.update.value_clip_ratio
        if clip_ratio is None:
            result = self.value_objective(values, batch.extras["returns"])
        else:
            result = ClippedValueObjective(
                self.value_objective.coefficient, clip_ratio
            )(values, batch.extras["old_values"], batch.extras["returns"])
        return LossBundle.combine((result,))

    @staticmethod
    def _explained_variance(old_values: Tensor, returns: Tensor) -> Tensor:
        return_variance = returns.var(unbiased=False)
        if return_variance <= 1e-12:
            return torch.zeros((), dtype=returns.dtype, device=returns.device)
        return 1.0 - (returns - old_values).var(unbiased=False) / return_variance

    def update(
        self, experience: PreparedRollout, runtime: OptimizerRuntime
    ) -> dict[str, float]:
        """在 fresh rollout 上按 policy→critic 执行 PPO 多轮 mini-batch 更新。"""

        device = next(self.parameters()).device
        totals: dict[str, Tensor] = {}
        mini_batch_count = 0
        for mini_batch in experience.minibatches(
            epochs=self.config.update.epochs,
            mini_batch_size=self.config.update.mini_batch_size,
            generator=runtime.generator,
        ):
            policy_bundle = self.compute_policy_loss_bundle(mini_batch)
            actor_gradient_norm = self.optimize(
                runtime.optimizer("actor"),
                policy_bundle,
                runtime.max_grad_norm("actor"),
                parameters=runtime.parameters("actor"),
            )
            runtime.record_optimizer_step()
            value_bundle = self.compute_value_loss_bundle(mini_batch)
            critic_gradient_norm = self.optimize(
                runtime.optimizer("critic"),
                value_bundle,
                runtime.max_grad_norm("critic"),
                parameters=runtime.parameters("critic"),
            )
            runtime.record_optimizer_step()
            metrics = {
                **policy_bundle.terms,
                **value_bundle.terms,
                "loss": policy_bundle.total.detach() + value_bundle.total.detach(),
                "actor_gradient_norm": actor_gradient_norm,
                "critic_gradient_norm": critic_gradient_norm,
                "gradient_norm": torch.maximum(
                    actor_gradient_norm, critic_gradient_norm
                ),
            }
            for name, value in metrics.items():
                totals[name] = totals.get(name, torch.zeros_like(value)) + value.detach()
            mini_batch_count += 1
        if mini_batch_count == 0:
            raise RuntimeError("MAPPO update 未产生任何 mini-batch")
        runtime.finish()
        averages = {name: value / mini_batch_count for name, value in totals.items()}
        batch = experience.batch
        averages["explained_variance"] = self._explained_variance(
            batch.extras["old_values"], batch.extras["returns"]
        ).to(device)
        return runtime.export_metrics(averages)
