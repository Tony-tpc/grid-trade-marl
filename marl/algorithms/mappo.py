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
from marl.value_scaling import TargetScale, agent_statistics, value_diagnostics


@dataclass(frozen=True, slots=True)
class MAPPOLossConfig:
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    normalize_targets: bool = False

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
    """组合 PPO 目标并拥有 actor→critic 多 epoch 更新顺序。

    trainer 只负责采集/GAE/checkpoint。act 只返回动作供环境执行，sample 额外
    返回 old log-prob/value 供 rollout 保存；两者不是两套策略实现。
    """

    def __init__(self, spec: EnvironmentSpec, config: MAPPOConfig) -> None:
        super().__init__(spec)
        config.validate(spec)
        self.config = config
        self.policy = config.policy.build(spec)
        self.critic = config.critic.build(spec)
        self.policy_objective = PPOClipObjective(config.loss.clip_ratio)
        self.value_objective = ValueMSEObjective(config.loss.value_coefficient)
        self.entropy_objective = EntropyObjective(config.loss.entropy_coefficient)
        self.target_scale = TargetScale(spec.num_agents, config.loss.normalize_targets)

    def values(self, observations: Tensor, state: Tensor | None = None) -> Tensor:
        """state [...,S] 优先，否则展平联合观测；输出始终保留 [...,N]。"""

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
        bundle = LossBundle.combine(
            (
                self.policy_objective(
                    output.log_prob,
                    batch.extras["old_log_prob"],
                    batch.extras["advantages"],
                ),
                self.entropy_objective(output.entropy),
            )
        )
        return LossBundle(bundle.total, {
            **bundle.terms, **agent_statistics("entropy", output.entropy),
        })

    def compute_value_loss_bundle(self, batch: MARLBatch) -> LossBundle:
        """只构建 critic 计算图；可选使用 rollout 保存的 old value 做 clipping。"""

        self._validate_training_batch(batch)
        values = self.values(batch.observations, batch.state)
        clip_ratio = self.config.update.value_clip_ratio
        if clip_ratio is None:
            result = self.value_objective(
                self.target_scale(values), self.target_scale(batch.extras["returns"])
            )
        else:
            result = ClippedValueObjective(
                self.value_objective.coefficient, clip_ratio
            )(values, batch.extras["old_values"], batch.extras["returns"],
              scale=self.target_scale.scale().to(values) if self.target_scale.enabled else None)
        bundle = LossBundle.combine((result,))
        return LossBundle(bundle.total, {
            **bundle.terms, **value_diagnostics(values, batch.extras["returns"]),
        })

    def update(
        self, experience: PreparedRollout, runtime: OptimizerRuntime
    ) -> dict[str, float]:
        """在 fresh rollout 上按 policy→critic 执行 PPO 多轮 mini-batch 更新。"""

        # 在修改 target 统计之前拒绝已消费对象，失败调用不能改变算法状态。
        if experience.consumed:
            raise RuntimeError("on-policy rollout 已消费，不能重复用于参数更新")
        device = next(self.parameters()).device
        # 一份 fresh rollout 仅更新一次，所有 epoch 使用固定统计。
        self.target_scale.update(experience.batch.extras["returns"])
        totals: dict[str, Tensor] = {}
        gradient_totals: dict[str, Tensor] = {}
        mini_batch_count = 0
        sample_count = 0
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
            runtime.record_optimizer_step("actor", actor_gradient_norm)
            value_bundle = self.compute_value_loss_bundle(mini_batch)
            critic_gradient_norm = self.optimize(
                runtime.optimizer("critic"),
                value_bundle,
                runtime.max_grad_norm("critic"),
                parameters=runtime.parameters("critic"),
            )
            runtime.record_optimizer_step("critic", critic_gradient_norm)
            metrics = {
                **policy_bundle.terms,
                **value_bundle.terms,
                "loss": policy_bundle.total.detach() + value_bundle.total.detach(),
            }
            gradient_metrics = {
                "actor_gradient_norm": actor_gradient_norm,
                "critic_gradient_norm": critic_gradient_norm,
                "gradient_norm": torch.maximum(
                    actor_gradient_norm, critic_gradient_norm
                ),
            }
            # 尾 batch 常比其余 batch 小，等权平均会夸大它对 loss/entropy/KL 的影响。
            # 此处只修正日志权重；每个 mini-batch 的反向传播和 step 完全不变。
            size = mini_batch.observations.shape[0]
            runtime.accumulate_metrics(totals, metrics, weight=size)
            runtime.accumulate_metrics(gradient_totals, gradient_metrics)
            mini_batch_count += 1
            sample_count += size
        if mini_batch_count == 0:
            raise RuntimeError("MAPPO update 未产生任何 mini-batch")
        runtime.finish()
        averages = runtime.mean_metrics(totals, sample_count)
        averages.update(runtime.mean_metrics(gradient_totals, mini_batch_count))
        batch = experience.batch
        # EV/RMSE 是非线性统计；训练 mini-batch 的均值不能当作整批拟合优度。
        # rollout_* 从整批 old value/return 重新计算，衡量更新前的价值估计。
        averages.update({
            f"rollout_{name}": value.to(device)
            for name, value in value_diagnostics(
                batch.extras["old_values"], batch.extras["returns"]
            ).items()
        })
        if batch.rewards is not None:
            averages.update(agent_statistics("reward", batch.rewards))
        return runtime.export_metrics(averages)
