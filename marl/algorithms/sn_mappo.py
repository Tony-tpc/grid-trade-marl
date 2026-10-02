"""修订 SN-MAPPO：组合已有 MAPPO 角色和高阶响应，不复制 PPO/GAE 实现。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch import Tensor

from marl.algorithms.base import BaseMARLAlgorithm
from marl.algorithms.mappo import MAPPO, MAPPOConfig, MAPPOLossConfig
from marl.core import MARLBatch
from marl.envs.base import ActionKind, EnvironmentSpec
from marl.envs.sequential import SequentialSpec
from marl.models import GRUBackboneConfig
from marl.modules.critic import CentralizedValueConfig
from marl.modules.policy import IndependentGaussianConfig
from marl.objectives import LossBundle
from marl.training.implicit import (
    ImplicitResponseConfig,
    dice_objective,
    flat_gradient,
    gradient_surrogate,
    implicit_response_gradient,
    leave_one_out_baseline,
)
from marl.training.on_policy import PPOUpdateConfig
from marl.training.optimization import OptimizerRuntime
from marl.training.sequential import Phase, SequentialRollout

_ROLE = MAPPOConfig(
    policy=IndependentGaussianConfig(backbone=GRUBackboneConfig(hidden_dim=32)),
    critic=CentralizedValueConfig(backbone=GRUBackboneConfig(hidden_dim=32)),
    loss=MAPPOLossConfig(clip_ratio=.15, entropy_coefficient=.001),
    update=PPOUpdateConfig(epochs=4, mini_batch_size=96, sequence_length=32, max_grad_norm=1.0),
)


@dataclass(frozen=True, slots=True)
class SNMAPPOConfig:
    """各角色保留完整 typed MAPPOConfig；角色尺寸只由 SequentialSpec 注入。"""
    schema_version: Literal[2] = 2
    algorithm: Literal['sn_mappo'] = 'sn_mappo'
    leader: MAPPOConfig = replace(
        _ROLE, policy=IndependentGaussianConfig(
            backbone=GRUBackboneConfig(hidden_dim=32, num_layers=2)
        ),
        update=replace(_ROLE.update, actor_learning_rate=1e-4),
    )
    coordinator: MAPPOConfig = _ROLE
    followers: MAPPOConfig = _ROLE
    implicit: bool = True
    response: ImplicitResponseConfig = ImplicitResponseConfig()
    kl_coefficient: float = .01
    response_trajectories: int = 8
    response_baseline: Literal['leave_one_out', 'none'] = 'leave_one_out'

    def __post_init__(self) -> None:
        """绑定并验证本实例所负责的组件与状态。

        Inputs:
            无显式参数；读取当前实例的配置或训练状态。

        Returns:
            None；非法配置在构造时抛 ValueError，不修改训练状态。
        """
        if self.algorithm != 'sn_mappo' or self.schema_version != 2:
            raise ValueError('SN-MAPPO 标识/版本错误')
        if self.kl_coefficient < 0:
            raise ValueError('KL 系数必须非负')
        if type(self.response_trajectories) is not int or self.response_trajectories < 1:
            raise ValueError('response_trajectories 必须为正整数')
        if self.response_baseline not in ('leave_one_out', 'none'):
            raise ValueError('未知 response_baseline')
        if self.response_baseline == 'leave_one_out' and self.response_trajectories < 2:
            raise ValueError('leave_one_out baseline 至少需要两条独立轨迹')
        if any(c.loss.normalize_targets for c in (self.leader, self.coordinator, self.followers)):
            raise ValueError('双阶段初版使用固定训练尺度，不启用在线 target normalization')
        if not self.leader.advantage == self.coordinator.advantage == self.followers.advantage:
            raise ValueError('当前共享 collector 要求三个角色使用相同 GAE 配置')

    def build(self, spec: EnvironmentSpec | SequentialSpec) -> SNMAPPO:
        """SNMAPPO 实例。

        Args:
            spec: 环境给出的联合/角色规格；包含 N/O/A/S 和完整事件 horizon。

        Returns:
            SNMAPPO 实例；没有创建 optimizer 或开始采样。
        """
        if not isinstance(spec, SequentialSpec):
            raise TypeError('SN-MAPPO 必须提供环境的 SequentialSpec')
        return SNMAPPO(spec, self)


class SNMAPPO(BaseMARLAlgorithm):
    """DS 更新 UC/共享计划，DN 更新个体 decoder；两个阶段都只消费新鲜轨迹。

    DS 的直接梯度、Hessian、交叉导数统一来自归一化折扣经济回报 + KL 的 DiCE
    目标。阻尼定义局部正则化响应近似；记录驻点梯度范数，不声称已达到 best response。
    共享计划是唯一求响应的参数组，decoder 固定；这不是完整消费者 Nash 均衡导数。
    DS 全轨迹单次 actor 更新，DN 复用 MAPPO 多 epoch 更新；PPO clip/entropy 仅用于 DN。
    """
    def __init__(self, spec: SequentialSpec, config: SNMAPPOConfig):
        """绑定并验证本实例所负责的组件与状态。

        Args:
            spec: 环境给出的联合/角色规格；包含 N/O/A/S 和完整事件 horizon。
            config: 冻结算法/求解配置，控制网络或 damping、残差阈值及迭代上限。

        Returns:
            None；绑定组件/状态，不开始环境交互或参数更新。
        """
        super().__init__(spec.joint)
        if spec.joint.action_kind != ActionKind.CONTINUOUS or spec.leader.num_agents != 1:
            raise ValueError('当前顺序拓扑要求一个连续 leader')
        if spec.followers.observation_dim != spec.joint.observation_dim+spec.coordinator.action_dim:
            raise ValueError('follower 输入必须包含广播计划')
        self.sequential_spec, self.config = spec, config
        if any(c.rollout.horizon not in (None, spec.joint.horizon)
               for c in (config.leader, config.coordinator, config.followers)):
            raise ValueError('顺序角色必须采集完整事件 horizon')
        self.leader = MAPPO(spec.leader, config.leader)
        self.coordinator = MAPPO(spec.coordinator, config.coordinator)
        self.followers = MAPPO(spec.followers, config.followers)
        self.versions = (0, 0, 0)
        self.next_phase: Phase = 'leader'

    def act(self, observations: Tensor, *, deterministic: bool = False,
            action_mask: Tensor | None = None, **kwargs: Tensor) -> Tensor:
        """不返回。

        Args:
            observations: 联合观测 [...,N,O]；顺序算法要求分阶段采样。
            deterministic: 是否确定性动作；本联合 act 入口不执行采样。
            action_mask: 连续动作只能为 None。
            kwargs: 基类动作接口保留的关键字；此入口不使用。

        Returns:
            不返回；抛 RuntimeError，必须使用顺序角色 sample。
        """
        raise RuntimeError('顺序博弈必须分阶段 sample，不存在同时执行的联合 act')

    def compute_loss_bundle(self, batch: MARLBatch) -> LossBundle:
        """不返回。

        Args:
            batch: 角色 MARLBatch；循环 [B,T,N,O]，否则 [T*B,N,O]，extras 含旧概率/GAE/raw_actions。

        Returns:
            不返回；抛 RuntimeError，角色 loss 和阶段更新有不同数据契约。
        """
        raise RuntimeError('使用角色 loss 或带版本/阶段的 SequentialRollout.update')

    def response_objectives(self, batches: tuple[MARLBatch, MARLBatch, MARLBatch]
                            ) -> tuple[LossBundle, LossBundle]:
        """同一 fresh 批次构造上下层目标；直接项和隐式项共用这些图。

        Args:
            batches: UC/coordinator/consumer 数据，GRU 为 [B,T,N,O]，MLP 为
                按完整 episode 排列的 [B*T,N,O]。每条轨迹的奖励、动作和 old logp 已 detach。

        Returns:
            (upper, lower) LossBundle；经济回报除以 sum(gamma**t)，KL 对时间取均值。
            baseline 只用同场景其他独立轨迹的奖励，不使用本轨迹 GAE/critic。
        """
        horizon = self.sequential_spec.joint.horizon
        models = (self.leader, self.coordinator, self.followers)
        probabilities, penalties = [], []
        for model, batch in zip(models, batches, strict=True):
            model._validate_training_batch(batch)
            assert batch.actions is not None
            output = model.policy.evaluate(batch.observations, batch.actions,
                                           hidden_state=batch.policy_state,
                                           raw_actions=batch.extras['raw_actions'])
            assert output.log_prob is not None
            probabilities.append(output.log_prob.reshape(-1, horizon, model.spec.num_agents))
            ratio = output.log_prob-batch.extras['old_log_prob']
            penalties.append((ratio.exp()-1-ratio).mean())
        joint = torch.cat(probabilities, -1)
        assert batches[0].rewards is not None and batches[2].rewards is not None
        rewards = (batches[0].rewards.reshape(-1, horizon),
                   batches[2].rewards.sum(-1).reshape(-1, horizon))
        gamma = self.config.leader.advantage.gamma
        normalizer = (gamma**torch.arange(horizon, device=joint.device, dtype=joint.dtype)).sum()

        def objective(reward: Tensor, kl: Tensor) -> LossBundle:
            """保留逐轨迹因果依赖，构造最小化目标。

            Args:
                reward: detached 经济奖励 [B,T]；下层已显式求消费者总收益。
                kl: 该角色策略的标量 KL 正则，保留参数计算图。

            Returns:
                LossBundle，包含经济目标、KL 与总目标；不更新参数。
            """
            baseline = (leave_one_out_baseline(reward)
                        if self.config.response_baseline == 'leave_one_out' else None)
            economic = -dice_objective(joint, reward, gamma, baseline=baseline)/normalizer
            total = economic+self.config.kl_coefficient*kl
            return LossBundle(total, {'policy_loss': economic, 'kl_penalty': kl, 'loss': total})

        return objective(rewards[0], penalties[0]), objective(rewards[1], penalties[1])

    def update(self, rollout: SequentialRollout,
               runtimes: tuple[OptimizerRuntime, OptimizerRuntime, OptimizerRuntime]
               ) -> dict[str, float]:
        """命名 float 指标。

        Args:
            rollout: 包含三个角色 fresh PreparedRollout、阶段和策略版本的 SequentialRollout。
            runtimes: leader/coordinator/followers 的三个 OptimizerRuntime，参数集合互不重叠。

        Returns:
            命名 float 指标；成功消费新鲜阶段数据并推进策略版本/下一阶段。
        """
        if rollout.phase != self.next_phase or rollout.versions != self.versions:
            raise RuntimeError('拒绝阶段错误或旧策略版本的 rollout')
        prepared = (rollout.leader, rollout.coordinator, rollout.followers)
        if any(r.consumed for r in prepared):
            raise RuntimeError('阶段数据已消费')
        if rollout.phase == 'followers':
            rollout.leader.consume()
            rollout.coordinator.consume()
            metrics = self.followers.update(rollout.followers, runtimes[2])
            self.versions = (self.versions[0], self.versions[1], self.versions[2]+1)
            self.next_phase = 'leader'
            return metrics

        if rollout.physical_steps != self.config.response_trajectories*self.spec.horizon:
            raise ValueError('DS 轨迹数与 response_trajectories 不一致')
        batches = (prepared[0].consume(), prepared[1].consume(), prepared[2].consume())
        models = (self.leader, self.coordinator, self.followers)
        # cuDNN RNN 不提供所需的 double backward；CPU/普通 autograd 路径保留二阶图。
        with torch.backends.cudnn.flags(enabled=False):
            leader_bundle, coordinator_bundle = self.response_objectives(batches)
            leader_params = tuple(self.leader.policy.parameters())
            coordinator_params = tuple(self.coordinator.policy.parameters())
            leader_gradient = flat_gradient(leader_bundle.total, leader_params).detach()
            coordinator_gradient = flat_gradient(
                coordinator_bundle.total, coordinator_params
            ).detach()
            metrics = {'response_enabled': float(self.config.implicit), 'response_success': 1.0}
            metrics['direct_gradient_norm'] = float(leader_gradient.norm())
            metrics['lower_gradient_norm'] = float(coordinator_gradient.norm())
            metrics['response_trajectories'] = float(self.config.response_trajectories)
            if self.config.implicit:
                response = implicit_response_gradient(
                    leader_bundle.total, coordinator_bundle.total, leader_params,
                    coordinator_params, self.config.response,
                )
                metrics.update(response_success=float(response.converged),
                               response_residual=response.residual,
                               response_damping=response.damping,
                               response_iterations=float(response.iterations),
                               response_undamped_residual=response.undamped_residual)
                for i, attempt in enumerate(response.attempts):
                    metrics.update({f'response_attempt_{i}/{k}': float(v) for k, v in {
                        'damping': attempt.damping, 'residual': attempt.residual,
                        'iterations': attempt.iterations, 'status': attempt.status,
                    }.items()})
                if response.gradient is not None and response.correction is not None:
                    metrics['response_correction_norm'] = float(response.correction.norm())
                    leader_gradient = response.gradient
                else:
                    metrics['leader_skipped'] = 1.0
            # 两个 actor 的梯度在任何参数改动之前完整计算，阻止阶段内混入新参数。
            for i, (parameters, gradient, bundle) in enumerate((
                (leader_params, leader_gradient, leader_bundle),
                (coordinator_params, coordinator_gradient, coordinator_bundle),
            )):
                runtime = runtimes[i]
                if i != 0 or metrics['response_success']:
                    proxy = LossBundle(gradient_surrogate(parameters, gradient), bundle.terms)
                    norm = self.optimize(runtime.optimizer('actor'), proxy,
                                         runtime.max_grad_norm('actor'), parameters=parameters)
                    runtime.record_optimizer_step('actor', norm)
                value = models[i].compute_value_loss_bundle(batches[i])
                norm = self.optimize(runtime.optimizer('critic'), value,
                                     runtime.max_grad_norm('critic'))
                runtime.record_optimizer_step('critic', norm)
                runtime.finish()
                prefix = 'uc' if i == 0 else 'coordinator'
                terms = {f'{prefix}/actor/{k}': v for k, v in bundle.terms.items()}
                terms.update({f'{prefix}/critic/{k}': v for k, v in value.terms.items()})
                metrics.update({f'{prefix}/{k}': v
                                for k, v in runtime.export_metrics(terms).items()})
        self.versions = (self.versions[0]+int(metrics['response_success']), self.versions[1]+1,
                         self.versions[2])
        self.next_phase = 'followers'
        return metrics
