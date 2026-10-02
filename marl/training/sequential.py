"""顺序角色 rollout 采集与完整周期恢复；不包含算法名称分支或市场规则。"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch

from marl.core.recurrent import RecurrentState, map_state
from marl.envs.sequential import SequentialEnvironment, SequentialSpec
from marl.returns import AdvantageEstimator
from marl.training.checkpoint import (
    build_trainer_checkpoint_state,
    load_checkpoint_state,
    restore_torch_rng_state,
    save_checkpoint_state,
    validate_trainer_checkpoint_state,
)
from marl.training.on_policy import OnPolicyActorCritic, PreparedRollout, RolloutBuffer
from marl.training.optimization import OptimizerRuntime

Phase = Literal['leader', 'followers']


@dataclass(slots=True)
class SequentialRollout:
    """三个角色的 fresh 数据、生成时策略版本、阶段和真实物理步数。"""
    leader: PreparedRollout
    coordinator: PreparedRollout
    followers: PreparedRollout
    phase: Phase
    versions: tuple[int, int, int]
    physical_steps: int
    diagnostics: dict[str, float] = field(default_factory=dict)


class SequentialLearner(Protocol):
    @property
    def leader(self) -> OnPolicyActorCritic:
        """负责先行动作采样及 value 的角色模型。

        Inputs:
            无显式参数；读取当前实例的配置或训练状态。

        Returns:
            负责先行动作采样及 value 的角色模型。
        """
        ...

    @property
    def coordinator(self) -> OnPolicyActorCritic:
        """负责共享随机广播计划及聚合消费者 value 的角色模型。

        Inputs:
            无显式参数；读取当前实例的配置或训练状态。

        Returns:
            负责共享随机广播计划及聚合消费者 value 的角色模型。
        """
        ...

    @property
    def followers(self) -> OnPolicyActorCritic:
        """具有独立 decoder 和逐消费者 value 输出的角色模型。

        Inputs:
            无显式参数；读取当前实例的配置或训练状态。

        Returns:
            具有独立 decoder 和逐消费者 value 输出的角色模型。
        """
        ...
    sequential_spec: SequentialSpec
    versions: tuple[int, int, int]
    next_phase: Phase

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
        ...

    def state_dict(self) -> Mapping[str, Any]:
        """声明网络参数导出能力；完整训练快照由 trainer 另行组装。

        Inputs:
            无显式参数；读取当前实例的配置或训练状态。

        Returns:
            模型参数和 buffers 的映射，遵循 nn.Module.state_dict 契约。
        """
        ...

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> Any:
        """None/模块加载结果。

        Args:
            state_dict: 与当前网络键名和形状一致的模型参数映射。
            strict: 是否要求完整模型键名匹配。

        Returns:
            None/模块加载结果；失败时 trainer 回滚模型、optimizer 和 RNG。
        """
        ...


class SequentialTrainer:
    """每周期 DS→更新→丢弃→DN→更新；采样和 GAE 复用 RolloutBuffer。"""
    def __init__(self, environment: SequentialEnvironment, algorithm: SequentialLearner,
                 estimator: AdvantageEstimator,
                 runtimes: tuple[OptimizerRuntime, OptimizerRuntime, OptimizerRuntime],
                 *, device: str | torch.device = 'cpu',
                 leader_trajectories: int = 1,
                 config_data: Mapping[str, object] | None = None,
                 seed: int = 0):
        """绑定并验证本实例所负责的组件与状态。

        Args:
            environment: 满足 SequentialEnvironment 的环境；commit 不推进时间，settle 推进一次。
            algorithm: 满足 SequentialLearner 的角色模型与阶段更新接口。
            estimator: 共享的逐 agent GAE 估计器；三个角色的 GAE 配置必须一致。
            runtimes: leader/coordinator/followers 的三个 OptimizerRuntime，参数集合互不重叠。
            device: 采样/更新设备；rollout 原始物理数据保存于 CPU。
            leader_trajectories: DS 在相同外部 seed 下采集的独立动作轨迹数，正整数。
            config_data: 配置数据快照，恢复时必须严格相等；不会动态导入代码。
            seed: NumPy 日窗口/阶段采样随机种子；不改变已构造模型参数。

        Returns:
            None；绑定组件/状态，不开始环境交互或参数更新。
        """
        self.environment, self.algorithm, self.estimator = environment, algorithm, estimator
        self.runtimes = runtimes
        self.device = torch.device(device)
        if type(leader_trajectories) is not int or leader_trajectories < 1:
            raise ValueError('leader_trajectories 必须为正整数')
        self.leader_trajectories = leader_trajectories
        self.config_data = deepcopy(dict(config_data)) if config_data is not None else None
        self.rng = np.random.default_rng(seed)
        self.physical_steps, self.cycles = 0, 0
        self.input_spec = {**asdict(algorithm.sequential_spec),
                           'leader_trajectories': leader_trajectories,
                           'context': dict(environment.checkpoint_context)}

    def collect(self, phase: Phase, seed: int) -> SequentialRollout:
        """冻结策略采集 DS 多条或 DN 单条 fresh 轨迹，沿 B 合并，不拼接时间。

        Args:
            phase: 当前待更新的 DS/leader 或 DN/followers 阶段。
            seed: 本批外部场景 seed；各轨迹复用场景，策略 RNG 连续前进。
                环境 reset 不应重置全局策略 RNG。各条轨迹独立重置 h/c。

        Returns:
            带统一策略版本和单次消费权的 SequentialRollout；记录全部实际采样步数。
        """
        count = self.leader_trajectories if phase == 'leader' else 1
        rollouts = [self._collect_episode(phase, seed) for _ in range(count)]
        if count == 1:
            return rollouts[0]
        first = rollouts[0]
        return SequentialRollout(
            PreparedRollout.concatenate([r.leader for r in rollouts]),
            PreparedRollout.concatenate([r.coordinator for r in rollouts]),
            PreparedRollout.concatenate([r.followers for r in rollouts]),
            phase, first.versions, sum(r.physical_steps for r in rollouts),
            {k: sum(r.diagnostics[k] for r in rollouts)/count for k in first.diagnostics},
        )

    def _collect_episode(self, phase: Phase, seed: int) -> SequentialRollout:
        """带版本和单次消费权的 SequentialRollout。

        Args:
            phase: leader 对应 DS，followers 对应 DN；必须匹配当前待更新阶段。
            seed: NumPy 日窗口/阶段采样随机种子；不改变已构造模型参数。

        Returns:
            带版本和单次消费权的 SequentialRollout；物理步数增加，参数保持不变。
        """
        if phase != self.algorithm.next_phase:
            raise RuntimeError('采样阶段与待更新阶段不一致')
        spec = self.algorithm.sequential_spec
        models = (self.algorithm.leader, self.algorithm.coordinator, self.algorithm.followers)
        specs = (spec.leader, spec.coordinator, spec.followers)
        buffers = tuple(RolloutBuffer(s, horizon=s.horizon, num_envs=1) for s in specs)
        policy_states: list[RecurrentState] = [None, None, None]
        value_states: list[RecurrentState] = [None, None, None]
        step = self.environment.reset(seed)
        plan_error = 0.0
        for t in range(spec.joint.horizon):
            state = torch.tensor(step.state, device=self.device).unsqueeze(0)
            leader_obs = torch.tensor(step.observations[:1], device=self.device).unsqueeze(0)
            with torch.inference_mode():
                leader = models[0].sample(leader_obs, state, policy_state=policy_states[0],
                                          value_state=value_states[0])
            committed = self.environment.commit_leader(leader.actions[0, 0].cpu().numpy())
            committed_state = torch.tensor(committed.state, device=self.device).unsqueeze(0)
            public = torch.tensor(committed.observations[1:], device=self.device).unsqueeze(0)
            coordinator_obs = public.mean(-2, keepdim=True)
            with torch.inference_mode():
                plan = models[1].sample(coordinator_obs, committed_state,
                                       policy_state=policy_states[1], value_state=value_states[1])
                follower_obs = torch.cat([public, plan.actions.expand(-1, public.shape[1], -1)], -1)
                followers = models[2].sample(
                    follower_obs, committed_state,
                    policy_state=policy_states[2], value_state=value_states[2],
                )
            next_step = self.environment.settle(followers.actions[0].cpu().numpy())
            difference = plan.actions-followers.actions.mean(-2, keepdim=True)
            plan_error += float(difference.square().mean())
            reward = torch.tensor(next_step.rewards).unsqueeze(0)
            outputs = (leader, plan, followers)
            observations = (leader_obs, coordinator_obs, follower_obs)
            states = (state, committed_state, committed_state)
            rewards = (reward[:, :1], reward[:, 1:].sum(-1, keepdim=True), reward[:, 1:])
            for i, (buffer, s, output) in enumerate(zip(buffers, specs, outputs, strict=True)):
                assert output.log_prob is not None and output.values is not None
                buffer.add(
                    observations=observations[i].cpu(), states=states[i].cpu(),
                    actions=output.actions.cpu(), raw_actions=output.extras['raw_actions'].cpu(),
                    rewards=rewards[i], old_log_prob=output.log_prob, old_values=output.values,
                    terminated=torch.full((1, s.num_agents), next_step.terminated,
                                          dtype=torch.bool),
                    truncated=torch.full((1, s.num_agents), next_step.truncated, dtype=torch.bool),
                    policy_state=policy_states[i] if policy_states[i] is not None else
                    map_state(output.policy_state, torch.zeros_like),
                    value_state=value_states[i] if value_states[i] is not None else
                    map_state(output.value_state, torch.zeros_like),
                )
                policy_states[i], value_states[i] = output.policy_state, output.value_state
            step = next_step
            self.physical_steps += 1
            if step.done and t+1 != spec.joint.horizon:
                raise RuntimeError('顺序 collector 暂不支持提前终止/部分 reset')
        if not step.terminated:
            raise ValueError('当前顺序 collector 只接受完整终止事件；不能用零值处理截断')
        prepared = tuple(buffer.finish(torch.zeros(1, s.num_agents), self.estimator).to(self.device)
                         for buffer, s in zip(buffers, specs, strict=True))
        return SequentialRollout(prepared[0], prepared[1], prepared[2], phase,
                                 self.algorithm.versions, spec.joint.horizon,
                                 {'plan_request_mse': plan_error/spec.joint.horizon})

    def train_cycle(self) -> dict[str, float]:
        """DS 与 DN 的独立诊断。

        Inputs:
            无显式参数；读取当前实例的配置或训练状态。

        Returns:
            DS 与 DN 的独立诊断；周期数增加一次，所有采样步计入总预算。
        """
        if self.algorithm.next_phase != 'leader':
            raise RuntimeError('上个周期尚未完成')
        metrics = {}
        for phase in ('leader', 'followers'):
            rollout = self.collect(phase, int(self.rng.integers(0, 2**31-1)))
            for name, value in self.algorithm.update(rollout, self.runtimes).items():
                metrics[f'{phase}/{name}'] = value
            metrics.update({f'{phase}/{k}': v for k, v in rollout.diagnostics.items()})
        self.cycles += 1
        return {**metrics, 'physical_steps': float(self.physical_steps),
                'cycles': float(self.cycles)}

    def state_dict(self) -> dict[str, Any]:
        """完整周期 schema 5 快照。

        Inputs:
            无显式参数；读取当前实例的配置或训练状态。

        Returns:
            完整周期 schema 5 快照；阶段中间调用会拒绝，返回张量已复制。
        """
        if self.algorithm.next_phase != 'leader':
            raise RuntimeError('只允许完整双阶段周期边界 checkpoint')
        state = build_trainer_checkpoint_state(
            self.algorithm.state_dict(), self.config_data,
            {str(i): runtime.state_dict() for i, runtime in enumerate(self.runtimes)},
            input_spec=self.input_spec,
        )
        state.update(phase='leader', versions=self.algorithm.versions, cycles=self.cycles,
                     physical_steps=self.physical_steps,
                     numpy_rng=deepcopy(self.rng.bit_generator.state))
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """None/模块加载结果。

        Args:
            state: 完整周期 checkpoint 映射；含模型、三个优化器、策略版本、计数和 RNG。

        Returns:
            None/模块加载结果；失败时 trainer 回滚模型、optimizer 和 RNG。
        """
        if self.algorithm.next_phase != 'leader' or state.get('phase') != 'leader':
            raise ValueError('checkpoint 必须位于完整周期边界')
        versions = state.get('versions')
        if (not isinstance(versions, (list, tuple)) or len(versions) != 3
                or any(type(v) is not int or v < 0 for v in versions)):
            raise ValueError('checkpoint 策略版本必须是三个非负整数')
        for name in ('physical_steps', 'cycles'):
            if type(state.get(name)) is not int or state[name] < 0:
                raise ValueError(f'checkpoint {name} 必须为非负整数')
        validate_trainer_checkpoint_state(state, self.config_data,
                                         algorithm_state=self.algorithm.state_dict(),
                                         input_spec=self.input_spec)
        previous = self.state_dict()

        def restore(saved: Mapping[str, Any]) -> None:
            """绑定并验证本实例所负责的组件与状态。

            Args:
                saved: 通过预检的 checkpoint，或异常恢复用的事务前快照。

            Returns:
                None；原地恢复模型、三套 optimizer、NumPy/Torch RNG 及计数。
            """
            self.algorithm.load_state_dict(saved['algorithm'])
            for i, runtime in enumerate(self.runtimes):
                runtime.load_state_dict(saved['optimization'][str(i)])
            self.rng.bit_generator.state = deepcopy(saved['numpy_rng'])
            self.algorithm.versions = tuple(saved['versions'])
            self.cycles, self.physical_steps = saved['cycles'], saved['physical_steps']
            restore_torch_rng_state(saved)

        try:
            restore(state)
        except Exception:
            restore(previous)
            raise

    def save_checkpoint(self, path: str | Path) -> None:
        """绑定并验证本实例所负责的组件与状态。

        Args:
            path: 可信本地 checkpoint 路径；保存时创建父目录。

        Returns:
            None；写入包含模型、optimizer、RNG 的可信本地 checkpoint。
        """
        save_checkpoint_state(self.state_dict(), path)

    def load_checkpoint(self, path: str | Path) -> None:
        """绑定并验证本实例所负责的组件与状态。

        Args:
            path: 可信本地 checkpoint 路径；保存时创建父目录。

        Returns:
            None；预检配置/数据上下文后恢复完整周期状态。
        """
        self.load_state_dict(load_checkpoint_state(path, map_location=self.device))
