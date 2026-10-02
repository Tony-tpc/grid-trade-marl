"""顺序博弈的环境能力：先承诺后响应，一个 settle 才算物理步。"""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from marl.envs.base import EnvironmentSpec, EnvironmentStep, RewardStructure


@dataclass(frozen=True, slots=True)
class SequentialSpec:
    """四个角色/联合 EnvironmentSpec 的不可变组合；环境拥有尺寸定义，算法只消费规格。"""
    joint: EnvironmentSpec
    leader: EnvironmentSpec
    coordinator: EnvironmentSpec
    followers: EnvironmentSpec

    def __post_init__(self) -> None:
        """尽早校验角色布局、公共 state、事件长度和个体奖励，禁止隐式广播补齐。"""
        roles = (self.leader, self.coordinator, self.followers)
        if self.leader.num_agents != 1 or self.coordinator.num_agents != 1:
            raise ValueError('顺序布局要求单 leader 和单广播计划')
        if self.followers.num_agents != self.joint.num_agents-1:
            raise ValueError('followers 数量必须等于 joint 数量减一')
        if any(s.horizon != self.joint.horizon or s.state_dim != self.joint.state_dim
               for s in roles):
            raise ValueError('所有角色必须共享 horizon 和全局 state 尺寸')
        if any(s.reward_structure != RewardStructure.INDIVIDUAL for s in (*roles, self.joint)):
            raise ValueError('顺序个体博弈需要 INDIVIDUAL rewards')
        if self.leader.observation_dim != self.joint.observation_dim or (
            self.coordinator.observation_dim != self.joint.observation_dim
        ):
            raise ValueError('leader/coordinator 观测尺寸与 joint 不一致')
        if self.followers.observation_dim != self.joint.observation_dim+self.coordinator.action_dim:
            raise ValueError('follower 观测必须包含广播计划')


@runtime_checkable
class SequentialEnvironment(Protocol):
    """联合观测第 0 行为 leader，其余行为 followers；计划信号另行广播。"""
    @property
    def spec(self) -> EnvironmentSpec:
        """返回静态 EnvironmentSpec；智能体数、观测/动作/state 尺寸和 horizon 由当前环境决定。"""
        ...

    @property
    def sequential_spec(self) -> SequentialSpec:
        """返回联合、UC、随机计划和条件消费者规格；消费者输入追加广播计划维度。"""
        ...

    @property
    def checkpoint_context(self) -> Mapping[str, object]:
        """返回数据 SHA、时间窗口、训练尺度及物理配置，用于拒绝不同数据/环境的 checkpoint 恢复。"""
        ...

    def reset(self, seed: int | None = None) -> EnvironmentStep:
        """按 seed 选择完整日或重置固定事件，清空债务/累计购电和上次动作；返回统一初始观测。"""
        ...

    def commit_leader(self, action: np.ndarray) -> EnvironmentStep:
        """提交 [4] UC 请求，返回含公开承诺的响应阶段观测；不推进物理时间。"""
        ...

    def settle(self, actions: np.ndarray) -> EnvironmentStep:
        """在已承诺 UC 动作下清算消费者 [N,4] 请求，推进一次物理步，返回逐个体奖励及账本。"""
        ...
