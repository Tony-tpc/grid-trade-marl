from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from marl.algorithms.base import AlgorithmConfig, BaseMARLAlgorithm
from marl.core.batch import MARLBatch
from marl.models.mlp import MLPBackbone
from marl.modules.mixer import QMixer


@dataclass(frozen=True, slots=True)
class QMIXConfig(AlgorithmConfig):
    """QMIX 的环境维度以及 agent/mixing 网络尺寸。"""

    num_agents: int = 2
    observation_dim: int = 16
    action_dim: int = 5
    # S：训练阶段可用的全局环境状态维度，不等同于单体局部观测 O。
    state_dim: int = 32
    hidden_dim: int = 128
    # Mixer 中间层宽度；由全局 state 生成这层的权重。
    mixing_dim: int = 32


class AgentQNetwork(nn.Module):
    """所有同构 agent 共享的局部 Q 网络。

    对每个局部观测输出 A 个 Q 值，因此 ``[B,N,O] -> [B,N,A]``。虽然网络参数共享，
    不同智能体仍输入自己的观测并得到自己的动作价值。
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = MLPBackbone(observation_dim, output_dim=hidden_dim)
        self.q_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, observations: Tensor) -> Tensor:
        # MLPBackbone 会保留 B/N 前置维；Linear 只作用于最后一个 feature 维。
        return cast(Tensor, self.q_head(self.backbone.forward(observations).features))


class QMIX(BaseMARLAlgorithm):
    """局部 Q 网络、单调 Mixer、Double-Q 目标组成的 QMIX。

    执行阶段，每个智能体只需对自己的 ``Q_i(o_i, a_i)`` 取 argmax。训练阶段，Mixer
    使用全局 state 把 N 个个体 Q 合成为团队 ``Q_tot``。Mixer 权重被约束为非负，保证
    ``Q_tot`` 对每个 ``Q_i`` 单调递增，因此分别贪心选择个体动作也能最大化团队 Q。

    QMIX 适合合作式离散动作任务；当前实现用共享 agent network 和团队平均奖励。
    """

    def __init__(self, config: QMIXConfig) -> None:
        super().__init__(config)
        self.agent_q_network = AgentQNetwork(
            config.observation_dim, config.action_dim, config.hidden_dim
        )
        self.mixer = QMixer(config.num_agents, config.state_dim, config.mixing_dim)
        # 目标 agent 网络和目标 mixer 一起构成稳定的 bootstrap 目标。
        self.target_agent_q_network = deepcopy(self.agent_q_network).requires_grad_(False)
        self.target_mixer = deepcopy(self.mixer).requires_grad_(False)

    @property
    def cfg(self) -> QMIXConfig:
        return self.config  # type: ignore[return-value]

    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
        **kwargs: Tensor,
    ) -> Tensor:
        # q_values: [B,N,A]；argmax 后为每个 agent 得到一个离散动作 [B,N]。
        q_values = self.agent_q_network.forward(observations)
        if action_mask is not None:
            # 非法动作设为当前 dtype 的最小值，确保 argmax 永远不会选中它。
            q_values = q_values.masked_fill(~action_mask.bool(), torch.finfo(q_values.dtype).min)
        return cast(Tensor, q_values.argmax(dim=-1))

    def compute_loss(self, batch: MARLBatch) -> dict[str, Tensor]:
        """计算团队 Q_tot 的一步 Double-Q TD loss。"""

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
        if not torch.allclose(batch.rewards, batch.rewards[..., :1].expand_as(batch.rewards)):
            raise ValueError("QMIX 仅适用于共享团队奖励，不能直接使用各智能体不同的 r_i")
        # ----- 1. 从每个 agent 的 A 个 Q 中取出 replay 中实际执行动作的 Q -----
        all_q = self.agent_q_network.forward(batch.observations)
        # actions.unsqueeze(-1): [B,N] -> [B,N,1]，供 gather 按最后一维索引。
        chosen_q = all_q.gather(-1, batch.actions.long().unsqueeze(-1)).squeeze(-1)
        # Mixer: N 个个体值 [B,N] + 全局 state [B,S] -> 团队值 [B,1]。
        total_q = self.mixer(chosen_q, batch.state).squeeze(-1)

        # ----- 2. Double-Q：在线网络选下一动作，目标网络评价该动作 -----
        with torch.no_grad():
            online_next_q = self.agent_q_network.forward(batch.next_observations)
            if batch.next_action_mask is not None:
                online_next_q = online_next_q.masked_fill(
                    ~batch.next_action_mask.bool(), torch.finfo(online_next_q.dtype).min
                )
            next_actions = online_next_q.argmax(dim=-1, keepdim=True)
            target_next_q = self.target_agent_q_network.forward(batch.next_observations)
            chosen_next_q = target_next_q.gather(-1, next_actions).squeeze(-1)
            # 目标 Mixer 必须配目标 agent Q 使用，得到稳定的下一状态团队价值。
            total_next_q = self.target_mixer(chosen_next_q, batch.next_state).squeeze(-1)
            dones = batch.dones if batch.dones is not None else torch.zeros_like(batch.rewards)
            reward, done = batch.rewards[..., 0], dones.float().amax(dim=-1)
            # 一步 Bellman target；episode 终止后不再引入下一状态价值。
            td_target = reward + self.cfg.gamma * (1.0 - done) * total_next_q

        # ----- 3. 用均方误差同时训练 agent network 与 Mixer -----
        td_error = total_q - td_target
        loss = F.mse_loss(total_q, td_target)
        return {"loss": loss, "td_loss": loss, "mean_abs_td_error": td_error.abs().mean()}

    def update_targets(self, hard: bool = False) -> None:
        """默认每步软更新；传 hard=True 可完整复制在线网络参数。"""
        updater = self.hard_update if hard else self.soft_update
        updater(self.target_agent_q_network, self.agent_q_network)
        updater(self.target_mixer, self.mixer)
