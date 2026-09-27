from __future__ import annotations

from torch import Tensor, nn

from marl.core.recurrent import RecurrentState
from marl.models.base import BackboneOutput, BaseBackbone
from marl.modules.action_head import ActionHeadOutput, BaseActionHead, DiscreteActionHead


class Actor(nn.Module):
    """单智能体策略网络：``observation -> backbone -> action head``。

    Actor 不知道智能体数量和参数共享关系；这些多智能体拓扑由
    :mod:`marl.modules.policy` 负责。输入 ``[..., O]``，输出动作、log-prob、
    entropy 和分布参数；hidden_state 单独返回，不是分布参数。
    """

    def __init__(self, backbone: BaseBackbone, action_head: BaseActionHead) -> None:
        super().__init__()
        self.backbone, self.action_head = backbone, action_head

    @staticmethod
    def _attach_hidden_state(
        result: ActionHeadOutput, hidden_state: RecurrentState
    ) -> ActionHeadOutput:
        result.hidden_state = hidden_state
        return result

    def _encode(
        self, observations: Tensor, hidden_state: RecurrentState, **kwargs: Tensor,
    ) -> BackboneOutput:
        # 单步 [B,O] 显式补 T=1；训练序列 [B,T,O] 交给原生 RNN 一次执行。
        single_step = self.backbone.is_recurrent and observations.ndim == 2
        encoded: BackboneOutput = self.backbone(
            observations.unsqueeze(1) if single_step else observations,
            hidden_state=hidden_state, **kwargs,
        )
        if single_step:
            encoded.features = encoded.features.squeeze(1)
        return encoded

    def forward(
        self,
        observations: Tensor,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
        hidden_state: RecurrentState = None,
        **backbone_kwargs: Tensor,
    ) -> ActionHeadOutput:
        """采样或确定性选择一个动作。"""

        # 通过 Module.__call__ 保留 PyTorch hook/包装器；不要直接调用 forward。
        encoded = self._encode(observations, hidden_state, **backbone_kwargs)
        result: ActionHeadOutput = self.action_head(encoded.features, deterministic, action_mask)
        return self._attach_hidden_state(result, encoded.hidden_state)

    def evaluate_actions(
        self,
        observations: Tensor,
        actions: Tensor,
        *,
        action_mask: Tensor | None = None,
        hidden_state: RecurrentState = None,
        **backbone_kwargs: Tensor,
    ) -> ActionHeadOutput:
        """在同一 Actor 中评估给定动作，供 PPO 等 on-policy 目标使用。"""

        encoded = self._encode(observations, hidden_state, **backbone_kwargs)
        result = self.action_head.evaluate_actions(
            encoded.features, actions, action_mask
        )
        return self._attach_hidden_state(result, encoded.hidden_state)

    def discrete_logits(
        self,
        observations: Tensor,
        *,
        action_mask: Tensor | None = None,
        hidden_state: RecurrentState = None,
        **backbone_kwargs: Tensor,
    ) -> Tensor:
        """只计算离散策略参数，不采样动作或构造分布。"""
        if not isinstance(self.action_head, DiscreteActionHead):
            raise TypeError("discrete_logits 需要 DiscreteActionHead")
        encoded = self._encode(observations, hidden_state, **backbone_kwargs)
        return self.action_head.logits(encoded.features, action_mask)
