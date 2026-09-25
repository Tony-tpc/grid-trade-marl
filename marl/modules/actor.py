from __future__ import annotations

from torch import Tensor, nn

from marl.models.base import BaseBackbone
from marl.modules.action_head import ActionHeadOutput, BaseActionHead, DiscreteActionHead


class Actor(nn.Module):
    """组合 Backbone 与动作头；算法只需决定用哪种组合。
    J(θ)=Es​[Qϕ​(s,πθ​(s))]
    在接受观测后使用自己的policy得到动作action
    """

    def __init__(self, backbone: BaseBackbone, action_head: BaseActionHead) -> None:
        super().__init__()
        self.backbone, self.action_head = backbone, action_head

    """
    Backbone（骨干网络）：负责从原始输入中提取基础特征，是模型的主体。
    Neck（颈部网络）：负责对 Backbone 提取的特征进行进一步融合、增强或多尺度处理。
    Head（任务头）：负责将特征
    """

    def forward(
        self,
        observations: Tensor,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
        hidden_state: Tensor | None = None,
        **backbone_kwargs: Tensor,
    ) -> ActionHeadOutput:
        """
        输出Action
        """
        encoded = self.backbone.forward(
            observations, hidden_state=hidden_state, **backbone_kwargs
        ) 
        result = self.action_head.forward(encoded.features, deterministic, action_mask)
        # 循环策略的状态放入分布参数，采样器无需依赖某个具体 Backbone 类型。
        if encoded.hidden_state is not None:
            result.distribution_params["hidden_state"] = encoded.hidden_state
        return result

    def discrete_logits(
        self,
        observations: Tensor,
        *,
        action_mask: Tensor | None = None,
        hidden_state: Tensor | None = None,
        **backbone_kwargs: Tensor,
    ) -> Tensor:
        """只计算离散策略参数，不采样动作或构造分布。"""
        if not isinstance(self.action_head, DiscreteActionHead):
            raise TypeError("discrete_logits 需要 DiscreteActionHead")
        encoded = self.backbone.forward(
            observations, hidden_state=hidden_state, **backbone_kwargs
        )
        return self.action_head.logits(encoded.features, action_mask)
