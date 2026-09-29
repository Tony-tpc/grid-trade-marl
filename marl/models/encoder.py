"""无跨样本状态的输入编码器；窗口维由显式布局声明。"""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from marl.core.layout import HistoryLayout
from marl.models.gru import GRUBackboneConfig
from marl.models.lstm import LSTMBackboneConfig
from marl.models.transformer import TransformerBackboneConfig


class InputEncoder(nn.Module):
    """接收 [*B,O]、返回 [*B,E]；不保存样本间 hidden state。"""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim, self.output_dim = input_dim, output_dim

    def validate(self, inputs: Tensor) -> None:
        """在网络入口验证最后一维，避免广播掩盖布局错误。"""
        if inputs.ndim < 1 or inputs.shape[-1] != self.input_dim:
            raise ValueError("输入最后一维与 encoder 布局不一致")

    def forward(self, inputs: Tensor) -> Tensor:
        """子类实现 [*B,input_dim] -> [*B,output_dim]；基类不执行计算。"""
        raise NotImplementedError


class IdentityEncoder(InputEncoder):
    """原样返回输入，不新增参数、不改变初始化随机数序列。"""

    def __init__(self, input_dim: int) -> None:
        super().__init__(input_dim, input_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        """检查 [*B,O] 后返回同一 Tensor，无复制、参数或状态变化。"""
        self.validate(inputs)
        return inputs


TemporalConfig = GRUBackboneConfig | LSTMBackboneConfig | TransformerBackboneConfig


class HistoryEncoder(InputEncoder):
    """[*B,O] -> 重排 [*B,W,C] -> 最后时刻隐藏输出 -> 拼接当前特征。"""

    history_indices: Tensor
    current_indices: Tensor

    def __init__(self, input_dim: int, layout: HistoryLayout, temporal: TemporalConfig) -> None:
        layout.validate(input_dim)
        network = temporal.build(len(layout.history_indices[0]))
        super().__init__(input_dim, network.output_dim + len(layout.current_indices))
        self.temporal = network
        self.register_buffer("history_indices", torch.tensor(layout.history_indices))
        self.register_buffer(
            "current_indices", torch.tensor(layout.current_indices, dtype=torch.long)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """窗口从零状态计算；梯度贯穿 W 步，批次与 replay 排列互不影响。"""
        self.validate(inputs)
        history = inputs[..., self.history_indices]
        features = self.temporal(history).features[..., -1, :]
        return torch.cat((features, inputs[..., self.current_indices]), dim=-1)


@dataclass(frozen=True, slots=True)
class IdentityEncoderConfig:
    kind: Literal["identity"] = "identity"

    def build(self, input_dim: int, layout: HistoryLayout | None = None) -> IdentityEncoder:
        return IdentityEncoder(input_dim)


@dataclass(frozen=True, slots=True)
class HistoryEncoderConfig:
    kind: Literal["history"] = "history"
    temporal: TemporalConfig = LSTMBackboneConfig()

    def __post_init__(self) -> None:
        if not isinstance(
            self.temporal, (GRUBackboneConfig, LSTMBackboneConfig, TransformerBackboneConfig)
        ):
            raise TypeError("history.temporal 只支持 GRU/LSTM/Transformer")

    def build(self, input_dim: int, layout: HistoryLayout | None = None) -> HistoryEncoder:
        if layout is None:
            raise ValueError("历史 encoder 需要 EnvironmentSpec 显式提供历史布局")
        return HistoryEncoder(input_dim, layout, self.temporal)


EncoderConfig = IdentityEncoderConfig | HistoryEncoderConfig
