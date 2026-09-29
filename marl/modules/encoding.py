"""集中式输入的显式装配，模型层无需认识 EnvironmentSpec。"""

import torch
from torch import Tensor, nn

from marl.envs.base import EnvironmentSpec
from marl.models import EncoderConfig, GNNBackboneConfig, IdentityEncoderConfig
from marl.models.encoder import IdentityEncoder, InputEncoder


class ObservationEncoder(InputEncoder):
    """独立编码 N 个局部观测，再可选聚合静态图；输入输出均保留 N。"""

    adjacency: Tensor | None

    def __init__(
        self, spec: EnvironmentSpec, encoder: EncoderConfig, graph: GNNBackboneConfig | None = None
    ) -> None:
        super().__init__(spec.observation_dim, spec.observation_dim)
        self.num_agents = spec.num_agents
        self.encoders = nn.ModuleList(
            encoder.build(spec.observation_dim, spec.observation_history)
            for _ in range(spec.num_agents)
        )
        first_encoder = self.encoders[0]
        assert isinstance(first_encoder, InputEncoder)
        self.output_dim = first_encoder.output_dim
        self.graph = None if graph is None else graph.build(self.output_dim)
        if self.graph is not None:
            self.output_dim = self.graph.output_dim
        self.register_buffer(
            "adjacency",
            None if spec.adjacency is None else torch.tensor(spec.adjacency, dtype=torch.float32),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """[*B,N,O] -> [*B,N,E]；图只读观测，不接收动作。"""
        self.validate(inputs)
        if inputs.ndim < 2 or inputs.shape[-2] != self.num_agents:
            raise ValueError("观测 encoder 输入必须为 [*B,N,O]")
        features = torch.stack(
            [encoder(inputs[..., i, :]) for i, encoder in enumerate(self.encoders)], dim=-2
        )
        if self.graph is not None:
            features = (
                self.graph(features).features
                if self.adjacency is None
                else self.graph(features, adjacency=self.adjacency).features
            )
        return features


class JointActionEncoder(InputEncoder):
    """保留 critic 的 [全部观测,全部动作] 接口，仅编码观测分支。"""

    def __init__(
        self, spec: EnvironmentSpec, encoder: EncoderConfig, graph: GNNBackboneConfig | None
    ) -> None:
        super().__init__(spec.num_agents * (spec.observation_dim + spec.action_dim), 1)
        self.num_agents, self.observation_dim = spec.num_agents, spec.observation_dim
        self.observations = ObservationEncoder(spec, encoder, graph)
        self.output_dim = spec.num_agents * (self.observations.output_dim + spec.action_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        """[*B,N*(O+A)] -> [*B,N*(E+A)]，动作梯度直接传入 Q backbone。"""
        self.validate(inputs)
        split = self.num_agents * self.observation_dim
        observations = inputs[..., :split].reshape(
            *inputs.shape[:-1], self.num_agents, self.observation_dim
        )
        return torch.cat((self.observations(observations).flatten(-2), inputs[..., split:]), -1)


class StateEncoder(InputEncoder):
    """按 state_layout 取节点，编码后拼接剩余全局特征。"""

    node_indices: Tensor
    current_indices: Tensor
    adjacency: Tensor | None

    def __init__(
        self, spec: EnvironmentSpec, encoder: EncoderConfig, graph: GNNBackboneConfig | None
    ) -> None:
        super().__init__(spec.state_dim, spec.state_dim)
        layout = spec.state_layout
        if layout is None:
            raise ValueError("全局 state 历史/图编码需要显式 state_layout")
        self.local = encoder.build(len(layout.node_indices[0]), layout.history)
        self.graph = None if graph is None else graph.build(self.local.output_dim)
        width = self.local.output_dim if self.graph is None else self.graph.output_dim
        self.output_dim = spec.num_agents * width + len(layout.current_indices)
        self.register_buffer("node_indices", torch.tensor(layout.node_indices))
        self.register_buffer(
            "current_indices", torch.tensor(layout.current_indices, dtype=torch.long)
        )
        self.register_buffer(
            "adjacency",
            None if spec.adjacency is None else torch.tensor(spec.adjacency, dtype=torch.float32),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """[*B,S] -> [*B,N*E+G]；节点与时间均来自布局，绝不猜测。"""
        self.validate(inputs)
        features = self.local(inputs[..., self.node_indices])
        if self.graph is not None:
            features = (
                self.graph(features).features
                if self.adjacency is None
                else self.graph(features, adjacency=self.adjacency).features
            )
        return torch.cat((features.flatten(-2), inputs[..., self.current_indices]), -1)


def state_encoder(
    spec: EnvironmentSpec, encoder: EncoderConfig, graph: GNNBackboneConfig | None
) -> InputEncoder:
    """默认 identity 保留原始 state 顺序、参数初始化顺序和数值。"""
    if isinstance(encoder, IdentityEncoderConfig) and graph is None:
        return IdentityEncoder(spec.state_dim)
    return StateEncoder(spec, encoder, graph)
