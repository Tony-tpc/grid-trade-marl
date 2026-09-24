from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from marl.core.batch import MARLBatch


@dataclass(frozen=True, slots=True, kw_only=True)
class AlgorithmConfig:
    """所有算法都会使用的超参数。

    ``gamma`` 是折扣因子：越接近 1，智能体越重视较远的未来奖励。
    ``tau`` 是目标网络软更新比例：每次只把在线网络的一小部分参数混入目标网络。
    ``grad_clip_norm`` 是梯度范数上限，用来降低训练突然发散的概率。

    ``frozen=True`` 表示配置创建后不可修改，防止实验中途意外改变超参数；
    ``slots=True`` 可减少配置对象的内存开销和拼写错误。
    """

    gamma: float = 0.99
    tau: float = 0.005
    grad_clip_norm: float | None = 10.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma 必须位于 [0, 1]")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau 必须位于 (0, 1]")


class BaseMARLAlgorithm(nn.Module, ABC):
    """所有 MARL 算法的生命周期接口。

    ``nn.Module`` 是所有 PyTorch 网络的基类。继承它以后，挂在 ``self`` 上的 Actor、
    Critic 和 Mixer 会被自动注册，因而能被 ``parameters()``、``state_dict()``、
    ``to(device)`` 等 PyTorch API 统一管理。

    子类负责两件算法特有的事情：

    * ``act``：给环境观测，返回智能体动作；
    * ``compute_loss``：给一批经验，返回需要反向传播的损失。

    梯度清零、反向传播、参数更新、目标网络更新和 checkpoint 在基类中统一处理。
    """

    def __init__(self, config: AlgorithmConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool = False,
        action_mask: Tensor | None = None,
        **kwargs: Tensor,
    ) -> Tensor:
        """为环境采样器产生动作。

        常见输入形状为 ``[B, N, O]``：B 是并行环境/批大小，N 是智能体数量，
        O 是每个智能体的观测维度。返回通常是 ``[B, N]``（离散动作）或
        ``[B, N, A]``（连续动作）。
        """

    @abstractmethod
    def compute_loss(self, batch: MARLBatch) -> dict[str, Tensor]:
        """构建算法的计算图并返回命名 loss。

        这里只做前向计算，不调用 ``backward`` 或 ``optimizer.step``。这样测试代码可以
        单独检查每个 loss，训练器也可以在需要时做梯度累积或混合精度训练。


        """

    def optimize(self, batch: MARLBatch, optimizer: torch.optim.Optimizer) -> dict[str, float]:
        """执行一次标准 PyTorch 训练步骤。
        用来进行整体梯度更新（包括 Actor 和 Critic 的参数更新）。
        顺序非常重要：计算 loss -> 清除旧梯度 -> 反向传播 -> 裁剪梯度 -> 更新参数。
        PyTorch 默认会累加梯度，所以每一步都必须先 ``zero_grad``。
        """
        losses = self.compute_loss(batch)
        if "loss" not in losses:
            raise KeyError("compute_loss 必须返回总损失键 'loss'")
        # set_to_none=True 比把每个梯度张量清零更省内存；下一次 backward 会重新创建它。
        optimizer.zero_grad(set_to_none=True)
        # backward() 沿计算图应用链式法则，把梯度写入每个 Parameter 的 .grad（求导）。
        losses["loss"].backward()
        if self.config.grad_clip_norm is not None:
            # 梯度裁剪只改变梯度，不改变参数；它对 RNN 和不稳定的 TD 学习尤其有用。
            nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip_norm)
        # optimizer.step() 才真正修改神经网络参数。
        optimizer.step()
        self.update_targets()  # 更新目标网络，确保训练稳定性（DQN）。
        # 输出每个 loss 的数值，便于日志记录或 TensorBoard 可视化。
        return {name: float(value.detach()) for name, value in losses.items()}

    def update_targets(self) -> None:
        """无 target network 的算法无需操作；off-policy 子类覆写该 hook。"""

    @staticmethod
    @contextmanager
    def frozen(module: nn.Module) -> Iterator[None]:
        """临时冻结模块参数，但保留对模块输入的梯度。

        actor loss 需要通过 critic 对 action 求导，却不应让该 loss 更新 critic 参数。
        ``requires_grad_(False)`` 只冻结 critic 的参数；梯度仍能从 critic 输出沿 action
        回到 actor，所以 actor 仍然知道“动作往哪个方向变化会得到更高 Q 值”。
        try:
            critic.requires_grad=False
            actor_loss计算
        finally:
            critic恢复
        """
        original = [parameter.requires_grad for parameter in module.parameters()]
        try:
            module.requires_grad_(False)
            yield
        finally:
            for parameter, requires_grad in zip(module.parameters(), original, strict=True):
                parameter.requires_grad_(requires_grad)

    @torch.no_grad()  # 执行这个函数时，不建立计算图，不记录梯度
    def soft_update(self, target: nn.Module, source: nn.Module) -> None:
        """执行 Polyak 软更新，适用于 MADDPG/MASAC/QMIX 等算法。

        公式为 ``target = (1 - tau) * target + tau * source``。目标网络变化较慢，
        可以避免 TD target 与正在拟合它的网络同时剧烈变化。
        """
        for target_parameter, source_parameter in zip(
            target.parameters(), source.parameters(), strict=True
        ):
            target_parameter.lerp_(source_parameter, self.config.tau)

    @staticmethod
    @torch.no_grad()
    def hard_update(target: nn.Module, source: nn.Module) -> None:
        """把 source 参数完整复制到 target；某些算法每隔若干步调用一次。"""
        target.load_state_dict(source.state_dict())

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        step: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """保存算法参数、优化器状态和恢复训练所需的附加信息。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "algorithm_state": self.state_dict(),
            "config": asdict(self.config),
            "step": step,
            "metadata": metadata or {},
        }
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        torch.save(payload, target)

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """恢复模型/优化器，并把 step 和 metadata 返回给训练循环。"""
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        if checkpoint.get("config") != asdict(self.config):
            raise ValueError("checkpoint 配置与当前算法配置不一致")
        self.load_state_dict(checkpoint["algorithm_state"])
        if optimizer is not None and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        return {
            "step": checkpoint.get("step", 0),
            "metadata": checkpoint.get("metadata", {}),
        }
