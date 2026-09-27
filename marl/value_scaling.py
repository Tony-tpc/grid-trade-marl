"""逐智能体 target 统计；只缩放 critic 误差，不改变对外 V/Q 的单位。"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TargetScale(nn.Module):
    """输入 ``[..., N]``，沿所有样本维更新 Welford 统计。

    开关关闭时不保存额外状态，兼容旧模型权重。开启时 count/mean/m2 随
    state_dict 保存；统计只由算法 update 显式更新，诊断 loss 不修改状态。
    最小标准差为 1，避免初始低方差把梯度放大。不是 PopArt，也不保证收敛。
    """

    count: Tensor
    mean: Tensor
    m2: Tensor

    def __init__(self, num_agents: int, enabled: bool = False) -> None:
        super().__init__()
        self.enabled = enabled
        self.register_buffer("count", torch.zeros((), dtype=torch.float64), persistent=enabled)
        self.register_buffer(
            "mean", torch.zeros(num_agents, dtype=torch.float64), persistent=enabled
        )
        self.register_buffer("m2", torch.zeros(num_agents, dtype=torch.float64), persistent=enabled)

    @torch.no_grad()
    def update(self, targets: Tensor) -> None:
        if not self.enabled:
            return
        if targets.shape[-1] != self.mean.numel() or targets.numel() == 0:
            raise ValueError("target 必须是非空的 [..., N] 张量")
        samples = targets.detach().reshape(-1, self.mean.numel()).double()
        variance, mean = torch.var_mean(samples, dim=0, unbiased=False)
        size = samples.shape[0]
        total = self.count + size
        delta = mean - self.mean
        self.m2.add_(variance * size + delta.square() * self.count * size / total)
        self.mean.add_(delta * size / total)
        self.count.copy_(total)

    def forward(self, values: Tensor) -> Tensor:
        if not self.enabled:
            return values
        return (values - self.mean.to(values)) / self.scale().to(values)

    def scale(self) -> Tensor:
        return (self.m2 / self.count.clamp_min(1)).clamp_min(1).sqrt()


@torch.no_grad()
def agent_statistics(name: str, values: Tensor) -> dict[str, Tensor]:
    """对 [..., N] 原始尺度数据返回逐智能体均值、绝对峰值与非有限事件。"""
    flat = values.detach().reshape(-1, values.shape[-1])
    mean = flat.mean(0)
    peak = flat.abs().amax(0)
    nonfinite = (~torch.isfinite(flat)).any(0).float()
    return {
        f"{name}_agent_{i}_{stat}": value[i]
        for stat, value in (("mean", mean), ("abs_max", peak), ("nonfinite", nonfinite))
        for i in range(flat.shape[-1])
    }


@torch.no_grad()
def value_diagnostics(predictions: Tensor, targets: Tensor) -> dict[str, Tensor]:
    """原始单位的价值误差；解释方差只沿样本维计算，不混合角色。"""
    prediction = predictions.detach().reshape(-1, predictions.shape[-1])
    target = targets.detach().reshape_as(prediction)
    error = prediction - target
    variance = target.var(0, unbiased=False)
    ev = torch.where(variance > 1e-12,
                     1 - error.var(0, unbiased=False) / variance.clamp_min(1e-12),
                     torch.zeros_like(variance))
    metrics = {**agent_statistics("value", prediction),
               **agent_statistics("target", target),
               **agent_statistics("td_error", error)}
    bias = error.mean(0)
    rmse = error.square().mean(0).sqrt()
    for i in range(prediction.shape[-1]):
        metrics[f"explained_variance_agent_{i}"] = ev[i]
        metrics[f"value_bias_agent_{i}"] = bias[i]
        metrics[f"value_rmse_agent_{i}"] = rmse[i]
    metrics["explained_variance"] = ev.mean()
    return metrics
