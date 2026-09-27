"""不含算法分支的 optimizer、AMP、计数器和 checkpoint 状态。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
from copy import deepcopy
from typing import Any, cast

import torch
from torch import Tensor, nn

from marl.target_updates import TargetUpdate


def metrics_to_float(metrics: Mapping[str, Tensor]) -> dict[str, float]:
    """把 detached 标量一次性搬回 CPU；日志边界之外避免逐项 .item() 同步。"""

    if not metrics:
        return {}
    names = tuple(metrics)
    values = torch.stack(
        [metrics[name].detach().reshape(()) for name in names]
    ).cpu().tolist()
    return dict(zip(names, values, strict=True))


class OptimizerRuntime:
    """保存 optimizer、设备端统计及恢复状态，不构图、不 backward、不编排更新。

    optimizer 的参数集合在构造后必须保持不变；一个参数只能属于一个 optimizer。
    ``update_count`` 计算法更新轮次，``optimizer_step_count`` 计真实 step 次数，
    例如 PPO 一次 rollout 会包含多次 actor/critic step，二者不能互换。
    """

    def __init__(
        self,
        optimizers: Mapping[str, torch.optim.Optimizer],
        max_grad_norms: Mapping[str, float | None],
        *,
        amp_dtype: torch.dtype | None = None,
        target_update: TargetUpdate | None = None,
        generator: torch.Generator | None = None,
        sync_metrics: bool = True,
    ) -> None:
        if not optimizers or set(optimizers) != set(max_grad_norms):
            raise ValueError("optimizers 与 max_grad_norms 必须具有相同的非空名称集合")
        if amp_dtype not in (None, torch.bfloat16):
            raise ValueError("amp_dtype 目前只支持 torch.bfloat16")
        self.optimizers = dict(optimizers)
        self._parameters = {
            name: tuple(
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            )
            for name, optimizer in self.optimizers.items()
        }
        owners: dict[int, str] = {}
        for name, parameters in self._parameters.items():
            if not parameters:
                raise ValueError(f"optimizer {name} 没有参数")
            for parameter in parameters:
                if id(parameter) in owners:
                    raise ValueError(
                        f"optimizer 参数重复归属：{owners[id(parameter)]} 与 {name}"
                    )
                owners[id(parameter)] = name
        if any(limit is not None and limit <= 0 for limit in max_grad_norms.values()):
            raise ValueError("max_grad_norm 必须大于 0 或为 None")
        self.max_grad_norms = dict(max_grad_norms)
        self.amp_dtype = amp_dtype
        self.target_update = target_update
        self.generator = generator
        self.sync_metrics = sync_metrics
        self.update_count = 0
        self.optimizer_step_count = 0
        self._metric_sums: dict[str, Tensor] = {}
        self._metric_count = 0
        self._step_metrics: dict[str, Tensor] = {}

    def optimizer(self, name: str) -> torch.optim.Optimizer:
        try:
            return self.optimizers[name]
        except KeyError as error:
            raise KeyError(f"未知 optimizer 名称: {name}") from error

    def max_grad_norm(self, name: str) -> float | None:
        if name not in self.max_grad_norms:
            raise KeyError(f"未知 optimizer 名称: {name}")
        return self.max_grad_norms[name]

    def parameters(self, name: str) -> tuple[nn.Parameter, ...]:
        """返回初始化时缓存的 optimizer 参数，避免每次更新重新遍历 param_groups。"""

        try:
            return self._parameters[name]
        except KeyError as error:
            raise KeyError(f"未知 optimizer 名称: {name}") from error

    def autocast(self, device: torch.device) -> AbstractContextManager[None]:
        if self.amp_dtype is not None and (
            device.type != "cuda" or not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("BF16 AMP 需要支持 BF16 的 CUDA 设备")
        return torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=self.amp_dtype is not None,
        )

    def record_optimizer_step(self, name: str | None = None, norm: Tensor | None = None) -> None:
        self.optimizer_step_count += 1
        if name is None or norm is None:
            return
        # 在平均之前统计每一次 optimizer step；不把平均梯度与阈值比较。
        norm = norm.detach()
        limit = self.max_grad_norm(name)
        clipped = (norm > limit).float() if limit is not None else torch.zeros_like(norm)
        events = {
            f"{name}_clip_count": clipped,
            f"{name}_step_count": torch.ones_like(norm),
            f"{name}_gradient_abs_max": norm,
            f"{name}_gradient_nonfinite": (~torch.isfinite(norm)).float(),
        }
        self.accumulate_metrics(self._step_metrics, events)

    @staticmethod
    def accumulate_metrics(
        destination: dict[str, Tensor], values: Mapping[str, Tensor], *, weight: int = 1
    ) -> None:
        """累加 detached 指标：普通均值按 weight 加权，事件数求和，峰值取最大。

        PPO 的普通指标以 mini-batch 样本数为权重；梯度范数以 optimizer step 为
        单位，需单独用默认 weight=1 累加。后缀 _count/_abs_max/_nonfinite 的
        字段从不按样本数缩放，避免把一次裁剪误计成多次事件。
        """

        if weight < 1:
            raise ValueError("metric weight 必须为正整数")
        # CUDA 上数十个 scalar add/max 会产生大量微小 kernel；合并为一次向量归约。
        # 权重也在打包后应用，避免修正 PPO 日志时又引入逐指标 scalar multiply。
        # CPU 保留简单循环；checkpoint 继续使用命名字段而不是打包存储格式。
        if values and next(iter(values.values())).is_cuda and len(values) > 8:
            names = tuple(values)
            incoming = torch.stack([values[name].detach() for name in names])
            if weight != 1:
                unweighted = torch.tensor(
                    [name.endswith(("_count", "_abs_max", "_nonfinite")) for name in names],
                    dtype=torch.bool, device=incoming.device,
                )
                incoming = torch.where(unweighted, incoming, incoming * weight)
            if not destination:
                destination.update(zip(names, incoming.unbind(), strict=True))
                return
            packed_previous = torch.stack([
                destination[name] if name in destination else torch.zeros_like(values[name])
                for name in names
            ])
            maxima = torch.tensor(
                [name.endswith(("_abs_max", "_nonfinite")) for name in names],
                dtype=torch.bool, device=incoming.device,
            )
            combined = torch.where(maxima, torch.maximum(packed_previous, incoming),
                                   packed_previous + incoming)
            destination.update(zip(names, combined.unbind(), strict=True))
            return
        for name, value in values.items():
            value = value.detach()
            if weight != 1 and not name.endswith(("_count", "_abs_max", "_nonfinite")):
                value = value * weight
            previous = destination.get(name)
            if previous is None:
                destination[name] = value
            elif name.endswith(("_abs_max", "_nonfinite")):
                destination[name] = torch.maximum(previous, value)
            else:
                destination[name] = previous + value

    @staticmethod
    def mean_metrics(totals: Mapping[str, Tensor], count: int) -> dict[str, Tensor]:
        """除以累计权重；计数和极值保留原义，裁剪率由真实 step 事件计算。"""

        if count < 1:
            raise ValueError("metric count 必须为正整数")
        result = {
            name: value if name.endswith(("_count", "_abs_max", "_nonfinite")) else value / count
            for name, value in totals.items()
        }
        for name in tuple(result):
            if name.endswith("_clip_count"):
                prefix = name.removesuffix("_clip_count")
                result[f"{prefix}_gradient_clip_rate"] = (
                    result[name] / result[f"{prefix}_step_count"].clamp_min(1)
                )
        return result

    def export_metrics(self, metrics: Mapping[str, Tensor]) -> dict[str, float]:
        """默认立即返回 CPU 数值；关闭同步时在当前设备累计 detached 指标。"""

        detached = {name: value.detach().reshape(()) for name, value in metrics.items()}
        detached.update(self._step_metrics)
        self._step_metrics = {}
        if self.sync_metrics:
            return metrics_to_float(self.mean_metrics(detached, 1))
        self.accumulate_metrics(self._metric_sums, detached)
        self._metric_count += 1
        return {}

    def flush_metrics(self) -> dict[str, float]:
        """同步并清空设备端累计指标；没有待处理指标时返回空字典。"""

        if self._metric_count == 0:
            return {}
        means = self.mean_metrics(self._metric_sums, self._metric_count)
        self._metric_sums.clear()
        self._metric_count = 0
        return metrics_to_float(means)

    def finish(
        self, target_pairs: Iterable[tuple[nn.Module, nn.Module]] = ()
    ) -> None:
        self.update_count += 1
        if self.target_update is None:
            return
        should_update = getattr(self.target_update, "should_step", None)
        if should_update is None or should_update(self.update_count):
            self.target_update.step(target_pairs)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "optimizers": {
                name: deepcopy(optimizer.state_dict())
                for name, optimizer in self.optimizers.items()
            },
            "generator": self.generator.get_state().clone() if self.generator is not None else None,
            "update_count": self.update_count,
            "optimizer_step_count": self.optimizer_step_count,
            "sync_metrics": self.sync_metrics,
            "metric_sums": {
                name: value.detach().clone()
                for name, value in self._metric_sums.items()
            },
            "metric_count": self._metric_count,
            "step_metrics": {name: value.clone() for name, value in self._step_metrics.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != 3:
            raise ValueError(
                "不兼容的 optimizer checkpoint：旧 schema 的累计统计无法精确恢复"
            )
        saved = cast(Mapping[str, dict[str, Any]], state["optimizers"])
        if set(saved) != set(self.optimizers):
            raise ValueError("checkpoint optimizer 名称与当前算法不一致")
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(saved[name])
        generator_state = state.get("generator")
        if generator_state is not None:
            if self.generator is None:
                self.generator = torch.Generator()
            self.generator.set_state(generator_state.cpu())
        self.update_count = int(state["update_count"])
        self.optimizer_step_count = int(state["optimizer_step_count"])
        self.sync_metrics = bool(state.get("sync_metrics", True))
        device = next(iter(self._parameters.values()))[0].device
        saved_metric_sums = cast(Mapping[str, Tensor], state.get("metric_sums", {}))
        self._metric_sums = {
            name: value.to(device) for name, value in saved_metric_sums.items()
        }
        self._metric_count = int(state.get("metric_count", 0))
        self._step_metrics = {
            name: value.to(device) for name, value in state.get("step_metrics", {}).items()
        }
