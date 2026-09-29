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
    """将标量指标打包后一次性同步到 CPU，减少逐项 GPU 同步。

    把 detached 标量一次性搬回 CPU；日志边界之外避免逐项 .item() 同步。

    Args:
        metrics: 名称到标量 Tensor 的映射；导出前会 detach。

    Returns:
        dict[str,float]；输入为空则返回 {}。
    """

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
        """验证 optimizer 名称和参数归属，缓存参数元组并初始化计数/日志状态。

        Args:
            optimizers: 名称到 optimizer 的非空映射；各 optimizer 不得拥有重复参数。
            max_grad_norms: 与 optimizers 名称完全一致的裁剪上限映射；None 表示不裁剪。
            amp_dtype: None 使用普通精度；torch.bfloat16 要求支持 BF16 的 CUDA。
            target_update: 可选 hard/soft target 更新器；finish 按算法更新次数调度。
            generator: 打乱 mini-batch 顺序的 Torch Generator；设备须与 randperm 一致。
            sync_metrics: True 立即导出 CPU float；False 在设备上累计，之后用 flush_metrics 读取。

        Returns:
            None；不执行 backward 或 step。
        """
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
        """按角色名称取回 optimizer，未知名称立即报错。

        Args:
            name: 字段或 optimizer 角色名称；用于定位对应状态以及报告错误。

        Returns:
            torch.optim.Optimizer；返回现有实例。
        """
        try:
            return self.optimizers[name]
        except KeyError as error:
            raise KeyError(f"未知 optimizer 名称: {name}") from error

    def max_grad_norm(self, name: str) -> float | None:
        """读取该 optimizer 的有效裁剪阈值。

        Args:
            name: 字段或 optimizer 角色名称；用于定位对应状态以及报告错误。

        Returns:
            float 或 None；未知名称抛 KeyError。
        """
        if name not in self.max_grad_norms:
            raise KeyError(f"未知 optimizer 名称: {name}")
        return self.max_grad_norms[name]

    def parameters(self, name: str) -> tuple[nn.Parameter, ...]:
        """读取初始化时缓存的参数，供 optimize 限定裁剪和梯度统计范围。

        返回初始化时缓存的 optimizer 参数，避免每次更新重新遍历 param_groups。

        Args:
            name: 字段或 optimizer 角色名称；用于定位对应状态以及报告错误。

        Returns:
            nn.Parameter 元组；不会复制参数。
        """

        try:
            return self._parameters[name]
        except KeyError as error:
            raise KeyError(f"未知 optimizer 名称: {name}") from error

    def autocast(self, device: torch.device) -> AbstractContextManager[None]:
        """创建自动混合精度上下文，启用 BF16 前校验 CUDA 支持。

        Args:
            device: 模型前向或张量迁移使用的设备，例如 cpu 或 cuda。

        Returns:
            上下文管理器；退出恢复原精度上下文。
        """
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
        """记录一次真实 optimizer step，可选累加裁剪、范数峰值和非有限事件。

        Args:
            name: 字段或 optimizer 角色名称；用于定位对应状态以及报告错误。
            norm: 可选的裁剪前标量梯度范数，用于记录裁剪次数与非有限事件。

        Returns:
            None；总 step 计数增加，不执行优化动作。
        """
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
        """累积 detached 日志：普通指标加权求和、事件计数求和、峰值取最大。

        累加 detached 指标：普通均值按 weight 加权，事件数求和，峰值取最大。

        PPO 的普通指标以 mini-batch 样本数为权重；梯度范数以 optimizer step 为
        单位，需单独用默认 weight=1 累加。后缀 _count/_abs_max/_nonfinite 的
        字段从不按样本数缩放，避免把一次裁剪误计成多次事件。

        Args:
            destination: 原地累加的命名张量指标字典。
            values: 本次待累加的命名标量张量指标。
            weight: 普通指标的正整数权重；事件计数与极值不乘此权重。

        Returns:
            None；原地修改 destination。
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
        """将普通累计值除以权重，保留计数和极值，并计算实际梯度裁剪率。

        除以累计权重；计数和极值保留原义，裁剪率由真实 step 事件计算。

        Args:
            totals: 累计指标；普通字段存加权和，事件字段存计数或极值。
            count: 普通指标的累计权重或次数，必须为正。

        Returns:
            新的 dict[str,Tensor]；不改变 totals。
        """

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
        """合并本轮 step 诊断；立即同步或将结果累计到设备端。

        默认立即返回 CPU 数值；关闭同步时在当前设备累计 detached 指标。

        Args:
            metrics: 名称到标量 Tensor 的映射；导出前会 detach。

        Returns:
            同步模式返回 dict[str,float]，延迟模式返回 {}；本轮 step 缓存被清空。
        """

        detached = {name: value.detach().reshape(()) for name, value in metrics.items()}
        detached.update(self._step_metrics)
        self._step_metrics = {}
        if self.sync_metrics:
            return metrics_to_float(self.mean_metrics(detached, 1))
        self.accumulate_metrics(self._metric_sums, detached)
        self._metric_count += 1
        return {}

    def flush_metrics(self) -> dict[str, float]:
        """导出并清空延迟累计的训练指标。

        同步并清空设备端累计指标；没有待处理指标时返回空字典。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            累计轮次平均的 dict[str,float]；没有累计值时返回 {}。
        """

        if self._metric_count == 0:
            return {}
        means = self.mean_metrics(self._metric_sums, self._metric_count)
        self._metric_sums.clear()
        self._metric_count = 0
        return metrics_to_float(means)

    def finish(
        self, target_pairs: Iterable[tuple[nn.Module, nn.Module]] = ()
    ) -> None:
        """结束一轮算法 update，并按 hard/soft 规则同步 target。

        Args:
            target_pairs: 按 (target, online) 排列的网络对；无目标网络时为空。

        Returns:
            None；update_count 加一，与 optimizer_step_count 区别开。
        """
        self.update_count += 1
        if self.target_update is None:
            return
        should_update = getattr(self.target_update, "should_step", None)
        if should_update is None or should_update(self.update_count):
            self.target_update.step(target_pairs)

    def state_dict(self) -> dict[str, Any]:
        """复制 optimizer、mini-batch RNG、更新计数及尚未 flush 的日志。

        Inputs:
            无显式参数；读取当前实例字段。

        Returns:
            schema 3 状态字典；不包含网络与环境。
        """
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
        """检查 schema 3 与 optimizer 名称后恢复 optimizer、RNG、计数和日志。

        Args:
            state: 待验证/恢复的 checkpoint 状态 Mapping；须来自兼容配置和 schema。

        Returns:
            None；直接修改 runtime，不独立提供事务回滚。
        """
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
