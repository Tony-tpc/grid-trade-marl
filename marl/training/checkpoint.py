"""trainer 共享的 checkpoint 文件与 PyTorch RNG 状态工具。"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import torch

TRAINER_CHECKPOINT_SCHEMA_VERSION = 4


def build_trainer_checkpoint_state(
    algorithm_state: Mapping[str, Any],
    config_data: Mapping[str, object] | None,
    optimization_state: Mapping[str, Any],
) -> dict[str, Any]:
    """复制训练器共有的模型、配置、优化器与 Torch RNG 状态。

    构造 on/off-policy trainer 共同拥有的 checkpoint 字段。

    Args:
        algorithm_state: 模型 state_dict；校验函数中为可选的当前模型结构参考。
        config_data: 实验构造时生成的规范化配置快照；None 只与同为 None 的快照匹配。
        optimization_state: OptimizerRuntime.state_dict()，包含优化器和更新计数等状态。

    Returns:
        schema 4 的字典；不包含环境进度，replay/NumPy RNG 由离策略训练器补充。
    """

    return {
        "schema_version": TRAINER_CHECKPOINT_SCHEMA_VERSION,
        "algorithm": deepcopy(dict(algorithm_state)),
        "config": deepcopy(dict(config_data)) if config_data is not None else None,
        **capture_torch_rng_state(),
        "optimization": deepcopy(dict(optimization_state)),
    }


def validate_trainer_checkpoint_state(
    state: Mapping[str, Any], config_data: Mapping[str, object] | None, *,
    algorithm_state: Mapping[str, Any] | None = None,
) -> None:
    """验证 schema/config；提供 algorithm_state 时额外校验参数名称、shape、dtype。

    统一检查 checkpoint schema 与实验配置，禁止静默部分恢复。

    Args:
        state: 待验证/恢复的 checkpoint 状态 Mapping；须来自兼容配置和 schema。
        config_data: 实验构造时生成的规范化配置快照；None 只与同为 None 的快照匹配。
        algorithm_state: 模型 state_dict；校验函数中为可选的当前模型结构参考。

    Returns:
        None；不匹配抛 ValueError，不修改任何训练状态。
    """

    if state.get("schema_version") != TRAINER_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "不兼容的 trainer checkpoint：旧 schema 缺少新版 backbone/序列训练契约；"
            "可显式加载 algorithm 权重用于评估/诊断，不支持静默精确续训"
        )
    expected = dict(config_data) if config_data is not None else None
    if state.get("config") != expected:
        raise ValueError("checkpoint config 与当前 trainer config 不一致")
    if algorithm_state is not None:
        saved = state.get("algorithm")
        if not isinstance(saved, Mapping) or set(saved) != set(algorithm_state):
            raise ValueError("checkpoint 网络参数名称与当前结构不一致")
        for name, current in algorithm_state.items():
            value = saved[name]
            if isinstance(current, torch.Tensor) and (
                not isinstance(value, torch.Tensor) or current.shape != value.shape
                or current.dtype != value.dtype
            ):
                raise ValueError(f"checkpoint 网络参数 {name} 形状/dtype 与当前结构不一致")


def capture_torch_rng_state() -> dict[str, object]:
    """复制当前 Torch CPU RNG 和可用的全部 CUDA RNG 状态。

    复制 CPU/CUDA RNG 状态，避免 checkpoint 被后续采样原地影响。

    Inputs:
        无显式参数；读取当前 Torch RNG。

    Returns:
        包含 torch_rng、cuda_rng 的字典；无 CUDA 时 cuda_rng 为 None。
    """

    return {
        "torch_rng": torch.get_rng_state().clone(),
        "cuda_rng": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }


def restore_torch_rng_state(state: Mapping[str, Any]) -> None:
    """恢复保存的 Torch 随机状态，供下一次采样/更新继续使用。

    恢复由 :func:`capture_torch_rng_state` 保存的 RNG 状态。

    Args:
        state: 待验证/恢复的 checkpoint 状态 Mapping；须来自兼容配置和 schema。

    Returns:
        None；修改进程级 Torch CPU/CUDA RNG。
    """

    torch_rng = state.get("torch_rng")
    if not isinstance(torch_rng, torch.Tensor):
        raise TypeError("checkpoint torch_rng 必须是 Tensor")
    torch.set_rng_state(torch_rng.cpu())
    cuda_rng = state.get("cuda_rng")
    if cuda_rng is not None and torch.cuda.is_available():
        if not isinstance(cuda_rng, list) or not all(
            isinstance(item, torch.Tensor) for item in cuda_rng
        ):
            raise TypeError("checkpoint cuda_rng 必须是 Tensor 列表或 None")
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_rng])


def save_checkpoint_state(state: Mapping[str, Any], path: str | Path) -> None:
    """创建父目录并使用 torch.save 序列化完整状态。

    创建父目录并保存 mapping checkpoint。

    Args:
        state: 待验证/恢复的 checkpoint 状态 Mapping；须来自兼容配置和 schema。
        path: 本地 checkpoint 文件路径；保存时自动创建父目录。

    Returns:
        None；写入或覆盖 path 指定文件。
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, target)


def load_checkpoint_state(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> Mapping[str, Any]:
    """反序列化可信 checkpoint 并检查顶层为 Mapping。

    加载可信的本地 checkpoint，并在 trainer 读取字段前验证顶层协议。

    包含 optimizer/NumPy RNG 的完整状态需要 weights_only=False；pickle 可执行
    代码，因此不得用此函数加载不可信下载文件。加载模型权重不等于完整续训。

    Args:
        path: 本地 checkpoint 文件路径；保存时自动创建父目录。
        map_location: torch.load 的张量映射设备；加载后的运行设备由已构造实验决定。

    Returns:
        状态 Mapping；只读文件，实际模型恢复由 trainer 完成。
    """

    state = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(state, Mapping):
        raise TypeError("trainer checkpoint 顶层必须是 mapping")
    return cast(Mapping[str, Any], state)
