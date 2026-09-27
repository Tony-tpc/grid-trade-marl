"""trainer 共享的 checkpoint 文件与 PyTorch RNG 状态工具。"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import torch

TRAINER_CHECKPOINT_SCHEMA_VERSION = 3


def build_trainer_checkpoint_state(
    algorithm_state: Mapping[str, Any],
    config_data: Mapping[str, object] | None,
    optimization_state: Mapping[str, Any],
) -> dict[str, Any]:
    """构造 on/off-policy trainer 共同拥有的 checkpoint 字段。"""

    return {
        "schema_version": TRAINER_CHECKPOINT_SCHEMA_VERSION,
        "algorithm": deepcopy(dict(algorithm_state)),
        "config": deepcopy(dict(config_data)) if config_data is not None else None,
        **capture_torch_rng_state(),
        "optimization": deepcopy(dict(optimization_state)),
    }


def validate_trainer_checkpoint_state(
    state: Mapping[str, Any], config_data: Mapping[str, object] | None
) -> None:
    """统一检查 checkpoint schema 与实验配置，禁止静默部分恢复。"""

    if state.get("schema_version") != TRAINER_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "不兼容的 trainer checkpoint：旧 schema 缺少新版统计语义；"
            "可显式加载 algorithm 权重用于评估/诊断，不支持静默精确续训"
        )
    expected = dict(config_data) if config_data is not None else None
    if state.get("config") != expected:
        raise ValueError("checkpoint config 与当前 trainer config 不一致")


def capture_torch_rng_state() -> dict[str, object]:
    """复制 CPU/CUDA RNG 状态，避免 checkpoint 被后续采样原地影响。"""

    return {
        "torch_rng": torch.get_rng_state().clone(),
        "cuda_rng": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }


def restore_torch_rng_state(state: Mapping[str, Any]) -> None:
    """恢复由 :func:`capture_torch_rng_state` 保存的 RNG 状态。"""

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
    """创建父目录并保存 mapping checkpoint。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, target)


def load_checkpoint_state(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> Mapping[str, Any]:
    """加载可信的本地 checkpoint，并在 trainer 读取字段前验证顶层协议。

    包含 optimizer/NumPy RNG 的完整状态需要 weights_only=False；pickle 可执行
    代码，因此不得用此函数加载不可信下载文件。加载模型权重不等于完整续训。
    """

    state = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(state, Mapping):
        raise TypeError("trainer checkpoint 顶层必须是 mapping")
    return cast(Mapping[str, Any], state)
