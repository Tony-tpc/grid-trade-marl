"""基准运行的调度、进度和可恢复边界；不拥有任何算法更新或环境规则。"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from marl.training import OffPolicyTrainer, OnPolicyTrainer
from marl.training.checkpoint import load_checkpoint_state, save_checkpoint_state


@dataclass
class TransitionSchedule:
    """按联合 transition 计数；warm-up 之后余数可保存恢复。"""
    learning_starts: int
    train_every: int = 1
    gradient_steps: int = 1
    seen: int = 0
    pending: int = 0

    def add(self, count: int) -> int:
        if count < 0 or min(self.learning_starts, self.train_every, self.gradient_steps) < 1:
            raise ValueError("调度参数必须为正，新增 transition 数不能为负")
        eligible_before = max(0, self.seen - self.learning_starts + 1)
        self.seen += count
        eligible_after = max(0, self.seen - self.learning_starts + 1)
        self.pending += eligible_after - eligible_before
        events, self.pending = divmod(self.pending, self.train_every)
        return events * self.gradient_steps


@lru_cache(maxsize=1)
def source_identity() -> dict[str, str]:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for folder in ("marl", "benchmarks"):
        for path in sorted((root / folder).rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                          text=True).strip(),
        "source_sha256": digest.hexdigest(),
    }


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(path)


def save_run(
    path: Path, trainer: OffPolicyTrainer | OnPolicyTrainer, settings: dict[str, Any],
    *, name: str, seed: int, episode: int, environment_steps: int,
    training_history: list[dict[str, Any]], evaluation_history: list[dict[str, Any]],
    metric_start: int, elapsed_seconds: float, exploration_rng: dict[str, Any] | None,
    schedule: dict[str, int] | None, checkpoint_interval: int, total_steps: int,
) -> None:
    """仅在完整 rollout 后保存；下一 episode reset seed 可以精确重建环境。"""
    state = dict(schema_version=1, settings=settings, source=source_identity(),
                 algorithm=name, seed=seed, episode=episode, environment_steps=environment_steps,
                 training_history=training_history, evaluation_history=evaluation_history,
                 metric_start=metric_start, elapsed_seconds=elapsed_seconds,
                 exploration_rng=exploration_rng, schedule=schedule)
    state["execution"] = {"device": str(trainer.device), "torch_threads": torch.get_num_threads(),
                          "torch_version": torch.__version__}
    atomic_json(path.with_suffix(".progress.json"), {
        key: value for key, value in state.items()
        if key not in ("training_history", "evaluation_history", "exploration_rng")
    } | {"eta_seconds": elapsed_seconds / max(environment_steps, 1)
         * max(0, total_steps - environment_steps),
         "updated_at_unix": time.time(), "torch_threads": torch.get_num_threads()})
    previous_evaluation_step = (evaluation_history[-2]["environment_steps"]
                                if len(evaluation_history) > 1 else 0)
    if (environment_steps // checkpoint_interval == previous_evaluation_step // checkpoint_interval
            and environment_steps != total_steps):
        return
    state["trainer"] = trainer.state_dict()
    pending = path.with_suffix(".tmp")
    save_checkpoint_state(state, pending)
    pending.replace(path)


def load_run(
    path: Path, trainer: OffPolicyTrainer | OnPolicyTrainer, settings: dict[str, Any],
    *, name: str, seed: int, device: torch.device,
) -> dict[str, Any]:
    state = dict(load_checkpoint_state(path, map_location=device))
    if state.get("schema_version") != 1 or "trainer" not in state:
        raise ValueError("旧实验 checkpoint 缺少 episode/RNG/调度状态，不能精确续训")
    if (state.get("settings") != settings or state.get("algorithm") != name
            or state.get("seed") != seed):
        raise ValueError("续训 settings、算法或 seed 不匹配")
    if state.get("source", {}).get("source_sha256") != source_identity()["source_sha256"]:
        raise ValueError("源码已改变，拒绝把不同实现静默合并为精确续训")
    if state.get("execution") != {"device": str(device), "torch_threads": torch.get_num_threads(),
                                  "torch_version": torch.__version__}:
        raise ValueError("续训设备、线程数或 PyTorch 版本改变，不能承诺精确恢复")
    trainer.load_state_dict(state["trainer"])
    return state
