"""在 Farama MPE2 simple_adversary 上检查算法训练稳定性与输出完整性。

本脚本是短程工程基准，不把有限轮训练解释成收敛或算法优劣证明。这里的
numerically_stable 仅表示训练完成、参数/指标无 NaN/Inf，且更新次数符合预期。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import subprocess
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from scipy.stats import t as student_t  # type: ignore[import-untyped]

from benchmarks.run_state import (
    TransitionSchedule,
    atomic_json,
    load_run,
    save_run,
    source_identity,
)
from marl.algorithms import (
    MAPPO,
    MAACConfig,
    MADDPGConfig,
    MAPPOConfig,
    MASACConfig,
    QMIXConfig,
)
from marl.algorithms.base import BaseMARLAlgorithm
from marl.algorithms.maac import MAACLossConfig
from marl.algorithms.maddpg import MADDPGLossConfig
from marl.algorithms.mappo import MAPPOLossConfig
from marl.algorithms.masac import MASACLossConfig
from marl.config import AlgorithmConfig, algorithm_config_from_dict, config_to_dict
from marl.core.recurrent import RecurrentState
from marl.envs import (
    ActionKind,
    MPE2SimpleAdversaryConfig,
    build_mpe2_simple_adversary,
)
from marl.experiment import build_experiment
from marl.models import BackboneConfig, GRUBackboneConfig, LSTMBackboneConfig, MLPBackboneConfig
from marl.modules.critic import (
    AttentionQConfig,
    CentralizedValueConfig,
    IndependentQConfig,
    TwinQConfig,
)
from marl.modules.policy import (
    IndependentDeterministicConfig,
    IndependentDiscreteConfig,
    IndependentGaussianConfig,
)
from marl.runtime import SyncVectorEnv, resolve_device
from marl.training import OffPolicyCollector, OffPolicyTrainer, OnPolicyTrainer
from marl.training.off_policy import (
    ActorCriticUpdateConfig,
    ReplayConfig,
    TemperatureActorCriticUpdateConfig,
)
from marl.training.on_policy import PPOUpdateConfig

ALGORITHMS = ("maac", "mappo", "maddpg", "masac", "qmix")
REQUIRED_METRICS = {
    "maac": {
        "loss", "critic_loss", "actor_loss", "entropy", "entropy_ratio",
        "actor_gradient_norm", "critic_gradient_norm",
    },
    "mappo": {
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "actor_gradient_norm",
        "critic_gradient_norm",
        "explained_variance",
        "entropy_ratio",
    },
    "maddpg": {
        "loss", "critic_loss", "actor_loss",
        "actor_gradient_norm", "critic_gradient_norm",
    },
    "masac": {
        "loss",
        "critic_loss",
        "actor_loss",
        "alpha_loss",
        "alpha",
        "actor_gradient_norm",
        "critic_gradient_norm",
        "temperature_gradient_norm",
    },
}

ACTION_GROUPS = {
    "discrete": ("maac", "mappo"),
    "continuous": ("maddpg", "masac"),
}


@dataclass(frozen=True, slots=True)
class BenchmarkSettings:
    train_episodes: int = 8
    eval_episodes: int = 4
    num_envs: int = 4
    horizon: int = 25
    replay_batch_size: int = 32
    hidden_dim: int = 64
    eval_interval_steps: int | None = None
    metrics_interval_steps: int | None = None
    target_environment_steps: int | None = None
    layer_norm: bool = False
    normalize_targets: bool = False
    detailed_evaluation: bool = True
    evaluation_seed: int = 1_000_000
    health_value_limit: float = 10_000.0
    health_low_entropy_ratio: float = 0.01
    health_clip_rate_limit: float = 0.95
    health_warning_intervals: int = 3
    learning_starts: int | None = None
    train_every_transitions: int = 1
    gradient_steps: int = 1
    checkpoint_interval_steps: int = 1000
    actor_backbone: str = "mlp"
    critic_backbone: str = "mlp"
    sequence_length: int | None = None

    def __post_init__(self) -> None:
        if any(kind not in ("mlp", "gru", "lstm")
               for kind in (self.actor_backbone, self.critic_backbone)):
            raise ValueError("MAPPO backbone 只支持 mlp/gru/lstm")
        required = (
            self.train_episodes,
            self.eval_episodes,
            self.num_envs,
            self.horizon,
            self.replay_batch_size,
            self.hidden_dim,
        )
        if any(value < 1 for value in required):
            raise ValueError("所有基准规模参数必须大于 0")
        if (
            min(self.train_every_transitions, self.gradient_steps, self.checkpoint_interval_steps)
            < 1
        ):
            raise ValueError("更新频率、梯度步和 checkpoint 间隔必须为正")
        if self.learning_starts is not None and self.learning_starts < self.replay_batch_size:
            raise ValueError("learning_starts 不能小于 replay batch_size")
        if self.target_environment_steps is not None and self.target_environment_steps < 1:
            raise ValueError("target_environment_steps 必须大于 0 或为 None")
        for name in ("eval_interval_steps", "metrics_interval_steps"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} 必须大于 0 或为 None")

    @property
    def resolved_eval_interval_steps(self) -> int:
        """默认约 100 个评估区间，向上对齐完整 rollout，短测试至少一轮。"""
        if self.eval_interval_steps is not None:
            return self.eval_interval_steps
        quantum = self.num_envs * self.horizon
        budget = self.target_environment_steps or self.train_episodes * quantum
        return max(1, (budget + 100 * quantum - 1) // (100 * quantum)) * quantum

    @property
    def resolved_metrics_interval_steps(self) -> int:
        return self.metrics_interval_steps or self.num_envs * self.horizon


def _training_iterations(settings: BenchmarkSettings) -> int:
    """把精确环境步预算换算为完整向量 rollout/episode 次数。"""

    steps_per_iteration = settings.num_envs * settings.horizon
    for name, interval in (
        ("eval_interval_steps", settings.resolved_eval_interval_steps),
        ("metrics_interval_steps", settings.resolved_metrics_interval_steps),
    ):
        if interval % steps_per_iteration:
            raise ValueError(f"{name} 必须能被 num_envs*horizon 整除")
    if settings.target_environment_steps is None:
        return settings.train_episodes
    quotient, remainder = divmod(settings.target_environment_steps, steps_per_iteration)
    if remainder:
        raise ValueError(
            "target_environment_steps 必须能被 num_envs*horizon 整除，"
            "以保证算法使用完全相同的环境步预算"
        )
    return quotient


def _benchmark_backbone(kind: str, settings: BenchmarkSettings) -> BackboneConfig:
    """实验参数接线，不是网络工厂；Config 本身负责 build。"""
    if kind == "gru":
        return GRUBackboneConfig(hidden_dim=settings.hidden_dim)
    if kind == "lstm":
        return LSTMBackboneConfig(hidden_dim=settings.hidden_dim)
    return MLPBackboneConfig(output_dim=settings.hidden_dim, layer_norm=settings.layer_norm)


def _algorithm_config(name: str, settings: BenchmarkSettings) -> AlgorithmConfig:
    replay_steps = settings.target_environment_steps or (
        settings.train_episodes * settings.num_envs * settings.horizon
    )
    replay = ReplayConfig(
        capacity=max(2048, replay_steps),
        batch_size=settings.replay_batch_size,
    )
    hidden = settings.hidden_dim
    if name == "maac":
        return MAACConfig(
            policy=IndependentDiscreteConfig(
                backbone=MLPBackboneConfig(output_dim=hidden, layer_norm=settings.layer_norm)
            ),
            critic=AttentionQConfig(
                hidden_dim=hidden, attention_heads=4, layer_norm=settings.layer_norm
            ),
            loss=MAACLossConfig(normalize_targets=settings.normalize_targets),
            replay=replay,
            update=ActorCriticUpdateConfig(learning_rate=3e-4, max_grad_norm=10.0),
        )
    if name == "mappo":
        return MAPPOConfig(
            policy=IndependentDiscreteConfig(
                backbone=_benchmark_backbone(settings.actor_backbone, settings)
            ),
            critic=CentralizedValueConfig(
                backbone=_benchmark_backbone(settings.critic_backbone, settings)
            ),
            loss=MAPPOLossConfig(normalize_targets=settings.normalize_targets),
            update=PPOUpdateConfig(
                learning_rate=3e-4,
                epochs=2,
                mini_batch_size=min(64, settings.horizon * settings.num_envs),
                max_grad_norm=10.0,
                sequence_length=settings.sequence_length,
            ),
        )
    if name == "maddpg":
        return MADDPGConfig(
            policy=IndependentDeterministicConfig(
                hidden_dim=hidden, layer_norm=settings.layer_norm
            ),
            critic=IndependentQConfig(hidden_dim=hidden, layer_norm=settings.layer_norm),
            loss=MADDPGLossConfig(normalize_targets=settings.normalize_targets),
            replay=replay,
            update=ActorCriticUpdateConfig(learning_rate=3e-4, max_grad_norm=10.0),
        )
    if name == "masac":
        return MASACConfig(
            policy=IndependentGaussianConfig(hidden_dim=hidden, layer_norm=settings.layer_norm),
            critic=TwinQConfig(hidden_dim=hidden, layer_norm=settings.layer_norm),
            loss=MASACLossConfig(normalize_targets=settings.normalize_targets),
            replay=replay,
            update=TemperatureActorCriticUpdateConfig(learning_rate=3e-4, max_grad_norm=10.0),
        )
    if name == "qmix":
        return QMIXConfig()
    raise ValueError(f"未知算法：{name}")


@contextmanager
def _evaluation_mode(*algorithms: BaseMARLAlgorithm | None):
    """评估结束恢复每个子模块原始模式，包括已冻结 target 的 eval 标志。"""
    modes = [(module, module.training) for algorithm in algorithms if algorithm is not None
             for module in algorithm.modules()]
    try:
        for algorithm in algorithms:
            if algorithm is not None:
                algorithm.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training


def _policy_actions(
    algorithm: BaseMARLAlgorithm,
    observations: np.ndarray,
    action_mask: np.ndarray | None,
    *,
    device: torch.device,
    deterministic: bool,
    policy_state: RecurrentState = None,
) -> tuple[np.ndarray, RecurrentState]:
    observation_tensor = torch.as_tensor(
        observations, dtype=torch.float32, device=device
    ).unsqueeze(0)
    mask_tensor = (
        torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
        if action_mask is not None
        else None
    )
    with torch.inference_mode():
        if isinstance(algorithm, MAPPO) and algorithm.recurrent_policy:
            output = algorithm.policy.act(
                observation_tensor, deterministic=deterministic,
                action_mask=mask_tensor, hidden_state=policy_state,
            )
            actions, policy_state = output.actions, output.policy_state
        else:
            if policy_state is not None:
                raise ValueError("无状态策略不接受评估 hidden state")
            actions = algorithm.act(
                observation_tensor, deterministic=deterministic, action_mask=mask_tensor,
            )
    return actions.squeeze(0).cpu().numpy(), policy_state


def _gaussian_exploration(
    rng: np.random.Generator, standard_deviation: float
) -> Callable[[np.ndarray], np.ndarray]:
    """绑定本轮 MADDPG 探索尺度，避免循环闭包引用后续值。"""

    def transform(actions: np.ndarray) -> np.ndarray:
        return np.clip(
            actions + rng.normal(0.0, standard_deviation, size=actions.shape),
            -1.0,
            1.0,
        ).astype(np.float32)

    return transform


def _evaluate(
    algorithm: BaseMARLAlgorithm,
    action_kind: ActionKind,
    settings: BenchmarkSettings,
    *,
    device: torch.device,
    seed: int,
    deterministic: bool = True,
    opponent: BaseMARLAlgorithm | None = None,
    random_opponent: bool = False,
    trained_role: str | None = None,
) -> tuple[list[str], list[list[float]]]:
    with _evaluation_mode(algorithm, opponent):
        episode_returns: list[list[float]] = []
        agent_ids: list[str] | None = None
        for episode in range(settings.eval_episodes):
            rng = np.random.default_rng(seed + 10_000 + episode)
            adapter = build_mpe2_simple_adversary(
                MPE2SimpleAdversaryConfig(horizon=settings.horizon),
                action_kind=action_kind,
            )
            try:
                step = adapter.reset(seed=seed + 10_000 + episode)
                agent_ids = list(cast(tuple[str, ...], step.info["agent_ids"]))
                returns = np.zeros(adapter.spec.num_agents, dtype=np.float64)
                policy_state: RecurrentState = None
                opponent_state: RecurrentState = None
                while not step.done:
                    actions, policy_state = _policy_actions(
                        algorithm,
                        step.observations,
                        step.action_mask,
                        device=device,
                        deterministic=deterministic, policy_state=policy_state,
                    )
                    if trained_role is not None:
                        if random_opponent:
                            if action_kind == ActionKind.DISCRETE:
                                assert step.action_mask is not None
                                other_actions = np.asarray(
                                    [rng.choice(np.flatnonzero(mask)) for mask in step.action_mask]
                                )
                            else:
                                other_actions = rng.uniform(-1.0, 1.0, size=actions.shape).astype(
                                    np.float32
                                )
                        else:
                            assert opponent is not None
                            other_actions, opponent_state = _policy_actions(
                                opponent,
                                step.observations,
                                step.action_mask,
                                device=device,
                                deterministic=deterministic, policy_state=opponent_state,
                            )
                        if trained_role == "adversary":
                            actions[1:] = other_actions[1:]
                        else:
                            actions[0] = other_actions[0]
                    step = adapter.step(actions)
                    returns += step.rewards
                episode_returns.append(returns.tolist())
            finally:
                adapter.close()
        if agent_ids is None:
            raise RuntimeError("评估没有产生 episode")
        return agent_ids, episode_returns


def _cross_play_actions(
    adversary_algorithm: BaseMARLAlgorithm,
    good_team_algorithm: BaseMARLAlgorithm,
    observations: np.ndarray,
    action_mask: np.ndarray | None,
    *,
    device: torch.device,
    adversary_state: RecurrentState = None, good_state: RecurrentState = None,
) -> tuple[np.ndarray, RecurrentState, RecurrentState]:
    """组合 A 的 adversary actor 与 B 的 good-agent actors。"""

    adversary_actions, adversary_state = _policy_actions(
        adversary_algorithm,
        observations,
        action_mask,
        device=device,
        deterministic=True, policy_state=adversary_state,
    )
    good_actions, good_state = _policy_actions(
        good_team_algorithm,
        observations,
        action_mask,
        device=device,
        deterministic=True,
        policy_state=good_state,
    )
    joint_actions = good_actions.copy()
    joint_actions[0] = adversary_actions[0]
    return cast(np.ndarray, joint_actions), adversary_state, good_state


def _evaluate_cross_play(
    adversary_algorithm: BaseMARLAlgorithm,
    good_team_algorithm: BaseMARLAlgorithm,
    action_kind: ActionKind,
    settings: BenchmarkSettings,
    *,
    device: torch.device,
    seed: int,
) -> tuple[float, float]:
    with _evaluation_mode(adversary_algorithm, good_team_algorithm):
        role_returns: list[tuple[float, float]] = []
        for episode in range(settings.eval_episodes):
            adapter = build_mpe2_simple_adversary(
                MPE2SimpleAdversaryConfig(horizon=settings.horizon),
                action_kind=action_kind,
            )
            try:
                step = adapter.reset(seed=seed + episode)
                returns = np.zeros(adapter.spec.num_agents, dtype=np.float64)
                adversary_state: RecurrentState = None
                good_state: RecurrentState = None
                while not step.done:
                    actions, adversary_state, good_state = _cross_play_actions(
                        adversary_algorithm,
                        good_team_algorithm,
                        step.observations,
                        step.action_mask,
                        device=device, adversary_state=adversary_state, good_state=good_state,
                    )
                    step = adapter.step(actions)
                    returns += step.rewards
                role_returns.append((float(returns[0]), float(returns[1:].mean())))
            finally:
                adapter.close()
        values = np.asarray(role_returns, dtype=np.float64)
        return float(values[:, 0].mean()), float(values[:, 1].mean())


def _metric_summary(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    return {
        name: {
            "first": float(rows[0][name]),
            "last": float(rows[-1][name]),
            "min": float(min(row[name] for row in rows)),
            "max": float(max(row[name] for row in rows)),
        }
        for name in rows[0]
    }


def _mean_metrics(
    rows: list[dict[str, float]], weights: list[int] | None = None,
) -> dict[str, float]:
    """汇总相邻两次评估之间的训练指标，供学习曲线和稳定性图使用。"""

    if not rows:
        return {}
    common = set.intersection(*(set(row) for row in rows))
    result = {}
    for name in sorted(common):
        values = [row[name] for row in rows]
        if name.endswith(("_abs_max", "_nonfinite")):
            result[name] = float(max(values))
        elif name.endswith("_count"):
            result[name] = float(sum(values))
        else:
            result[name] = float(np.average(values, weights=weights))
    for name in tuple(result):
        if name.endswith("_clip_count"):
            prefix = name.removesuffix("_clip_count")
            result[f"{prefix}_gradient_clip_rate"] = (
                result[name] / max(1, result[f"{prefix}_step_count"])
            )
    return result


def _augment_metrics(
    metrics: dict[str, float],
    config: AlgorithmConfig,
    action_dim: int,
) -> dict[str, float]:
    """增加离散归一化熵，不把区间平均梯度冒充真实裁剪率。"""

    result = dict(metrics)
    if "entropy" in result and config.algorithm in ("maac", "mappo"):
        result["entropy_ratio"] = result["entropy"] / float(np.log(action_dim))
    del config
    return result


def _record_training_checkpoint(
    trainer: OffPolicyTrainer | OnPolicyTrainer,
    config: AlgorithmConfig,
    history: list[dict[str, Any]],
    *,
    environment_steps: int,
    elapsed_seconds: float,
    audit_path: Path | None = None,
    value_limit: float | None = None,
) -> None:
    """一次同步记录真实训练区间；不运行额外前向或改变训练随机数。"""
    previous_updates = history[-1]["updates"] if history else 0
    spec = trainer.spec if isinstance(trainer, OffPolicyTrainer) else trainer.environment.spec
    history.append({
        "environment_steps": environment_steps,
        "elapsed_seconds": elapsed_seconds,
        "updates": trainer.optimization.update_count,
        "optimizer_steps": trainer.optimization.optimizer_step_count,
        "metric_weight": trainer.optimization.update_count - previous_updates,
        "metrics": _augment_metrics(
            trainer.optimization.flush_metrics(), config, spec.action_dim
        ),
    })
    _write_audit_event(audit_path, "training", history[-1])
    metrics = history[-1]["metrics"]
    if any(not np.isfinite(value) for value in metrics.values()) or any(
        value > 0 for key, value in metrics.items() if key.endswith("_nonfinite")
    ):
        raise FloatingPointError(f"非有限训练指标，已保留审计记录: {audit_path}")
    if value_limit is not None:
        excessive = {key: value for key, value in metrics.items()
                     if key.startswith(("value_agent_", "target_agent_", "q2_agent_"))
                     and key.endswith("_abs_max") and value > value_limit}
        if excessive:
            raise RuntimeError(f"原始价值尺度超过基准安全阈值 {value_limit}: {excessive}")


def _training_metric_mean(history: list[dict[str, Any]]) -> dict[str, float]:
    measured = [point for point in history if point["metrics"] and point["metric_weight"]]
    return _mean_metrics(
        [point["metrics"] for point in measured],
        [point["metric_weight"] for point in measured],
    )


def _audit_log_path(checkpoint_dir: Path | None, name: str, seed: int) -> Path | None:
    if checkpoint_dir is None:
        return None
    directory = checkpoint_dir.parent / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}_seed_{seed}.jsonl"


def _write_audit_event(path: Path | None, kind: str, point: dict[str, Any]) -> None:
    """每次采样后落盘一行；异常退出时仍保留此前真实审计记录。"""
    if path is None:
        return
    mode = "w" if kind == "evaluation" and point["environment_steps"] == 0 else "a"
    with path.open(mode, encoding="utf-8") as stream:
        stream.write(json.dumps({"kind": kind, **point}, ensure_ascii=False) + "\n")


def _checkpoint_metadata(
    name: str,
    seed: int,
    trainer: OffPolicyTrainer | OnPolicyTrainer,
    config: AlgorithmConfig,
    settings: BenchmarkSettings,
    device: torch.device,
    checkpoint_dir: Path | None,
) -> dict[str, Any] | None:
    if checkpoint_dir is None:
        return None
    checkpoint = checkpoint_dir / f"{name}_seed_{seed}.pt"
    trainer.save_checkpoint(checkpoint)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(checkpoint.resolve()),
        "sha256": digest.hexdigest(),
        "audit_log_path": str(
            (checkpoint_dir.parent / "diagnostics" / f"{name}_seed_{seed}.jsonl").resolve()
        ),
        "config": config_to_dict(config),
        "environment": {
            "name": "mpe2.simple_adversary_v3",
            "version": version("mpe2"),
            "settings": asdict(settings),
        },
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "updates": trainer.optimization.update_count,
        "optimizer_steps": trainer.optimization.optimizer_step_count,
    }


def _evaluation_checkpoint(
    algorithm: BaseMARLAlgorithm,
    action_kind: ActionKind,
    settings: BenchmarkSettings,
    *,
    device: torch.device,
    seed: int,
    episode: int,
    environment_steps: int,
    updates: int,
    elapsed_seconds: float,
    metric_rows: list[dict[str, Any]],
    audit_path: Path | None = None,
) -> dict[str, Any]:
    """在固定评估种子上记录逐智能体回报和当前训练状态。"""

    agent_ids, evaluation_returns = _evaluate(
        algorithm,
        action_kind,
        settings,
        device=device,
        seed=settings.evaluation_seed,
    )
    # 随机评估、对手初始化不能消耗训练 RNG。共同场景种子与训练种子分离。
    comparisons: dict[str, Any] = {}
    if settings.detailed_evaluation:
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            baseline_name = "mappo" if action_kind == ActionKind.DISCRETE else "masac"
            torch.manual_seed(settings.evaluation_seed)
            # 对手不能随被测稳定化开关改变，否则 LayerNorm 消融也换了控制组。
            opponent_settings = replace(settings, layer_norm=False, normalize_targets=False)
            opponent = _algorithm_config(baseline_name, opponent_settings).build(
                algorithm.spec
            ).to(device)
            for mode in ("deterministic", "stochastic"):
                deterministic = mode == "deterministic"
                for source in ("random", "untrained"):
                    for role in ("adversary", "good_team"):
                        torch.manual_seed(settings.evaluation_seed + 1)
                        _, samples = _evaluate(
                            algorithm, action_kind, settings, device=device,
                            seed=settings.evaluation_seed, deterministic=deterministic,
                            opponent=opponent, random_opponent=source == "random",
                            trained_role=role,
                        )
                        comparisons[f"{source}/{mode}/{role}"] = samples
            torch.manual_seed(settings.evaluation_seed + 1)
            _, samples = _evaluate(
                algorithm, action_kind, settings, device=device,
                seed=settings.evaluation_seed, deterministic=False,
            )
            comparisons["self_play/stochastic"] = samples
    values = np.asarray(evaluation_returns, dtype=np.float64)
    point = {
        "episode": episode,
        "environment_steps": environment_steps,
        "updates": updates,
        "elapsed_seconds": elapsed_seconds,
        "agent_ids": agent_ids,
        "evaluation_returns": evaluation_returns,
        "fixed_opponent_evaluations": comparisons,
        "evaluation_seed": settings.evaluation_seed,
        "mean_return_per_agent": values.mean(axis=0).tolist(),
        "std_return_per_agent": values.std(axis=0).tolist(),
        "mean_return_by_role": {
            "adversary": float(values[:, 0].mean()),
            "good_team": float(values[:, 1:].mean()),
        },
        "std_return_by_role": {
            "adversary": float(values[:, 0].std()),
            "good_team": float(values[:, 1:].std()),
        },
        "mean_metrics_since_previous_evaluation": _training_metric_mean(metric_rows),
    }
    _write_audit_event(audit_path, "evaluation", point)
    return point


def _parameters_are_finite(algorithm: BaseMARLAlgorithm) -> bool:
    return all(
        bool(torch.isfinite(parameter).all().item())
        for parameter in algorithm.parameters()
    )


def _seed_result(
    name: str,
    seed: int,
    algorithm: BaseMARLAlgorithm,
    training_history: list[dict[str, Any]],
    update_count: int,
    optimizer_step_count: int,
    action_kind: ActionKind,
    settings: BenchmarkSettings,
    device: torch.device,
    evaluation_history: list[dict[str, Any]],
    environment_steps: int,
    wall_clock_seconds: float,
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    del action_kind, device
    if not evaluation_history:
        raise RuntimeError("训练没有产生定期评估记录")
    final_evaluation = evaluation_history[-1]
    metrics = [point["metrics"] for point in training_history if point["metrics"]]
    agent_ids = final_evaluation["agent_ids"]
    evaluation_returns = final_evaluation["evaluation_returns"]
    required = REQUIRED_METRICS[name]
    observed = set.intersection(*(set(row) for row in metrics)) if metrics else set()
    finite_metrics = bool(metrics) and all(
        np.isfinite(value) for row in metrics for value in row.values()
    )
    finite_returns = bool(evaluation_returns) and bool(
        np.isfinite(np.asarray(evaluation_returns)).all()
    )
    finite_history = all(
        np.isfinite(np.asarray(checkpoint["evaluation_returns"])).all()
        and all(
            np.isfinite(value)
            for value in checkpoint["mean_metrics_since_previous_evaluation"].values()
        )
        for checkpoint in evaluation_history
    )
    output_complete = (
        required <= observed
        and len(agent_ids) == algorithm.spec.num_agents
        and all(len(values) == algorithm.spec.num_agents for values in evaluation_returns)
    )
    numerically_stable = (
        finite_metrics
        and finite_returns
        and finite_history
        and _parameters_are_finite(algorithm)
        and update_count > 0
    )
    return {
        "seed": seed,
        "updates": update_count,
        "optimizer_steps": optimizer_step_count,
        "required_metrics": sorted(required),
        "observed_metrics": sorted(observed),
        "metric_summary": _metric_summary(metrics),
        "agent_ids": agent_ids,
        "evaluation_returns": evaluation_returns,
        "mean_evaluation_return_per_agent": np.mean(
            np.asarray(evaluation_returns), axis=0
        ).tolist(),
        "evaluation_history": evaluation_history,
        "training_history": training_history,
        "convergence_assessment": "not_established",
        "audit_counts": {
            "training_points": len(training_history),
            "evaluation_points": len(evaluation_history),
        },
        "environment_steps": environment_steps,
        "wall_clock_seconds": wall_clock_seconds,
        "updates_per_second": update_count / max(wall_clock_seconds, 1e-12),
        "gradient_clip_rate": {
            key.removesuffix("_gradient_clip_rate"): value
            for key, value in _training_metric_mean(training_history).items()
            if key.endswith("_gradient_clip_rate")
        },
        "gradient_clip_rate_available": True,
        "execution_complete": (
            output_complete and environment_steps ==
            _training_iterations(settings) * settings.num_envs * settings.horizon
            and evaluation_history[0]["environment_steps"] == 0
            and evaluation_history[-1]["environment_steps"] == environment_steps
        ),
        "numerics_finite": numerically_stable,
        "optimization_health": _optimization_health(training_history, settings),
        "learning_evaluation": {"convergence": "not_established"},
        "checkpoint": checkpoint,
        "parameters_finite": _parameters_are_finite(algorithm),
        "output_complete": output_complete,
        "numerically_stable": numerically_stable,
    }


def _optimization_health(
    history: list[dict[str, Any]], settings: BenchmarkSettings
) -> dict[str, Any]:
    """经验警报不是所有环境通用的理论边界；保留触发点与原始尺度。"""
    warnings = []
    streaks: dict[str, int] = {}
    for point in history:
        for key, value in point["metrics"].items():
            if key.startswith(("value_", "target_", "q2_")) and key.endswith("_abs_max"):
                if value > settings.health_value_limit:
                    warnings.append({"step": point["environment_steps"],
                                     "metric": key, "value": value,
                                     "threshold": settings.health_value_limit})
            low_entropy = key == "entropy_ratio" and value < settings.health_low_entropy_ratio
            high_clip = (key.endswith("_gradient_clip_rate")
                         and value > settings.health_clip_rate_limit)
            streaks[key] = streaks.get(key, 0) + 1 if low_entropy or high_clip else 0
            if streaks[key] == settings.health_warning_intervals:
                warnings.append({"step": point["environment_steps"], "metric": key,
                                 "value": value, "consecutive_intervals": streaks[key]})
    return {"status": "warning" if warnings else "no_warning_detected",
            "scale_warnings": warnings,
            "note": "阈值为本基准经验警报；无警报不等于收敛或健康证明"}


def _train_off_policy(
    name: str,
    seed: int,
    settings: BenchmarkSettings,
    device: torch.device,
    checkpoint_dir: Path | None,
    *, resume: bool = False,
) -> tuple[dict[str, Any], BaseMARLAlgorithm]:
    action_kind = (
        ActionKind.DISCRETE
        if name == "maac"
        else ActionKind.CONTINUOUS
    )
    adapters = [
        build_mpe2_simple_adversary(
            MPE2SimpleAdversaryConfig(horizon=settings.horizon),
            action_kind=action_kind,
        )
        for _ in range(settings.num_envs)
    ]
    vector = SyncVectorEnv(adapters)
    rng = np.random.default_rng(seed)
    schedule = TransitionSchedule(
        settings.learning_starts or settings.replay_batch_size,
        settings.train_every_transitions, settings.gradient_steps,
    )
    try:
        config = _algorithm_config(name, settings)
        experiment = build_experiment(vector.spec, config, device=device, seed=seed)
        trainer = experiment.trainer
        if not isinstance(trainer, OffPolicyTrainer):
            raise TypeError(f"{name} 应使用 OffPolicyTrainer")
        trainer.optimization.sync_metrics = False
        collector = OffPolicyCollector(vector, experiment.algorithm, device=device)
        audit_path = _audit_log_path(checkpoint_dir, name, seed)
        training_history: list[dict[str, Any]] = []
        evaluation_history: list[dict[str, Any]] = []
        metric_start = 0
        environment_steps = 0
        evaluation_history.append(
            _evaluation_checkpoint(
                experiment.algorithm,
                action_kind,
                settings,
                device=device,
                seed=seed,
                episode=0,
                environment_steps=0,
                updates=0,
                elapsed_seconds=0.0,
                metric_rows=[],
                audit_path=None if resume else audit_path,
            )
        )
        started = time.perf_counter()
        iterations = _training_iterations(settings)
        eval_interval = settings.resolved_eval_interval_steps
        metrics_interval = settings.resolved_metrics_interval_steps
        next_evaluation_step = eval_interval
        next_metrics_step = metrics_interval
        start_episode = 0
        run_path = (checkpoint_dir.parent / "resume" / f"{name}_seed_{seed}.pt"
                    if checkpoint_dir is not None else None)
        if resume and run_path is not None and run_path.exists():
            restored = load_run(run_path, trainer, asdict(settings),
                                name=name, seed=seed, device=device)
            start_episode, environment_steps = restored["episode"], restored["environment_steps"]
            training_history, evaluation_history = (
                restored["training_history"], restored["evaluation_history"]
            )
            metric_start = restored["metric_start"]
            rng.bit_generator.state = restored["exploration_rng"]
            schedule = TransitionSchedule(**restored["schedule"])
            started -= restored["elapsed_seconds"]
            next_evaluation_step = (environment_steps // eval_interval + 1) * eval_interval
            next_metrics_step = (environment_steps // metrics_interval + 1) * metrics_interval
            _restore_audit(audit_path, training_history, evaluation_history)
        for episode in range(start_episode, iterations):
            seeds = [
                seed + episode * settings.num_envs + environment_index
                for environment_index in range(settings.num_envs)
            ]
            action_transform = None
            if name == "maddpg":
                exploration = 0.3 * max(
                    0.1,
                    1.0 - episode / max(iterations - 1, 1),
                )

                action_transform = _gaussian_exploration(rng, exploration)
            for transitions in collector.rollout(
                seeds,
                action_transform=action_transform,
            ):
                trainer.record_many(transitions)
                for _ in range(schedule.add(len(transitions))):
                    trainer.update()
                environment_steps += len(transitions)
            completed_episode = episode + 1
            evaluate = (
                environment_steps >= next_evaluation_step
                or completed_episode == iterations
            )
            if environment_steps >= next_metrics_step or evaluate:
                _record_training_checkpoint(
                    trainer, config, training_history,
                    environment_steps=environment_steps,
                    elapsed_seconds=time.perf_counter() - started,
                    audit_path=audit_path,
                    value_limit=settings.health_value_limit,
                )
                while next_metrics_step <= environment_steps:
                    next_metrics_step += metrics_interval
            if evaluate:
                evaluation_history.append(
                    _evaluation_checkpoint(
                        experiment.algorithm,
                        action_kind,
                        settings,
                        device=device,
                        seed=seed,
                        episode=completed_episode,
                        environment_steps=environment_steps,
                        updates=trainer.optimization.update_count,
                        elapsed_seconds=time.perf_counter() - started,
                        metric_rows=training_history[metric_start:],
                        audit_path=audit_path,
                    )
                )
                metric_start = len(training_history)
                while next_evaluation_step <= environment_steps:
                    next_evaluation_step += eval_interval
            if run_path is not None and evaluate:
                save_run(
                    run_path, trainer, asdict(settings), name=name, seed=seed,
                    episode=completed_episode, environment_steps=environment_steps,
                    training_history=training_history, evaluation_history=evaluation_history,
                    metric_start=metric_start, elapsed_seconds=time.perf_counter() - started,
                    exploration_rng=dict(rng.bit_generator.state), schedule=asdict(schedule),
                    checkpoint_interval=settings.checkpoint_interval_steps,
                    total_steps=iterations * settings.num_envs * settings.horizon,
                )
        checkpoint = _checkpoint_metadata(
            name,
            seed,
            trainer,
            config,
            settings,
            device,
            checkpoint_dir,
        )
        result = _seed_result(
            name,
            seed,
            experiment.algorithm,
            training_history,
            trainer.optimization.update_count,
            trainer.optimization.optimizer_step_count,
            action_kind,
            settings,
            device,
            evaluation_history,
            environment_steps,
            time.perf_counter() - started,
            checkpoint,
        )
        return result, experiment.algorithm
    finally:
        for adapter in adapters:
            adapter.close()


def _train_mappo(
    seed: int,
    settings: BenchmarkSettings,
    device: torch.device,
    checkpoint_dir: Path | None,
    *, resume: bool = False,
) -> tuple[dict[str, Any], BaseMARLAlgorithm]:
    adapters = [
        build_mpe2_simple_adversary(
            MPE2SimpleAdversaryConfig(horizon=settings.horizon),
            action_kind=ActionKind.DISCRETE,
        )
        for _ in range(settings.num_envs)
    ]
    vector = SyncVectorEnv(adapters)
    try:
        config = cast(MAPPOConfig, _algorithm_config("mappo", settings))
        experiment = build_experiment(vector, config, device=device, seed=seed)
        trainer = experiment.trainer
        if not isinstance(trainer, OnPolicyTrainer):
            raise TypeError("MAPPO 应使用 OnPolicyTrainer")
        trainer.optimization.sync_metrics = False
        audit_path = _audit_log_path(checkpoint_dir, "mappo", seed)
        training_history: list[dict[str, Any]] = []
        evaluation_history: list[dict[str, Any]] = []
        metric_start = 0
        environment_steps = 0
        evaluation_history.append(
            _evaluation_checkpoint(
                experiment.algorithm,
                ActionKind.DISCRETE,
                settings,
                device=device,
                seed=seed,
                episode=0,
                environment_steps=0,
                updates=0,
                elapsed_seconds=0.0,
                metric_rows=[],
                audit_path=None if resume else audit_path,
            )
        )
        started = time.perf_counter()
        iterations = _training_iterations(settings)
        eval_interval = settings.resolved_eval_interval_steps
        metrics_interval = settings.resolved_metrics_interval_steps
        next_evaluation_step = eval_interval
        next_metrics_step = metrics_interval
        start_episode = 0
        run_path = (checkpoint_dir.parent / "resume" / f"mappo_seed_{seed}.pt"
                    if checkpoint_dir is not None else None)
        if resume and run_path is not None and run_path.exists():
            restored = load_run(run_path, trainer, asdict(settings),
                                name="mappo", seed=seed, device=device)
            start_episode, environment_steps = restored["episode"], restored["environment_steps"]
            training_history, evaluation_history = (
                restored["training_history"], restored["evaluation_history"]
            )
            metric_start = restored["metric_start"]
            started -= restored["elapsed_seconds"]
            next_evaluation_step = (environment_steps // eval_interval + 1) * eval_interval
            next_metrics_step = (environment_steps // metrics_interval + 1) * metrics_interval
            _restore_audit(audit_path, training_history, evaluation_history)
        for episode in range(start_episode, iterations):
            seeds = [
                seed + episode * settings.num_envs + index
                for index in range(settings.num_envs)
            ]
            trainer.train_rollout(seeds)
            environment_steps += settings.num_envs * settings.horizon
            completed_episode = episode + 1
            evaluate = (
                environment_steps >= next_evaluation_step
                or completed_episode == iterations
            )
            if environment_steps >= next_metrics_step or evaluate:
                _record_training_checkpoint(
                    trainer, config, training_history,
                    environment_steps=environment_steps,
                    elapsed_seconds=time.perf_counter() - started,
                    audit_path=audit_path,
                    value_limit=settings.health_value_limit,
                )
                while next_metrics_step <= environment_steps:
                    next_metrics_step += metrics_interval
            if evaluate:
                evaluation_history.append(
                    _evaluation_checkpoint(
                        experiment.algorithm,
                        ActionKind.DISCRETE,
                        settings,
                        device=device,
                        seed=seed,
                        episode=completed_episode,
                        environment_steps=environment_steps,
                        updates=trainer.optimization.update_count,
                        elapsed_seconds=time.perf_counter() - started,
                        metric_rows=training_history[metric_start:],
                        audit_path=audit_path,
                    )
                )
                metric_start = len(training_history)
                while next_evaluation_step <= environment_steps:
                    next_evaluation_step += eval_interval
            if run_path is not None and evaluate:
                save_run(
                    run_path, trainer, asdict(settings), name="mappo", seed=seed,
                    episode=completed_episode, environment_steps=environment_steps,
                    training_history=training_history, evaluation_history=evaluation_history,
                    metric_start=metric_start, elapsed_seconds=time.perf_counter() - started,
                    exploration_rng=None, schedule=None,
                    checkpoint_interval=settings.checkpoint_interval_steps,
                    total_steps=iterations * settings.num_envs * settings.horizon,
                )
        checkpoint = _checkpoint_metadata(
            "mappo",
            seed,
            trainer,
            config,
            settings,
            device,
            checkpoint_dir,
        )
        result = _seed_result(
            "mappo",
            seed,
            experiment.algorithm,
            training_history,
            trainer.optimization.update_count,
            trainer.optimization.optimizer_step_count,
            ActionKind.DISCRETE,
            settings,
            device,
            evaluation_history,
            environment_steps,
            time.perf_counter() - started,
            checkpoint,
        )
        return result, experiment.algorithm
    finally:
        for adapter in adapters:
            adapter.close()


def _qmix_incompatibility(settings: BenchmarkSettings) -> dict[str, Any]:
    adapter = build_mpe2_simple_adversary(
        MPE2SimpleAdversaryConfig(horizon=settings.horizon),
        action_kind=ActionKind.DISCRETE,
    )
    try:
        try:
            _algorithm_config("qmix", settings).build(adapter.spec)
        except ValueError as error:
            if "共享" not in str(error):
                raise
            return {
                "algorithm": "qmix",
                "status": "expected_incompatible",
                "reason": str(error),
                "output_complete": True,
                "numerically_stable": None,
            }
        raise AssertionError("QMIX 不应接受 individual-reward 非合作环境")
    finally:
        adapter.close()


def _comparison_groups(results: list[dict[str, Any]]) -> dict[str, Any]:
    """只在同一动作空间内汇总回报与排名。"""

    passed = {
        result["algorithm"]: result
        for result in results
        if result["status"] == "passed"
    }
    groups: dict[str, Any] = {}
    for group, algorithm_names in ACTION_GROUPS.items():
        scores: dict[str, Any] = {}
        for name in algorithm_names:
            if name not in passed:
                continue
            role_values = {
                role: np.asarray(
                    [
                        seed["evaluation_history"][-1]["mean_return_by_role"][role]
                        for seed in passed[name]["seeds"]
                    ],
                    dtype=np.float64,
                )
                for role in ("adversary", "good_team")
            }
            scores[name] = {
                role: {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "ci95": (
                        float(student_t.ppf(0.975, len(values) - 1)
                              * values.std(ddof=1) / np.sqrt(len(values)))
                        if len(values) > 1
                        else 0.0
                    ),
                }
                for role, values in role_values.items()
            }
        groups[group] = {
            "algorithms": list(scores),
            "scores": scores,
            "rankings": {
                role: sorted(
                    scores,
                    key=lambda name: scores[name][role]["mean"],
                    reverse=True,
                )
                for role in ("adversary", "good_team")
            },
        }
    return groups


def _cross_play_report(
    trained: dict[tuple[str, int], BaseMARLAlgorithm | Path],
    settings: BenchmarkSettings,
    device: torch.device,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    fixed_seed = 700_000
    for group, algorithm_names in ACTION_GROUPS.items():
        policies = [
            (name, seed, trained[(name, seed)])
            for name in algorithm_names
            for seed in sorted(seed for algorithm, seed in trained if algorithm == name)
        ]
        if not policies:
            continue
        labels = [f"{name}:seed={seed}" for name, seed, _ in policies]
        adversary_matrix: list[list[float]] = []
        good_team_matrix: list[list[float]] = []
        action_kind = (
            ActionKind.DISCRETE if group == "discrete" else ActionKind.CONTINUOUS
        )
        for _, _, adversary_source in policies:
            adversary_algorithm = _load_policy(adversary_source, settings, action_kind, device)
            adversary_row: list[float] = []
            good_team_row: list[float] = []
            for _, _, good_source in policies:
                good_team_algorithm = _load_policy(good_source, settings, action_kind, device)
                adversary_return, good_team_return = _evaluate_cross_play(
                    adversary_algorithm,
                    good_team_algorithm,
                    action_kind,
                    settings,
                    device=device,
                    seed=fixed_seed,
                )
                adversary_row.append(adversary_return)
                good_team_row.append(good_team_return)
                if good_team_algorithm is not adversary_algorithm:
                    good_team_algorithm.cpu()
            adversary_matrix.append(adversary_row)
            good_team_matrix.append(good_team_row)
            adversary_algorithm.cpu()
        reports.append(
            {
                "action_kind": group,
                "evaluation_seed_base": fixed_seed,
                "adversary_policies": labels,
                "good_team_policies": labels,
                "adversary_returns": adversary_matrix,
                "good_team_returns": good_team_matrix,
            }
        )
    return reports


def _load_policy(
    source: BaseMARLAlgorithm | Path, settings: BenchmarkSettings,
    action_kind: ActionKind, device: torch.device,
) -> BaseMARLAlgorithm:
    if not isinstance(source, Path):
        return source.to(device)
    state = torch.load(source, map_location="cpu", weights_only=False)
    config = algorithm_config_from_dict(state["config"])
    adapter = build_mpe2_simple_adversary(
        MPE2SimpleAdversaryConfig(horizon=settings.horizon), action_kind=action_kind
    )
    try:
        algorithm = config.build(adapter.spec)
    finally:
        adapter.close()
    algorithm.load_state_dict(state["algorithm"])
    return algorithm.to(device).eval()


def write_summary_csv(report: dict[str, Any], output: Path) -> None:
    """写出每算法/seed 的终局分角色结果，便于外部统计。"""

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "action_kind", "algorithm", "seed", "environment_steps", "updates",
        "optimizer_steps", "adversary_return", "good_team_return",
        "wall_clock_seconds", "steps_per_second", "output_complete",
        "numerically_stable", "checkpoint_sha256",
    )
    group_by_algorithm = {
        name: group
        for group, names in ACTION_GROUPS.items()
        for name in names
    }
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in report["results"]:
            if result["status"] != "passed":
                continue
            for seed in result["seeds"]:
                final = seed["evaluation_history"][-1]
                checkpoint = seed.get("checkpoint") or {}
                writer.writerow(
                    {
                        "action_kind": group_by_algorithm[result["algorithm"]],
                        "algorithm": result["algorithm"],
                        "seed": seed["seed"],
                        "environment_steps": seed["environment_steps"],
                        "updates": seed["updates"],
                        "optimizer_steps": seed["optimizer_steps"],
                        "adversary_return": final["mean_return_by_role"]["adversary"],
                        "good_team_return": final["mean_return_by_role"]["good_team"],
                        "wall_clock_seconds": seed["wall_clock_seconds"],
                        "steps_per_second": seed["environment_steps"]
                        / max(seed["wall_clock_seconds"], 1e-12),
                        "output_complete": seed["output_complete"],
                        "numerically_stable": seed["numerically_stable"],
                        "checkpoint_sha256": checkpoint.get("sha256", ""),
                    }
                )


def _restore_audit(
    path: Path | None, training: list[dict[str, Any]], evaluation: list[dict[str, Any]],
) -> None:
    if path is None:
        return
    if path.exists():
        path.rename(path.with_suffix(f".before-resume-{time.time_ns()}.jsonl"))
    events = [(p["environment_steps"], 0, "training", p) for p in training]
    events.extend((p["environment_steps"], 1, "evaluation", p) for p in evaluation)
    for _, _, kind, point in sorted(events, key=lambda item: item[:2]):
        _write_audit_event(path, kind, point)


def _run_one(
    name: str, seed: int, settings: BenchmarkSettings, device: torch.device,
    checkpoint_dir: Path | None, resume: bool, threads: int,
) -> tuple[dict[str, Any], BaseMARLAlgorithm | Path | None]:
    """一个算法×seed 一个独立任务；失败归档，其他任务继续。"""
    torch.set_num_threads(threads)
    completed = (checkpoint_dir.parent / "completed" / f"{name}_seed_{seed}.json"
                 if checkpoint_dir is not None else None)
    try:
        if resume and completed is not None and completed.exists():
            saved = json.loads(completed.read_text(encoding="utf-8"))
            if (saved["settings"] != asdict(settings) or
                    saved["source"]["source_sha256"] != source_identity()["source_sha256"]):
                raise ValueError("已完成任务的源码/配置不同，不能跳过")
            if saved.get("execution") != {"device": str(device), "torch_threads": threads}:
                raise ValueError("已完成任务的设备/线程配置不同")
            result = saved["result"]
            checkpoint = Path(result["checkpoint"]["path"])
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if digest != result["checkpoint"]["sha256"]:
                raise ValueError("已完成 checkpoint 校验失败")
            if not result["execution_complete"] or not result["numerics_finite"]:
                raise ValueError("已完成任务不满足完整性/有限值要求")
            return result, checkpoint
        resume = (resume and checkpoint_dir is not None
                  and (checkpoint_dir.parent / "resume" / f"{name}_seed_{seed}.pt").is_file())
        if name == "mappo":
            result, algorithm = _train_mappo(seed, settings, device, checkpoint_dir, resume=resume)
        else:
            result, algorithm = _train_off_policy(
                name, seed, settings, device, checkpoint_dir, resume=resume
            )
        result["source"] = source_identity()
        if completed is not None:
            atomic_json(completed, {"settings": asdict(settings), "source": source_identity(),
                                    "execution": {"device": str(device), "torch_threads": threads},
                                    "result": result})
            return result, Path(result["checkpoint"]["path"])
        return result, algorithm.cpu()
    except Exception as error:  # noqa: BLE001 - 每个任务保留异常与未完成标记。
        result = {"seed": seed, "status": "failed", "execution_complete": False,
                  "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
        if checkpoint_dir is not None:
            atomic_json(checkpoint_dir.parent / "failures" / f"{name}_seed_{seed}.json", result)
        return result, None


def run_benchmark(
    algorithms: list[str], seeds: list[int], settings: BenchmarkSettings, *,
    device: torch.device, artifact_dir: Path | None = None,
    workers: int = 1, resume: bool = False, threads: int = 1,
) -> dict[str, Any]:
    """复用相同种子任务；Windows spawn 并行，不引入第二套训练循环。"""
    if workers < 1 or threads < 1:
        raise ValueError("workers/threads 必须为正")
    if device.type == "cuda" and workers != 1:
        raise ValueError("单张 CUDA 显卡只允许一个训练 worker")
    if (workers > 1 or resume) and artifact_dir is None:
        raise ValueError("并行或续训必须指定 artifact_dir")
    if workers > 1 and threads != 1:
        raise ValueError("多 worker 模式每个进程固定一个 Torch 线程")
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    trained: dict[tuple[str, int], BaseMARLAlgorithm | Path] = {}
    checkpoint_dir = artifact_dir / "checkpoints" if artifact_dir is not None else None
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    tasks = [(name, seed) for name in algorithms if name != "qmix" for seed in seeds]
    if workers == 1:
        for name, seed in tasks:
            row, policy = _run_one(name, seed, settings, device, checkpoint_dir, resume, threads)
            rows[name, seed] = row
            if policy is not None:
                trained[name, seed] = policy
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = {pool.submit(_run_one, name, seed, settings, device,
                                   checkpoint_dir, resume, threads): (name, seed)
                       for name, seed in tasks}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    row, policy = future.result()
                except Exception as error:  # noqa: BLE001 - 包括 worker 意外退出。
                    row, policy = {"seed": key[1], "status": "failed",
                                   "error": f"{type(error).__name__}: {error}"}, None
                rows[key] = row
                if policy is not None:
                    trained[key] = policy
    for name in algorithms:
        if name == "qmix":
            results.append(_qmix_incompatibility(settings))
            continue
        seed_results = [rows[name, seed] for seed in seeds]
        failures = [row for row in seed_results if "error" in row]
        success = [row for row in seed_results if "error" not in row]
        results.append({
            "algorithm": name, "status": "failed" if failures else "passed", "seeds": success,
            "failures": failures,
            "output_complete": not failures and all(row["output_complete"] for row in success),
            "numerically_stable": not failures and all(row["numerics_finite"] for row in success),
        })
    report = {
        "report_schema_version": 3, "benchmark": "Farama MPE2 simple_adversary_v3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "device": str(device),
        "source": source_identity(), "workers": workers, "torch_threads_per_worker": threads,
        "gpu_telemetry": _gpu_telemetry(),
        "versions": {package: version(package)
                     for package in ("gymnasium", "pettingzoo", "mpe2", "torch")},
        "settings": asdict(settings),
        "audit_schedule": {"eval_interval_steps": settings.resolved_eval_interval_steps,
                           "metrics_interval_steps": settings.resolved_metrics_interval_steps,
                           "metrics_weighting": "algorithm_update_count",
                           "gradient_clip_rate": "actual clipped steps / optimizer steps"},
        "seeds": seeds,
        "stability_definition": "finite is not convergence; inspect health and opponents",
        "results": results, "comparison_groups": _comparison_groups(results),
        "cross_play": _cross_play_report(trained, settings, device),
    }
    report["total_wall_clock_seconds"] = time.perf_counter() - started
    return report


def _gpu_telemetry() -> dict[str, Any]:
    try:
        sample = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,clocks.sm,temperature.gpu,power.draw,utilization.gpu",
             "--format=csv,noheader"], text=True, timeout=5).strip()
        return {"sample": sample, "note": "结束时快照，不用于解释整段训练耗时"}
    except (OSError, subprocess.SubprocessError):
        return {"status": "unavailable"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPE2 simple_adversary 非合作博弈训练稳定性基准")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHMS,
        default=list(ALGORITHMS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17])
    parser.add_argument("--train-episodes", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--replay-batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--eval-interval-steps",
        type=int,
        help="策略评估间隔；默认约 100 个区间，对齐完整 rollout",
    )
    parser.add_argument(
        "--metrics-interval-steps",
        type=int,
        help="训练诊断间隔；默认每个完整 rollout，不额外执行策略评估",
    )
    parser.add_argument("--environment-steps", type=int)
    parser.add_argument("--layer-norm", action="store_true")
    parser.add_argument("--normalize-targets", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--learning-starts", type=int)
    parser.add_argument("--train-every-transitions", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=1000)
    parser.add_argument("--actor-backbone", choices=("mlp", "gru", "lstm"), default="mlp")
    parser.add_argument("--critic-backbone", choices=("mlp", "gru", "lstm"), default="mlp")
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument(
        "--basic-evaluation",
        action="store_true",
        help="仅自博弈评估；用于单独测量运行路径，不用于最终学习验收",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results") / "mpe2_simple_adversary.json",
    )
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = BenchmarkSettings(
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        num_envs=args.num_envs,
        horizon=args.horizon,
        replay_batch_size=args.replay_batch_size,
        hidden_dim=args.hidden_dim,
        eval_interval_steps=args.eval_interval_steps,
        metrics_interval_steps=args.metrics_interval_steps,
        target_environment_steps=args.environment_steps,
        layer_norm=args.layer_norm,
        normalize_targets=args.normalize_targets,
        detailed_evaluation=not args.basic_evaluation,
        learning_starts=args.learning_starts,
        train_every_transitions=args.train_every_transitions,
        gradient_steps=args.gradient_steps,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
        actor_backbone=args.actor_backbone, critic_backbone=args.critic_backbone,
        sequence_length=args.sequence_length,
    )
    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    artifact_dir = args.artifact_dir or (args.output.parent / f"{args.output.stem}_artifacts")
    report = run_benchmark(
        list(dict.fromkeys(args.algorithms)),
        list(dict.fromkeys(args.seeds)),
        settings,
        device=device,
        artifact_dir=artifact_dir,
        workers=args.workers,
        resume=args.resume,
        threads=args.torch_threads,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    csv_output = args.csv_output or args.output.with_suffix(".csv")
    write_summary_csv(report, csv_output)
    for result in report["results"]:
        print(
            f"algorithm={result['algorithm']} status={result['status']} "
            f"finite={result.get('numerically_stable')} "
            f"complete={result.get('output_complete')}"
        )
    print(f"report={args.output.resolve()}")
    print(f"summary={csv_output.resolve()}")
    if any(result["status"] == "failed" for result in report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
