"""在 Farama MPE2 simple_adversary 上检查算法训练稳定性与输出完整性。

本脚本是短程工程基准，不把有限轮训练解释成收敛或算法优劣证明。这里的
numerically_stable 仅表示训练完成、参数/指标无 NaN/Inf，且更新次数符合预期。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from marl.algorithms import (
    MAACConfig,
    MADDPGConfig,
    MAPPOConfig,
    MASACConfig,
    QMIXConfig,
)
from marl.algorithms.base import BaseMARLAlgorithm
from marl.config import AlgorithmConfig
from marl.envs import (
    ActionKind,
    MPE2SimpleAdversaryConfig,
    Transition,
    build_mpe2_simple_adversary,
)
from marl.experiment import build_experiment
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
from marl.training import OffPolicyTrainer, OnPolicyTrainer
from marl.training.off_policy import OffPolicyUpdateConfig, ReplayConfig
from marl.training.on_policy import PPOUpdateConfig

ALGORITHMS = ("maac", "mappo", "maddpg", "masac", "qmix")
REQUIRED_METRICS = {
    "maac": {"loss", "critic_loss", "actor_loss", "entropy", "gradient_norm"},
    "mappo": {
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "gradient_norm",
        "explained_variance",
    },
    "maddpg": {"loss", "critic_loss", "actor_loss", "gradient_norm"},
    "masac": {
        "loss",
        "critic_loss",
        "actor_loss",
        "alpha_loss",
        "alpha",
        "gradient_norm",
    },
}


@dataclass(frozen=True, slots=True)
class BenchmarkSettings:
    train_episodes: int = 8
    eval_episodes: int = 4
    num_envs: int = 4
    horizon: int = 25
    replay_batch_size: int = 32
    hidden_dim: int = 64
    eval_interval: int = 10

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(int(value) < 1 for value in values.values()):
            raise ValueError("所有基准规模参数必须大于 0")


def _algorithm_config(name: str, settings: BenchmarkSettings) -> AlgorithmConfig:
    replay = ReplayConfig(
        capacity=max(2048, settings.train_episodes * settings.horizon),
        batch_size=settings.replay_batch_size,
    )
    off_policy_update = OffPolicyUpdateConfig(learning_rate=3e-4, max_grad_norm=10.0)
    hidden = settings.hidden_dim
    if name == "maac":
        return MAACConfig(
            policy=IndependentDiscreteConfig(hidden_dim=hidden),
            critic=AttentionQConfig(hidden_dim=hidden, attention_heads=4),
            replay=replay,
            update=off_policy_update,
        )
    if name == "mappo":
        return MAPPOConfig(
            policy=IndependentDiscreteConfig(hidden_dim=hidden),
            critic=CentralizedValueConfig(hidden_dim=hidden),
            update=PPOUpdateConfig(
                learning_rate=3e-4,
                epochs=2,
                mini_batch_size=min(64, settings.horizon * settings.num_envs),
                max_grad_norm=10.0,
            ),
        )
    if name == "maddpg":
        return MADDPGConfig(
            policy=IndependentDeterministicConfig(hidden_dim=hidden),
            critic=IndependentQConfig(hidden_dim=hidden),
            replay=replay,
            update=off_policy_update,
        )
    if name == "masac":
        return MASACConfig(
            policy=IndependentGaussianConfig(hidden_dim=hidden),
            critic=TwinQConfig(hidden_dim=hidden),
            replay=replay,
            update=off_policy_update,
        )
    if name == "qmix":
        return QMIXConfig()
    raise ValueError(f"未知算法：{name}")


def _policy_actions(
    algorithm: BaseMARLAlgorithm,
    observations: np.ndarray,
    action_mask: np.ndarray | None,
    *,
    device: torch.device,
    deterministic: bool,
) -> np.ndarray:
    observation_tensor = torch.as_tensor(
        observations, dtype=torch.float32, device=device
    ).unsqueeze(0)
    mask_tensor = (
        torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
        if action_mask is not None
        else None
    )
    with torch.inference_mode():
        actions = algorithm.act(
            observation_tensor,
            deterministic=deterministic,
            action_mask=mask_tensor,
        )
    return actions.squeeze(0).cpu().numpy()


def _evaluate(
    algorithm: BaseMARLAlgorithm,
    action_kind: ActionKind,
    settings: BenchmarkSettings,
    *,
    device: torch.device,
    seed: int,
) -> tuple[list[str], list[list[float]]]:
    episode_returns: list[list[float]] = []
    agent_ids: list[str] | None = None
    for episode in range(settings.eval_episodes):
        adapter = build_mpe2_simple_adversary(
            MPE2SimpleAdversaryConfig(horizon=settings.horizon),
            action_kind=action_kind,
        )
        try:
            step = adapter.reset(seed=seed + 10_000 + episode)
            agent_ids = list(cast(tuple[str, ...], step.info["agent_ids"]))
            returns = np.zeros(adapter.spec.num_agents, dtype=np.float64)
            while not step.done:
                actions = _policy_actions(
                    algorithm,
                    step.observations,
                    step.action_mask,
                    device=device,
                    deterministic=True,
                )
                step = adapter.step(actions)
                returns += step.rewards
            episode_returns.append(returns.tolist())
        finally:
            adapter.close()
    if agent_ids is None:
        raise RuntimeError("评估没有产生 episode")
    return agent_ids, episode_returns


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


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """汇总相邻两次评估之间的训练指标，供学习曲线和稳定性图使用。"""

    if not rows:
        return {}
    common = set.intersection(*(set(row) for row in rows))
    return {
        name: float(np.mean([row[name] for row in rows]))
        for name in sorted(common)
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
    metric_rows: list[dict[str, float]],
) -> dict[str, Any]:
    """在固定评估种子上记录逐智能体回报和当前训练状态。"""

    agent_ids, evaluation_returns = _evaluate(
        algorithm,
        action_kind,
        settings,
        device=device,
        seed=seed,
    )
    values = np.asarray(evaluation_returns, dtype=np.float64)
    return {
        "episode": episode,
        "environment_steps": environment_steps,
        "updates": updates,
        "elapsed_seconds": elapsed_seconds,
        "agent_ids": agent_ids,
        "evaluation_returns": evaluation_returns,
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
        "mean_metrics_since_previous_evaluation": _mean_metrics(metric_rows),
    }


def _parameters_are_finite(algorithm: BaseMARLAlgorithm) -> bool:
    return all(
        bool(torch.isfinite(parameter).all().item())
        for parameter in algorithm.parameters()
    )


def _seed_result(
    name: str,
    seed: int,
    algorithm: BaseMARLAlgorithm,
    metrics: list[dict[str, float]],
    update_count: int,
    action_kind: ActionKind,
    settings: BenchmarkSettings,
    device: torch.device,
    evaluation_history: list[dict[str, Any]],
    environment_steps: int,
    wall_clock_seconds: float,
) -> dict[str, Any]:
    del action_kind, settings, device
    if not evaluation_history:
        raise RuntimeError("训练没有产生定期评估记录")
    final_evaluation = evaluation_history[-1]
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
        "required_metrics": sorted(required),
        "observed_metrics": sorted(observed),
        "metric_summary": _metric_summary(metrics),
        "agent_ids": agent_ids,
        "evaluation_returns": evaluation_returns,
        "mean_evaluation_return_per_agent": np.mean(
            np.asarray(evaluation_returns), axis=0
        ).tolist(),
        "evaluation_history": evaluation_history,
        "environment_steps": environment_steps,
        "wall_clock_seconds": wall_clock_seconds,
        "updates_per_second": update_count / max(wall_clock_seconds, 1e-12),
        "parameters_finite": _parameters_are_finite(algorithm),
        "output_complete": output_complete,
        "numerically_stable": numerically_stable,
    }


def _train_off_policy(
    name: str,
    seed: int,
    settings: BenchmarkSettings,
    device: torch.device,
) -> dict[str, Any]:
    action_kind = (
        ActionKind.DISCRETE
        if name == "maac"
        else ActionKind.CONTINUOUS
    )
    adapter = build_mpe2_simple_adversary(
        MPE2SimpleAdversaryConfig(horizon=settings.horizon),
        action_kind=action_kind,
    )
    rng = np.random.default_rng(seed)
    try:
        config = _algorithm_config(name, settings)
        experiment = build_experiment(adapter.spec, config, device=device, seed=seed)
        trainer = experiment.trainer
        if not isinstance(trainer, OffPolicyTrainer):
            raise TypeError(f"{name} 应使用 OffPolicyTrainer")
        metrics: list[dict[str, float]] = []
        evaluation_history: list[dict[str, Any]] = []
        metric_start = 0
        environment_steps = 0
        started = time.perf_counter()
        for episode in range(settings.train_episodes):
            for environment_index in range(settings.num_envs):
                episode_seed = seed + episode * settings.num_envs + environment_index
                step = adapter.reset(seed=episode_seed)
                while not step.done:
                    actions = _policy_actions(
                        experiment.algorithm,
                        step.observations,
                        step.action_mask,
                        device=device,
                        deterministic=False,
                    )
                    if name == "maddpg":
                        exploration = 0.3 * max(
                            0.1,
                            1.0 - episode / max(settings.train_episodes - 1, 1),
                        )
                        actions = np.clip(
                            actions + rng.normal(0.0, exploration, size=actions.shape),
                            -1.0,
                            1.0,
                        ).astype(np.float32)
                    following = adapter.step(actions)
                    trainer.record(Transition(step, actions, following))
                    if trainer.ready:
                        metrics.append(trainer.update())
                    environment_steps += 1
                    step = following
            completed_episode = episode + 1
            if (
                completed_episode % settings.eval_interval == 0
                or completed_episode == settings.train_episodes
            ):
                evaluation_history.append(
                    _evaluation_checkpoint(
                        experiment.algorithm,
                        action_kind,
                        settings,
                        device=device,
                        seed=seed,
                        episode=completed_episode,
                        environment_steps=environment_steps,
                        updates=trainer.update_plan.update_count,
                        elapsed_seconds=time.perf_counter() - started,
                        metric_rows=metrics[metric_start:],
                    )
                )
                metric_start = len(metrics)
        return _seed_result(
            name,
            seed,
            experiment.algorithm,
            metrics,
            trainer.update_plan.update_count,
            action_kind,
            settings,
            device,
            evaluation_history,
            environment_steps,
            time.perf_counter() - started,
        )
    finally:
        adapter.close()


def _train_mappo(
    seed: int,
    settings: BenchmarkSettings,
    device: torch.device,
) -> dict[str, Any]:
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
        metrics = []
        evaluation_history: list[dict[str, Any]] = []
        metric_start = 0
        environment_steps = 0
        started = time.perf_counter()
        for episode in range(settings.train_episodes):
            seeds = [
                seed + episode * settings.num_envs + index
                for index in range(settings.num_envs)
            ]
            metrics.append(trainer.train_rollout(seeds))
            environment_steps += settings.num_envs * settings.horizon
            completed_episode = episode + 1
            if (
                completed_episode % settings.eval_interval == 0
                or completed_episode == settings.train_episodes
            ):
                evaluation_history.append(
                    _evaluation_checkpoint(
                        experiment.algorithm,
                        ActionKind.DISCRETE,
                        settings,
                        device=device,
                        seed=seed,
                        episode=completed_episode,
                        environment_steps=environment_steps,
                        updates=trainer.update_plan.update_count,
                        elapsed_seconds=time.perf_counter() - started,
                        metric_rows=metrics[metric_start:],
                    )
                )
                metric_start = len(metrics)
        return _seed_result(
            "mappo",
            seed,
            experiment.algorithm,
            metrics,
            trainer.update_plan.update_count,
            ActionKind.DISCRETE,
            settings,
            device,
            evaluation_history,
            environment_steps,
            time.perf_counter() - started,
        )
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


def run_benchmark(
    algorithms: list[str],
    seeds: list[int],
    settings: BenchmarkSettings,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """运行选定算法；单个算法失败时保留错误并继续其余项目。"""

    results: list[dict[str, Any]] = []
    for name in algorithms:
        if name == "qmix":
            try:
                results.append(_qmix_incompatibility(settings))
            except Exception as error:  # noqa: BLE001 - 基准必须保留其他算法结果。
                results.append(
                    {
                        "algorithm": name,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            continue

        seed_results: list[dict[str, Any]] = []
        error_text: str | None = None
        for seed in seeds:
            try:
                if name == "mappo":
                    seed_results.append(_train_mappo(seed, settings, device))
                else:
                    seed_results.append(
                        _train_off_policy(name, seed, settings, device)
                    )
            except Exception as error:  # noqa: BLE001 - JSON 需要记录具体失败。
                error_text = f"seed={seed} {type(error).__name__}: {error}"
                break
        if error_text is not None:
            results.append(
                {
                    "algorithm": name,
                    "status": "failed",
                    "error": error_text,
                    "seeds": seed_results,
                }
            )
            continue
        results.append(
            {
                "algorithm": name,
                "status": "passed",
                "seeds": seed_results,
                "output_complete": all(
                    result["output_complete"] for result in seed_results
                ),
                "numerically_stable": all(
                    result["numerically_stable"] for result in seed_results
                ),
            }
        )

    return {
        "benchmark": "Farama MPE2 simple_adversary_v3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "versions": {
            package: version(package)
            for package in ("gymnasium", "pettingzoo", "mpe2", "torch")
        },
        "settings": asdict(settings),
        "seeds": seeds,
        "stability_definition": (
            "completed updates with finite metrics, parameters and periodic per-agent "
            "evaluation returns across all seeds; this is not a convergence claim"
        ),
        "results": results,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MPE2 simple_adversary 非合作博弈训练稳定性基准"
    )
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
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results") / "mpe2_simple_adversary.json",
    )
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
        eval_interval=args.eval_interval,
    )
    device = resolve_device(args.device)
    report = run_benchmark(
        list(dict.fromkeys(args.algorithms)),
        list(dict.fromkeys(args.seeds)),
        settings,
        device=device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for result in report["results"]:
        print(
            f"algorithm={result['algorithm']} status={result['status']} "
            f"stable={result.get('numerically_stable')} "
            f"complete={result.get('output_complete')}"
        )
    print(f"report={args.output.resolve()}")
    if any(result["status"] == "failed" for result in report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
