"""MAPPO backbone 对照：固定预算、完整状态推理、逐 seed 产物与显式验收。"""

from __future__ import annotations

import argparse
import cProfile
import csv
import hashlib
import json
import pstats
import time
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from benchmarks.benchmark_noncooperative import (
    REQUIRED_METRICS,
    BenchmarkSettings,
    _algorithm_config,
    _evaluate,
    _evaluate_cross_play,
    _evaluation_mode,
    _policy_actions,
    _train_mappo,
)
from benchmarks.run_state import atomic_json, source_identity
from marl.algorithms import MAPPO, MAPPOConfig
from marl.config import config_to_dict
from marl.core.recurrent import RecurrentState
from marl.envs import ActionKind, MemoryCueAdapter
from marl.experiment import build_experiment
from marl.runtime import SyncVectorEnv, resolve_device

KINDS = ("mlp", "gru", "lstm")


def evaluate_memory(algorithm: MAPPO, device: torch.device) -> list[list[float]]:
    """全部四种联合提示各评估一次；不重复确定性样本冒充更多独立证据。"""
    returns = []
    with _evaluation_mode(algorithm):
        for seed in range(4):
            env = MemoryCueAdapter()
            step = env.reset(seed)
            state: RecurrentState = None
            total = np.zeros(2)
            while not step.done:
                actions, state = _policy_actions(
                    algorithm,
                    step.observations,
                    step.action_mask,
                    device=device,
                    deterministic=True,
                    policy_state=state,
                )
                step = env.step(actions)
                total += step.rewards
            returns.append(total.tolist())
    return returns


def _memory_run(kind: str, seed: int, steps: int, device: torch.device, directory: Path):
    settings = BenchmarkSettings(
        horizon=10,
        target_environment_steps=steps,
        actor_backbone=kind,
        sequence_length=10 if kind != "mlp" else None,
        eval_episodes=4,
        eval_interval_steps=200,
        detailed_evaluation=False,
    )
    config = cast(MAPPOConfig, _algorithm_config("mappo", settings))
    experiment = build_experiment(
        SyncVectorEnv([MemoryCueAdapter() for _ in range(4)]), config, seed=seed, device=device
    )
    evaluations = [
        {
            "environment_steps": 0,
            "evaluation_returns": evaluate_memory(experiment.algorithm, device),
        }
    ]
    history = []
    started = time.perf_counter()
    for index in range(steps // 40):
        metrics = experiment.trainer.train_rollout([seed + index * 4 + i for i in range(4)])
        metrics["entropy_ratio"] = metrics["entropy"] / np.log(2)
        completed_steps = (index + 1) * 40
        point = {"environment_steps": completed_steps, "metrics": metrics}
        history.append(point)
        if completed_steps % 200 == 0 or completed_steps == steps:
            evaluations.append(
                {
                    "environment_steps": point["environment_steps"],
                    "evaluation_returns": evaluate_memory(experiment.algorithm, device),
                }
            )
        # 每轮真实诊断落盘，异常不丢失已完成历史。
        with (directory / "diagnostics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(point, allow_nan=False) + "\n")
    checkpoint = directory / "checkpoint.pt"
    experiment.trainer.save_checkpoint(checkpoint)
    result = {
        "seed": seed,
        "environment_steps": steps,
        "training_history": history,
        "evaluation_history": evaluations,
        "evaluation_returns": evaluations[-1]["evaluation_returns"],
        "heldout_returns": evaluations[-1]["evaluation_returns"],
        "optimizer_steps": experiment.trainer.optimization.optimizer_step_count,
        "updates": experiment.trainer.optimization.update_count,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
        "config": config_to_dict(config),
        "settings": asdict(settings),
        "wall_clock_seconds": time.perf_counter() - started,
    }
    return result, experiment.algorithm


def _profile_summary(profile: cProfile.Profile) -> dict[str, float]:
    """主机累计调用时间，不是 CUDA kernel 时间；嵌套项不可直接相加。"""
    names = {
        "collect": "collection",
        "update": "update",
        "to": "transfer",
        "_evaluate": "evaluation",
        "evaluate_memory": "evaluation",
    }
    summary: dict[str, float] = {}
    for (filename, _, name), entry in pstats.Stats(profile).stats.items():  # type: ignore[attr-defined]
        if name in names and ("marl" in filename.lower() or "benchmark" in filename.lower()):
            stage = names[name]
            summary[stage] = summary.get(stage, 0.0) + entry[3]
    return summary


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    """对照启动清单核验，不以报告里已有的少数运行冒充完整矩阵。"""
    expected = {
        (task, kind, seed) for task in report["tasks"] for kind in KINDS for seed in report["seeds"]
    }
    actual = [(r["task"], r["variant_id"], r["seed"]) for r in report["runs"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("实验任务矩阵缺失或重复")
    memory_pass: bool | None = True if "memory" in report["tasks"] else None
    for row in report["runs"]:
        budget = report["budgets"][row["task"]]
        quantum = 40 if row["task"] == "memory" else 100
        interval = 200 if row["task"] == "memory" else 100
        expected_points = list(range(0, budget + 1, interval))
        if expected_points[-1] != budget:
            expected_points.append(budget)
        if row["environment_steps"] != budget or (
            [point["environment_steps"] for point in row["evaluation_history"]] != expected_points
        ):
            raise ValueError("评估步数/预算不完整")
        if [p["environment_steps"] for p in row["training_history"]] != list(
            range(quantum, budget + 1, quantum)
        ):
            raise ValueError("逐 rollout 训练历史缺失")
        evaluation_count = 4 if row["task"] == "memory" else 8
        agents = 2 if row["task"] == "memory" else 3
        for point in row["evaluation_history"]:
            array = np.asarray(point["evaluation_returns"])
            if array.shape != (evaluation_count, agents) or not np.isfinite(array).all():
                raise ValueError("评估矩阵形状/数值错误")
        required = REQUIRED_METRICS["mappo"] | {
            "actor_gradient_clip_rate",
            "critic_gradient_clip_rate",
            "rollout_explained_variance",
        }
        if row["variant_id"] != "mlp":
            required |= {"padding_fraction", "policy_hidden_norm"}
        if row["variant_id"] == "lstm":
            required.add("policy_cell_norm")
        for point in row["training_history"]:
            metrics = point["metrics"]
            if not required <= metrics.keys():
                raise ValueError(f"缺少训练诊断: {sorted(required - metrics.keys())}")
            if not all(np.isfinite(v) for v in metrics.values()) or any(
                value > 0 for name, value in metrics.items() if name.endswith("_nonfinite")
            ):
                raise ValueError("训练诊断出现 NaN/Inf 或非有限值计数")
        heldout_count = 4 if row["task"] == "memory" else 64
        heldout = np.asarray(row["heldout_returns"])
        if heldout.shape != (heldout_count, agents) or not np.isfinite(heldout).all():
            raise ValueError("独立终局评估不完整")
        path = Path(row["checkpoint"]["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["checkpoint"]["sha256"]:
            raise ValueError("checkpoint 校验值错误")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["schema_version"] != 5 or payload["config"] != row["config"]:
            raise ValueError("checkpoint schema/config 不匹配")
        if payload["optimization"]["optimizer_step_count"] != row["optimizer_steps"]:
            raise ValueError("optimizer steps 不匹配")
        if not all(torch.isfinite(t).all() for t in payload["algorithm"].values()):
            raise ValueError("checkpoint 存在非有限参数")
        # 一次 update 有 2 个 epoch，每个 minibatch actor/critic 各 step 一次。
        batches = 2 if row["task"] == "mpe" else 1
        expected_optimizer_steps = budget // quantum * batches * 2 * 2
        if row["optimizer_steps"] != expected_optimizer_steps:
            raise ValueError("实际 optimizer steps 与预算推导不一致")
        if row["task"] == "memory":
            success = float((np.asarray(row["evaluation_returns"]) + 1).mean() / 2)
            row["success_rate"] = success
            passed = abs(success - 0.5) < 1e-8 if row["variant_id"] == "mlp" else success >= 0.8
            row["memory_acceptance_passed"] = passed
            memory_pass = memory_pass and passed
        row["output_complete"] = True
        row["numerically_stable"] = True
    if "mpe" in report["tasks"]:
        cross = report["cross_play"]
        size = len(report["seeds"]) * 3
        if len(cross["labels"]) != size or any(
            np.asarray(cross[key]).shape != (size, size)
            or not np.isfinite(np.asarray(cross[key])).all()
            for key in ("adversary", "good_team")
        ):
            raise ValueError("cross-play 矩阵不完整")
    return {
        "outputs_complete": True,
        "numerically_stable": True,
        "memory_acceptance_passed": memory_pass,
        "convergence": "not_established",
    }


def write_csv(report: dict[str, Any], output: Path) -> None:
    fields = (
        "task",
        "variant_id",
        "seed",
        "environment_steps",
        "optimizer_steps",
        "parameters",
        "wall_clock_seconds",
        "success_rate",
        "checkpoint_sha256",
    )
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["runs"]:
            writer.writerow(
                {
                    **{key: row.get(key, "") for key in fields[:-1]},
                    "checkpoint_sha256": row["checkpoint"]["sha256"],
                }
            )


def run(
    directory: Path,
    *,
    seeds: tuple[int, ...] = (7, 17, 29),
    tasks: tuple[str, ...] = ("memory", "mpe"),
    device: str = "cpu",
    memory_steps: int = 20_000,
    mpe_steps: int = 10_000,
) -> dict[str, Any]:
    if memory_steps < 40 or memory_steps % 40 or mpe_steps < 100 or mpe_steps % 100:
        raise ValueError("memory/mpe 预算必须分别是 40/100 的正整数倍")
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("输出目录非空；请选择新目录，禁止覆盖或混合已有实验")
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    resolved = resolve_device(device)
    report: dict[str, Any] = {
        "schema_version": 1,
        "algorithm": "mappo",
        "tasks": tasks,
        "seeds": seeds,
        "budgets": {"memory": memory_steps, "mpe": mpe_steps},
        "source": source_identity(),
        "device": str(resolved),
        "versions": {name: version(name) for name in ("torch", "numpy", "mpe2")},
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "torch_threads": torch.get_num_threads(),
        "runs": [],
        "profile_note": "CPU 累计调用时间，嵌套项不能相加；不代表 CUDA kernel 时间",
    }
    policies = []
    for task in tasks:
        for kind in KINDS:
            for seed in seeds:
                target = directory / task / kind / str(seed)
                target.mkdir(parents=True)
                profile = cProfile.Profile()
                if resolved.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(resolved)
                started = time.perf_counter()
                profile.enable()
                if task == "memory":
                    row, algorithm = _memory_run(kind, seed, memory_steps, resolved, target)
                else:
                    settings = BenchmarkSettings(
                        target_environment_steps=mpe_steps,
                        eval_interval_steps=100,
                        metrics_interval_steps=100,
                        eval_episodes=8,
                        actor_backbone=kind,
                        sequence_length=16 if kind != "mlp" else None,
                        detailed_evaluation=False,
                    )
                    row, base_algorithm = _train_mappo(
                        seed, settings, resolved, target / "checkpoints"
                    )
                    if not isinstance(base_algorithm, MAPPO):
                        raise TypeError("backbone 对照只允许 MAPPO")
                    algorithm = base_algorithm
                    _, heldout = _evaluate(
                        algorithm,
                        ActionKind.DISCRETE,
                        replace(settings, eval_episodes=64),
                        device=resolved,
                        seed=2_000_000,
                    )
                    row["heldout_returns"] = heldout
                    row["settings"] = asdict(settings)
                    row["config"] = config_to_dict(algorithm.config)
                    policies.append((f"{kind}/{seed}", algorithm))
                profile.disable()
                row.update(
                    algorithm="mappo",
                    task=task,
                    variant_id=kind,
                    parameters=sum(p.numel() for p in algorithm.parameters()),
                    total_seconds=time.perf_counter() - started,
                    profile=_profile_summary(profile),
                    peak_cuda_bytes=(
                        torch.cuda.max_memory_allocated(resolved)
                        if resolved.type == "cuda"
                        else None
                    ),
                )
                row["source"] = report["source"]
                report["runs"].append(row)
                atomic_json(target / "result.json", row)
                atomic_json(directory / "progress.json", report)
                print(
                    f"{task}/{kind}/{seed}: {row['environment_steps']} steps, "
                    f"{row['total_seconds']:.1f}s",
                    flush=True,
                )
    if policies:
        size = len(policies)
        adversary, good = np.zeros((size, size)), np.zeros((size, size))
        settings = BenchmarkSettings(eval_episodes=8, detailed_evaluation=False)
        for i, (_, left) in enumerate(policies):
            for j, (_, right) in enumerate(policies):
                adversary[i, j], good[i, j] = _evaluate_cross_play(
                    left,
                    right,
                    ActionKind.DISCRETE,
                    settings,
                    device=resolved,
                    seed=3_000_000,
                )
        report["cross_play"] = {
            "labels": [label for label, _ in policies],
            "adversary": adversary.tolist(),
            "good_team": good.tolist(),
        }
    report["acceptance"] = validate_report(report)
    atomic_json(directory / "report.json", report)
    write_csv(report, directory / "summary.csv")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    parser.add_argument("--tasks", nargs="+", choices=("memory", "mpe"), default=["memory", "mpe"])
    parser.add_argument("--memory-steps", type=int, default=20_000)
    parser.add_argument("--mpe-steps", type=int, default=10_000)
    args = parser.parse_args()
    report = run(
        args.output,
        seeds=tuple(args.seeds),
        tasks=tuple(args.tasks),
        device=args.device,
        memory_steps=args.memory_steps,
        mpe_steps=args.mpe_steps,
    )
    print(json.dumps(report["acceptance"], indent=2))
    if report["acceptance"]["memory_acceptance_passed"] is False:
        raise SystemExit("固定预算的记忆能力验收未通过；保留全部结果，不扩大预算")


if __name__ == "__main__":
    main()
