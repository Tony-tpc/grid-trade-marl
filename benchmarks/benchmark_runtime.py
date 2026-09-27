"""比较批量运行路径与正确性修复后的逐环境/逐 minibatch 基线。"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import subprocess
import time
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks.benchmark_noncooperative import BenchmarkSettings, _train_mappo, _train_off_policy
from marl.algorithms import MAAC, MAPPO, MAACConfig, MAPPOConfig
from marl.core import MARLBatch
from marl.envs import ActionKind, MPE2SimpleAdversaryConfig, Transition, build_mpe2_simple_adversary
from marl.modules.critic import AttentionQConfig, CentralizedValueConfig
from marl.modules.policy import IndependentDiscreteConfig
from marl.runtime import SyncVectorEnv, TensorReplayBuffer, resolve_device
from marl.training import OffPolicyCollector, OptimizerRuntime, PPOUpdateConfig, PreparedRollout


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(device: torch.device, function: Any) -> tuple[float, int | None]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = time.perf_counter()
    function()
    _synchronize(device)
    elapsed = time.perf_counter() - started
    peak = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None
    )
    return elapsed, peak


def _action(
    algorithm: MAAC,
    observations: np.ndarray,
    action_mask: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    with torch.inference_mode():
        return (
            algorithm.act(
                torch.as_tensor(observations, dtype=torch.float32, device=device).unsqueeze(0),
                deterministic=True,
                action_mask=torch.as_tensor(
                    action_mask, dtype=torch.bool, device=device
                ).unsqueeze(0),
            )
            .squeeze(0)
            .cpu()
            .numpy()
        )


def _off_policy_benchmark(
    device: torch.device, *, iterations: int, num_envs: int, horizon: int
) -> dict[str, Any]:
    config = replace(
        MAACConfig(),
        policy=IndependentDiscreteConfig(hidden_dim=32),
        critic=AttentionQConfig(hidden_dim=32, attention_heads=4),
    )
    probe = build_mpe2_simple_adversary(
        MPE2SimpleAdversaryConfig(horizon=horizon), action_kind=ActionKind.DISCRETE
    )
    algorithm = config.build(probe.spec).to(device)
    probe.close()
    capacity = iterations * num_envs * horizon

    def make_adapters() -> list[Any]:
        return [
            build_mpe2_simple_adversary(
                MPE2SimpleAdversaryConfig(horizon=horizon),
                action_kind=ActionKind.DISCRETE,
            )
            for _ in range(num_envs)
        ]

    sequential_adapters = make_adapters()
    sequential_replay = TensorReplayBuffer(algorithm.spec, capacity)

    def sequential() -> None:
        for iteration in range(iterations):
            for index, adapter in enumerate(sequential_adapters):
                step = adapter.reset(seed=iteration * num_envs + index)
                while not step.done:
                    assert step.action_mask is not None
                    actions = _action(
                        algorithm, step.observations, step.action_mask, device
                    )
                    following = adapter.step(actions)
                    sequential_replay.add(Transition(step, actions, following))
                    step = following

    batched_adapters = make_adapters()
    vector = SyncVectorEnv(batched_adapters)
    collector = OffPolicyCollector(vector, algorithm, device=device)
    batched_replay = TensorReplayBuffer(algorithm.spec, capacity)

    def batched() -> None:
        for iteration in range(iterations):
            seeds = [iteration * num_envs + index for index in range(num_envs)]
            for transitions in collector.rollout(seeds, deterministic=True):
                batched_replay.record_many(transitions)

    sequential_seconds, sequential_peak = _measure(device, sequential)
    batched_seconds, batched_peak = _measure(device, batched)
    for adapter in (*sequential_adapters, *batched_adapters):
        adapter.close()
    steps = capacity
    return {
        "environment_steps": steps,
        "sequential_steps_per_second": steps / sequential_seconds,
        "batched_steps_per_second": steps / batched_seconds,
        "speedup": sequential_seconds / batched_seconds,
        "sequential_peak_bytes": sequential_peak,
        "batched_peak_bytes": batched_peak,
        "peak_memory_ratio": (
            batched_peak / sequential_peak
            if batched_peak is not None and sequential_peak
            else None
        ),
        "replay_capacity_and_count_match": (
            sequential_replay.size == batched_replay.size
            and sequential_replay.position == batched_replay.position
        ),
    }


class _LegacyPreparedRollout(PreparedRollout):
    def __init__(self, batch: MARLBatch, device: torch.device) -> None:
        super().__init__(batch)
        self.device = device

    def minibatches(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        for mini_batch in super().minibatches(**kwargs):
            yield mini_batch.to(self.device, non_blocking=self.device.type == "cuda")


def _mappo_runtime(
    algorithm: MAPPO,
    config: MAPPOConfig,
    device: torch.device,
    *,
    generator_device: torch.device,
) -> OptimizerRuntime:
    return OptimizerRuntime(
        {
            "actor": torch.optim.Adam(
                algorithm.policy.parameters(),
                lr=config.update.resolved_actor_learning_rate,
            ),
            "critic": torch.optim.Adam(
                algorithm.critic.parameters(),
                lr=config.update.resolved_critic_learning_rate,
            ),
        },
        {
            "actor": config.update.resolved_actor_max_grad_norm,
            "critic": config.update.resolved_critic_max_grad_norm,
        },
        generator=torch.Generator(device=generator_device).manual_seed(11),
    )


def _mappo_benchmark(
    device: torch.device, *, repeats: int, num_envs: int, horizon: int
) -> dict[str, Any]:
    adapter = build_mpe2_simple_adversary(
        MPE2SimpleAdversaryConfig(horizon=horizon), action_kind=ActionKind.DISCRETE
    )
    size = num_envs * horizon
    config = replace(
        MAPPOConfig(),
        policy=IndependentDiscreteConfig(hidden_dim=32),
        critic=CentralizedValueConfig(hidden_dim=32),
        update=PPOUpdateConfig(epochs=2, mini_batch_size=size),
    )
    torch.manual_seed(7)
    reference = config.build(adapter.spec).to(device)
    legacy = deepcopy(reference)
    optimized = deepcopy(reference)
    adapter.close()
    batch = MARLBatch(
        observations=torch.randn(size, reference.spec.num_agents, reference.spec.observation_dim),
        actions=torch.randint(0, reference.spec.action_dim, (size, reference.spec.num_agents)),
        state=torch.randn(size, reference.spec.state_dim),
        action_mask=torch.ones(
            size, reference.spec.num_agents, reference.spec.action_dim, dtype=torch.bool
        ),
        extras={
            "old_log_prob": torch.randn(size, reference.spec.num_agents),
            "old_values": torch.randn(size, reference.spec.num_agents),
            "advantages": torch.randn(size, reference.spec.num_agents),
            "returns": torch.randn(size, reference.spec.num_agents),
        },
    )
    if device.type == "cuda":
        batch = batch.pin_memory()
    legacy_runtime = _mappo_runtime(
        legacy, config, device, generator_device=torch.device("cpu")
    )
    optimized_runtime = _mappo_runtime(
        optimized, config, device, generator_device=device
    )

    def legacy_updates() -> None:
        for _ in range(repeats):
            legacy.update(_LegacyPreparedRollout(batch, device), legacy_runtime)

    def optimized_updates() -> None:
        for _ in range(repeats):
            prepared = PreparedRollout(
                batch.to(device, non_blocking=device.type == "cuda")
            )
            optimized.update(prepared, optimized_runtime)

    legacy_seconds, legacy_peak = _measure(device, legacy_updates)
    optimized_seconds, optimized_peak = _measure(device, optimized_updates)
    samples = repeats * size * config.update.epochs
    max_parameter_difference = max(
        float((left - right).abs().max().item())
        for left, right in zip(
            legacy.state_dict().values(),
            optimized.state_dict().values(),
            strict=True,
        )
    )
    return {
        "training_samples": samples,
        "legacy_samples_per_second": samples / legacy_seconds,
        "optimized_samples_per_second": samples / optimized_seconds,
        "throughput_ratio": legacy_seconds / optimized_seconds,
        "legacy_peak_bytes": legacy_peak,
        "optimized_peak_bytes": optimized_peak,
        "peak_memory_ratio": (
            optimized_peak / legacy_peak
            if optimized_peak is not None and legacy_peak
            else None
        ),
        "max_parameter_difference": max_parameter_difference,
        "numerically_equivalent": max_parameter_difference <= 1e-5,
    }


def benchmark_full_training(
    device: torch.device, threads: int, steps: int, repeats: int, output: Path,
    reference: Path | None = None,
) -> dict[str, Any]:
    """真实环境+采集+replay+更新+诊断+评估；不可用 collector-only 代替。"""
    if repeats < 5:
        raise ValueError("端到端验收至少重复 5 次")
    torch.set_num_threads(threads)
    settings = BenchmarkSettings(target_environment_steps=steps, eval_interval_steps=steps,
                                 eval_episodes=1, layer_norm=True, normalize_targets=True,
                                 detailed_evaluation=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(reference.read_text(encoding="utf-8")) if reference else None
    results = {}
    for name in ("maac", "mappo", "maddpg", "masac"):
        times, peaks = [], []
        counts = []
        state_path = output.with_name(f"{output.stem}_{name}_weights.pt")
        # 完整预热一次，不把首次初始化算作 CUDA/CPU 稳态速度。
        for repetition in range(-1, repeats):
            captured: list[Any] = []

            def run(
                name: str = name, repetition: int = repetition, captured: list[Any] = captured,
            ) -> None:
                arguments = (7 + max(repetition, 0), settings, device, None)
                captured.append(_train_mappo(*arguments) if name == "mappo"
                                else _train_off_policy(name, *arguments))

            elapsed, peak = _measure(device, run)
            result, algorithm = captured.pop()
            if repetition >= 0:
                times.append(elapsed)
                peaks.append(peak)
                counts.append(result["optimizer_steps"])
            if repetition == repeats - 1:
                torch.save({key: value.detach().cpu() for key, value
                            in algorithm.state_dict().items()}, state_path)
            del algorithm
        row: dict[str, Any] = {
            "seconds": times, "median_seconds": float(np.median(times)),
            "min_seconds": min(times), "max_seconds": max(times),
            "environment_steps_per_run": steps, "optimizer_steps": counts,
            "environment_steps_per_second": steps / float(np.median(times)),
            "optimizer_steps_per_second": float(np.median(counts)) / float(np.median(times)),
            "peak_bytes": max((p for p in peaks if p is not None), default=None),
            "final_weights": str(state_path.resolve()),
        }
        if previous is not None:
            before = previous["results"][name]
            if before["optimizer_steps"] != counts or any(
                asdict(settings).get(key) != value for key, value in previous["settings"].items()
            ):
                raise ValueError("性能对照的配置或 optimizer-step 数不同")
            row["speedup"] = before["median_seconds"] / row["median_seconds"]
            row["peak_memory_ratio"] = (row["peak_bytes"] / before["peak_bytes"]
                                        if before["peak_bytes"] else None)
            old = torch.load(before["final_weights"], weights_only=True)
            new = torch.load(state_path, weights_only=True)
            row["numerically_equivalent"] = old.keys() == new.keys() and all(
                torch.allclose(old[key], new[key], atol=1e-5, rtol=1e-4) for key in old
            )
        results[name] = row
        print(f"{name}: {row['environment_steps_per_second']:.2f} env steps/s", flush=True)
    report = {"device": str(device), "torch_threads": threads,
              "source_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], text=True).strip(),
              "settings": asdict(settings), "repeats": repeats, "results": results}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="MARL 批量运行路径性能回归")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--mappo-repeats", type=int, default=5)
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument("--full-training", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--profile-algorithm", choices=("maac", "mappo", "maddpg", "masac"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/runtime_benchmark.json"),
    )
    args = parser.parse_args()
    device = resolve_device(args.device)
    if args.profile_algorithm:
        torch.set_num_threads(args.threads)
        settings = BenchmarkSettings(
            target_environment_steps=args.steps, eval_interval_steps=args.steps,
            layer_norm=True, normalize_targets=True, detailed_evaluation=False, eval_episodes=1,
        )
        profiler = cProfile.Profile()
        profiler.enable()
        if args.profile_algorithm == "mappo":
            _train_mappo(7, settings, device, None)
        else:
            _train_off_policy(args.profile_algorithm, 7, settings, device, None)
        profiler.disable()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(args.output.with_suffix(".pstats")))
        statistics = pstats.Stats(profiler)
        # 累计 CPU 墙钟时间相互包含；CUDA 为主机调度时间，不冒充 kernel 用时。
        stages = {"environment": ("reset", "step"),
                  "policy_forward": ("act", "sample", "evaluate", "logits_for_agent"),
                  "replay": ("record_many",),
                  "critic_forward": ("compute_critic_loss_bundle", "compute_value_loss_bundle"),
                  "actor_forward": ("compute_actor_loss_bundle", "compute_policy_loss_bundle"),
                  "temperature_forward": ("compute_temperature_loss_bundle",),
                  "backward_clip_optimizer": ("optimize",),
                  "evaluation": ("_evaluation_checkpoint", "_evaluate"),
                  "checkpoint": ("save_run", "_checkpoint_metadata")}
        raw = [
            {"file": key[0], "line": key[1], "function": key[2],
             "calls": value[1], "self_seconds": value[2], "cumulative_seconds": value[3]}
            for key, value in statistics.stats.items()  # type: ignore[attr-defined]
            if any(key[2] in names for names in stages.values())
        ]
        args.output.write_text(json.dumps({
            "algorithm": args.profile_algorithm, "device": str(device),
            "stage_names": stages, "functions": raw,
            "note": "CPU 调度墙钟时间；嵌套函数不能相加，CUDA kernel 需另用 torch.profiler",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if args.full_training:
        benchmark_full_training(device, args.threads, args.steps, args.repeats,
                                args.output, args.reference)
        return
    report: dict[str, Any] = {
        "device": str(device),
        "off_policy": _off_policy_benchmark(
            device,
            iterations=args.iterations,
            num_envs=args.num_envs,
            horizon=args.horizon,
        ),
        "mappo": _mappo_benchmark(
            device,
            repeats=args.mappo_repeats,
            num_envs=args.num_envs,
            horizon=args.horizon,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.enforce:
        off_policy = report["off_policy"]
        mappo = report["mappo"]
        if off_policy["speedup"] < 1.20:
            raise SystemExit("off-policy 批量 collector 吞吐提升不足 20%")
        if mappo["throughput_ratio"] < 0.95:
            raise SystemExit("MAPPO 吞吐下降超过 5%")
        if not mappo["numerically_equivalent"]:
            raise SystemExit("MAPPO 整批迁移与逐 minibatch 迁移不满足数值容差")
        for result in (off_policy, mappo):
            ratio = result["peak_memory_ratio"]
            if ratio is not None and ratio > 1.25:
                raise SystemExit("峰值显存增长超过 25%")


if __name__ == "__main__":
    main()
