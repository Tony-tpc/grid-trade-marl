"""等价提速、Adam 隔离、Windows spawn 和完整轮次恢复回归。"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

import benchmarks.benchmark_noncooperative as benchmark
from benchmarks.run_state import TransitionSchedule
from marl.algorithms import MAACConfig, MASACConfig
from marl.core import MARLBatch
from marl.envs import ActionKind, EnvironmentSpec, RewardStructure
from marl.runtime import TensorReplayBuffer
from marl.training.optimization import OptimizerRuntime


def _spec(discrete: bool) -> EnvironmentSpec:
    return EnvironmentSpec(2, 3, 4, 6,
                           ActionKind.DISCRETE if discrete else ActionKind.CONTINUOUS,
                           8, RewardStructure.INDIVIDUAL)


def _batch(discrete: bool) -> MARLBatch:
    return MARLBatch(
        observations=torch.randn(5, 2, 3), next_observations=torch.randn(5, 2, 3),
        rewards=torch.randn(5, 2), terminated=torch.zeros(5, 2, dtype=torch.bool),
        truncated=torch.zeros(5, 2, dtype=torch.bool),
        actions=torch.zeros(5, 2, dtype=torch.long) if discrete else torch.zeros(5, 2, 4),
    )


def test_maac_other_actor_adam_momentum_is_not_stepped() -> None:
    algorithm = MAACConfig().build(_spec(True))
    optimizer = torch.optim.Adam(algorithm.policy.parameters(), lr=.01)
    batch = _batch(True)
    algorithm.optimize(optimizer, algorithm.compute_actor_loss_bundle(batch, 1), 10.)
    other = algorithm.get_submodule("policy.actors.1")
    before = {name: value.clone() for name, value in other.state_dict().items()}
    counters = [optimizer.state[p]["step"].clone() for p in other.parameters()]
    algorithm.optimize(optimizer, algorithm.compute_actor_loss_bundle(batch, 0), 10.)
    for name, value in other.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    for parameter, counter in zip(other.parameters(), counters, strict=True):
        assert parameter.grad is None
        assert optimizer.state[parameter]["step"] == counter


def test_temperature_matches_reference_rng_gradient_without_any_critic() -> None:
    algorithm = MASACConfig().build(_spec(False))
    batch = _batch(False)
    torch.manual_seed(783)
    _, reference = algorithm._actor_and_temperature_results(batch)
    reference.loss.backward()
    assert algorithm.entropy_objective.log_alpha.grad is not None
    gradient = algorithm.entropy_objective.log_alpha.grad.clone()
    after_rng = torch.get_rng_state()
    algorithm.zero_grad(set_to_none=True)
    torch.manual_seed(783)
    def fail(*args):
        raise AssertionError("temperature 必须不执行 critic")
    handles = [module.register_forward_pre_hook(fail)
               for group in (algorithm.critics.first, algorithm.critics.second)
               for module in group.critics]
    try:
        actual = algorithm.compute_temperature_loss_bundle(batch)
        actual.total.backward()
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(actual.total, reference.loss, rtol=0, atol=0)
    torch.testing.assert_close(algorithm.entropy_objective.log_alpha.grad, gradient, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), after_rng)
    assert all(p.grad is None for p in algorithm.policy.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 pinned/CUDA")
def test_pinned_sample_owns_storage_and_matches_cpu() -> None:
    replay = TensorReplayBuffer(_spec(False), 5, pin_memory=True)
    replay.size = 5
    for value in (replay.observations, replay.next_observations, replay.states,
                  replay.next_states, replay.actions, replay.rewards):
        value.copy_(torch.arange(value.numel()).reshape_as(value))
    replay.terminated.zero_()
    replay.truncated.zero_()
    cpu = replay.sample(3, np.random.default_rng(73))
    assert cpu.observations.is_pinned()
    snapshot = cpu.observations.clone()
    cuda = replay.sample(3, np.random.default_rng(73), device="cuda")
    for _ in range(10):
        replay.sample(3, np.random.default_rng(91), device="cuda")
    torch.testing.assert_close(cpu.observations, snapshot)
    torch.testing.assert_close(cuda.observations.cpu(), snapshot)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 指标归约")
def test_packed_cuda_metrics_match_cpu_sums_peaks_and_nonfinite_flags() -> None:
    cpu: dict[str, torch.Tensor] = {}
    cuda: dict[str, torch.Tensor] = {}
    for index in range(5):
        values = {f"metric_{i}": torch.tensor(float(i - index)) for i in range(12)}
        values.update({"value_abs_max": torch.tensor(float(5 - index)),
                       "value_nonfinite": torch.tensor(float(index == 2)),
                       "actor_step_count": torch.tensor(1.),
                       "actor_clip_count": torch.tensor(float(index % 2))})
        OptimizerRuntime._accumulate(cpu, values)
        OptimizerRuntime._accumulate(cuda, {key: value.cuda() for key, value in values.items()})
    expected = OptimizerRuntime._averages(cpu, 5)
    actual = OptimizerRuntime._averages(cuda, 5)
    for key, value in expected.items():
        torch.testing.assert_close(actual[key].cpu(), value, rtol=0, atol=0)
    assert actual["value_nonfinite"].item() == 1.
    assert actual["value_abs_max"].item() == 5.
    assert actual["actor_gradient_clip_rate"].item() == pytest.approx(.4)


def test_transition_schedule_counts_joint_transitions_and_restores_remainder() -> None:
    schedule = TransitionSchedule(32, 4, 2)
    assert sum(schedule.add(4) for _ in range(7)) == 0
    assert schedule.add(4) == 0
    assert schedule.pending == 1
    restored = TransitionSchedule(**asdict(schedule))
    assert restored.add(3) == 2
    assert restored.add(8) == 4
    compatible = TransitionSchedule(32)
    assert sum(compatible.add(4) for _ in range(25)) == 69


@pytest.mark.parametrize("name", ["maac", "mappo", "maddpg", "masac"])
def test_interrupted_rollout_boundary_matches_uninterrupted(
    name: str, tmp_path: Path,
) -> None:
    pytest.importorskip("mpe2")
    settings = benchmark.BenchmarkSettings(
        train_episodes=4, num_envs=1, horizon=2, eval_episodes=1,
        eval_interval_steps=2, checkpoint_interval_steps=2, replay_batch_size=2,
        hidden_dim=8, layer_norm=True, normalize_targets=True, detailed_evaluation=False,
        train_every_transitions=3, gradient_steps=2,
    )
    def run(directory: Path, resume: bool = False):
        if name == "mappo":
            return benchmark._train_mappo(7, settings, torch.device("cpu"),
                                          directory / "checkpoints", resume=resume)
        return benchmark._train_off_policy(name, 7, settings, torch.device("cpu"),
                                            directory / "checkpoints", resume=resume)
    expected, expected_model = run(tmp_path / "whole")
    original_save = benchmark.save_run
    def interrupt(*args, **kwargs):
        original_save(*args, **kwargs)
        if kwargs["episode"] == 2:
            raise RuntimeError("模拟进程退出")
    with patch.object(benchmark, "save_run", interrupt):
        with pytest.raises(RuntimeError, match="模拟进程退出"):
            run(tmp_path / "interrupted")
    actual, actual_model = run(tmp_path / "interrupted", resume=True)
    assert actual["updates"] == expected["updates"]
    assert actual["optimizer_steps"] == expected["optimizer_steps"]
    assert actual["evaluation_returns"] == expected["evaluation_returns"]
    for key, value in expected_model.state_dict().items():
        torch.testing.assert_close(actual_model.state_dict()[key], value, rtol=0, atol=0)
    old = torch.load(expected["checkpoint"]["path"], weights_only=False)
    new = torch.load(actual["checkpoint"]["path"], weights_only=False)
    for key in ("torch_rng",):
        assert torch.equal(old[key], new[key])
    for optimizer, state in old["optimization"]["optimizers"].items():
        for key, value in state["state"].items():
            for field, tensor in value.items():
                torch.testing.assert_close(
                    new["optimization"]["optimizers"][optimizer]["state"][key][field],
                    tensor, rtol=0, atol=0,
                )


def test_spawn_workers_are_seed_equivalent_and_resume_completed(tmp_path: Path) -> None:
    pytest.importorskip("mpe2")
    settings = benchmark.BenchmarkSettings(
        train_episodes=1, num_envs=1, horizon=2, eval_episodes=1,
        replay_batch_size=2, hidden_dim=8, detailed_evaluation=False,
    )
    parallel = benchmark.run_benchmark(["maac"], [7, 17], settings, device=torch.device("cpu"),
                                       artifact_dir=tmp_path / "parallel", workers=2)
    sequential = benchmark.run_benchmark(["maac"], [7, 17], settings, device=torch.device("cpu"),
                                         artifact_dir=tmp_path / "sequential")
    assert parallel["results"][0]["status"] == "passed"
    for left, right in zip(parallel["results"][0]["seeds"],
                           sequential["results"][0]["seeds"], strict=True):
        assert left["evaluation_returns"] == right["evaluation_returns"]
        a = torch.load(left["checkpoint"]["path"], weights_only=False)["algorithm"]
        b = torch.load(right["checkpoint"]["path"], weights_only=False)["algorithm"]
        for name, value in a.items():
            torch.testing.assert_close(value, b[name], rtol=0, atol=0)
    resumed = benchmark.run_benchmark(["maac"], [7, 17], settings, device=torch.device("cpu"),
                                      artifact_dir=tmp_path / "parallel", workers=2, resume=True)
    assert resumed["results"] == parallel["results"]
    failed = benchmark.run_benchmark(["maac"], [7], replace(settings, hidden_dim=12),
                                     device=torch.device("cpu"), artifact_dir=tmp_path / "parallel",
                                     resume=True)
    assert failed["results"][0]["status"] == "failed"
    assert (tmp_path / "parallel" / "failures" / "maac_seed_7.json").is_file()
