"""Farama MPE2 非合作博弈适配器的形状、动作和终止语义测试。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from gymnasium import spaces

from benchmarks.benchmark_noncooperative import (
    BenchmarkSettings,
    _training_iterations,
    run_benchmark,
    write_summary_csv,
)
from benchmarks.plot_noncooperative import generate_plots
from marl.algorithms import QMIXConfig
from marl.envs import (
    ActionKind,
    MPE2SimpleAdversaryAdapter,
    RewardStructure,
    build_mpe2_simple_adversary_adapter,
    environment_config_from_dict,
    load_environment_config,
    mpe2_simple_adversary_config_from_environment,
)


class FakeParallelAdversary:
    possible_agents = ["adversary_0", "agent_0", "agent_1"]
    agents = possible_agents.copy()
    state_space: spaces.Space[Any] = spaces.Box(
        -np.inf, np.inf, (28,), dtype=np.float32
    )

    def __init__(self, *, continuous: bool) -> None:
        self.continuous = continuous
        self.time = 0
        self.last_actions: dict[str, Any] | None = None
        self._observations = {
            "adversary_0": spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32),
            "agent_0": spaces.Box(-np.inf, np.inf, (10,), dtype=np.float32),
            "agent_1": spaces.Box(-np.inf, np.inf, (10,), dtype=np.float32),
        }
        self._actions = {
            agent: (
                spaces.Box(0.0, 1.0, (5,), dtype=np.float32)
                if continuous
                else spaces.Discrete(5)
            )
            for agent in self.possible_agents
        }

    def observation_space(self, agent: str) -> spaces.Space[Any]:
        return self._observations[agent]

    def action_space(self, agent: str) -> spaces.Space[Any]:
        return self._actions[agent]

    def _values(self) -> dict[str, np.ndarray]:
        return {
            agent: np.full(space.shape, self.time + index, dtype=np.float32)
            for index, (agent, space) in enumerate(self._observations.items())
        }
    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        del seed
        self.time = 0
        self.agents = self.possible_agents.copy()
        return self._values(), {agent: {} for agent in self.possible_agents}

    def step(self, actions: dict[str, Any]):
        self.last_actions = actions
        self.time += 1
        truncated = self.time >= 2
        if truncated:
            self.agents = []
        rewards = {"adversary_0": -1.0, "agent_0": 0.5, "agent_1": 0.5}
        false = {agent: False for agent in self.possible_agents}
        truncations = {
            agent: truncated for agent in self.possible_agents
        }
        infos = {agent: {"time": self.time} for agent in self.possible_agents}
        return self._values(), rewards, false, truncations, infos

    def state(self) -> np.ndarray:
        return np.arange(28, dtype=np.float32) + self.time

    def close(self) -> None:
        return None


def test_discrete_adapter_pads_observations_and_preserves_individual_rewards() -> None:
    environment = FakeParallelAdversary(continuous=False)
    adapter = MPE2SimpleAdversaryAdapter(
        environment, ActionKind.DISCRETE, horizon=2
    )
    assert adapter.spec.observation_dim == 10
    assert adapter.spec.action_dim == 5
    assert adapter.spec.state_dim == 28
    assert adapter.spec.reward_structure == RewardStructure.INDIVIDUAL

    step = adapter.reset(seed=7)
    assert step.observations.shape == (3, 10)
    np.testing.assert_array_equal(step.observations[0, 8:], 0.0)
    assert step.action_mask is not None
    assert step.action_mask.all()

    following = adapter.step(np.asarray([0, 1, 2], dtype=np.int64))
    np.testing.assert_array_equal(following.rewards, [-1.0, 0.5, 0.5])
    assert not following.done
    final = adapter.step(np.asarray([2, 3, 4], dtype=np.int64))
    assert final.truncated and not final.terminated
    with pytest.raises(RuntimeError, match="reset"):
        adapter.step(np.zeros(3, dtype=np.int64))

    with pytest.raises(ValueError, match="共享"):
        QMIXConfig().build(adapter.spec)


def test_continuous_adapter_maps_policy_range_to_mpe2_box() -> None:
    environment = FakeParallelAdversary(continuous=True)
    adapter = MPE2SimpleAdversaryAdapter(
        environment, ActionKind.CONTINUOUS, horizon=2
    )
    step = adapter.reset()
    assert step.action_mask is None
    actions = np.tile(np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0]), (3, 1))
    adapter.step(actions)
    assert environment.last_actions is not None
    expected = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    for value in environment.last_actions.values():
        np.testing.assert_allclose(value, expected)

    with pytest.raises(ValueError, match="连续动作"):
        adapter.step(np.zeros((3, 4), dtype=np.float32))


def test_environment_yaml_builds_real_mpe2_adapter() -> None:
    pytest.importorskip("mpe2")
    path = (
        Path(__file__).parents[1]
        / "examples"
        / "configs"
        / "environments"
        / "mpe2_simple_adversary.yaml"
    )
    selected = load_environment_config(path)
    settings = mpe2_simple_adversary_config_from_environment(selected)
    adapter = build_mpe2_simple_adversary_adapter(selected)
    try:
        step = adapter.reset(seed=11)
        assert settings.num_good_agents == 2
        assert adapter.spec.num_agents == 3
        assert adapter.spec.observation_dim == 10
        assert adapter.spec.state_dim == 28
        assert step.observations.shape == (3, 10)
        assert step.rewards.tolist() == [0.0, 0.0, 0.0]
    finally:
        adapter.close()


def test_mpe2_environment_config_rejects_invalid_options() -> None:
    selected = environment_config_from_dict(
        {
            "schema_version": 1,
            "environment": "mpe2_simple_adversary",
            "action_kind": "discrete",
            "options": {"unknown": 1},
        }
    )
    with pytest.raises(ValueError, match="未知字段"):
        mpe2_simple_adversary_config_from_environment(selected)

    selected = environment_config_from_dict(
        {
            "schema_version": 1,
            "environment": "mpe2_simple_adversary",
            "action_kind": "discrete",
            "options": {"horizon": True},
        }
    )
    with pytest.raises(ValueError, match="horizon"):
        mpe2_simple_adversary_config_from_environment(selected)


def test_all_algorithms_have_complete_noncooperative_smoke_report(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mpe2")
    report = run_benchmark(
        ["maac", "mappo", "maddpg", "masac", "qmix"],
        [3],
        BenchmarkSettings(
            train_episodes=1,
            eval_episodes=1,
            num_envs=1,
            horizon=2,
            replay_batch_size=2,
            hidden_dim=8,
        ),
        device=torch.device("cpu"),
        artifact_dir=tmp_path / "artifacts",
    )
    results = {
        result["algorithm"]: result
        for result in report["results"]
    }
    for algorithm in ("maac", "mappo", "maddpg", "masac"):
        assert results[algorithm]["status"] == "passed"
        assert results[algorithm]["output_complete"] is True
        assert results[algorithm]["numerically_stable"] is True
        seed_result = results[algorithm]["seeds"][0]
        assert seed_result["environment_steps"] == 2
        assert len(seed_result["evaluation_history"]) == 2
        assert seed_result["evaluation_history"][0]["environment_steps"] == 0
        assert seed_result["evaluation_history"][0]["updates"] == 0
        assert seed_result["optimizer_steps"] >= seed_result["updates"]
        assert len(seed_result["checkpoint"]["sha256"]) == 64
        assert Path(seed_result["checkpoint"]["path"]).is_file()
    assert results["qmix"]["status"] == "expected_incompatible"
    assert report["comparison_groups"]["discrete"]["algorithms"] == ["maac", "mappo"]
    assert report["comparison_groups"]["continuous"]["algorithms"] == [
        "maddpg",
        "masac",
    ]
    assert len(report["cross_play"]) == 2
    assert all(
        np.asarray(group["adversary_returns"]).shape == (2, 2)
        for group in report["cross_play"]
    )
    assert all(
        np.asarray(group["good_team_returns"]).shape == (2, 2)
        for group in report["cross_play"]
    )

    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    summary_path = tmp_path / "summary.csv"
    write_summary_csv(report, summary_path)
    assert len(summary_path.read_text(encoding="utf-8").splitlines()) == 5
    plots = generate_plots(report_path, tmp_path / "plots")
    assert {plot.name for plot in plots} == {
        "learning_curves_discrete.png",
        "learning_curves_continuous.png",
        "final_returns_discrete.png",
        "final_returns_continuous.png",
        "return_distributions_discrete.png",
        "return_distributions_continuous.png",
        "loss_stability_discrete.png",
        "loss_stability_continuous.png",
        "gradient_entropy_stability_discrete.png",
        "gradient_entropy_stability_continuous.png",
        "cross_play_discrete.png",
        "cross_play_continuous.png",
        "training_efficiency.png",
    }
    assert all(plot.stat().st_size > 0 for plot in plots)


def test_benchmark_requires_exact_training_and_evaluation_step_boundaries() -> None:
    with pytest.raises(ValueError, match="target_environment_steps"):
        _training_iterations(
            BenchmarkSettings(
                num_envs=2,
                horizon=3,
                eval_interval_steps=12,
                target_environment_steps=13,
            )
        )
    with pytest.raises(ValueError, match="eval_interval_steps"):
        _training_iterations(
            BenchmarkSettings(
                num_envs=2,
                horizon=3,
                eval_interval_steps=10,
                target_environment_steps=12,
            )
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_mappo_metric_summary_keeps_one_device_on_cuda() -> None:
    pytest.importorskip("mpe2")
    report = run_benchmark(
        ["mappo"],
        [5],
        BenchmarkSettings(
            train_episodes=1,
            eval_episodes=1,
            num_envs=1,
            horizon=2,
            replay_batch_size=2,
            hidden_dim=8,
        ),
        device=torch.device("cuda"),
    )
    result = report["results"][0]
    assert result["status"] == "passed"
    assert result["output_complete"] is True
    assert result["numerically_stable"] is True
