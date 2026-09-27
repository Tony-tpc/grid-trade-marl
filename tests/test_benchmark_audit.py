"""审计采样频率与收敛趋势：只用真实采样，不改变训练过程。"""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

from benchmarks.benchmark_noncooperative import (
    BenchmarkSettings,
    _train_mappo,
    _train_off_policy,
    _training_iterations,
    _training_metric_mean,
)
from benchmarks.plot_noncooperative import _metric_series, _rolling_diagnostics


@pytest.mark.parametrize("budget,interval,diagnostics", [(10_000, 100, 100), (100_000, 1000, 1000)])
def test_default_audit_density(budget: int, interval: int, diagnostics: int) -> None:
    settings = BenchmarkSettings(target_environment_steps=budget)
    assert settings.resolved_eval_interval_steps == interval
    assert settings.resolved_metrics_interval_steps == 100
    assert budget // settings.resolved_eval_interval_steps + 1 == 101
    assert budget // settings.resolved_metrics_interval_steps == diagnostics


def test_metric_intervals_validate_rollout_boundaries() -> None:
    with pytest.raises(ValueError, match="metrics_interval_steps"):
        _training_iterations(BenchmarkSettings(metrics_interval_steps=99))
    with pytest.raises(ValueError, match="metrics_interval_steps"):
        BenchmarkSettings(metrics_interval_steps=0)
    with pytest.raises(ValueError, match="eval_interval_steps"):
        BenchmarkSettings(eval_interval_steps=0)


def test_metric_mean_weights_warmup_and_partial_intervals() -> None:
    history = [
        {"metrics": {}, "metric_weight": 0},
        {"metrics": {"loss": 10.0}, "metric_weight": 1},
        {"metrics": {"loss": 2.0}, "metric_weight": 3},
    ]
    assert _training_metric_mean(history) == {"loss": 4.0}
    assert _training_metric_mean([]) == {}


def test_rolling_trends_use_actual_steps_and_only_complete_windows() -> None:
    steps = np.asarray([0, 100, 300, 600, 1000, 1300], dtype=float)
    values = np.stack((2 * steps / 1000 + 1, -steps / 1000 + 3))
    x, slopes, deviations = _rolling_diagnostics(steps, values)
    np.testing.assert_array_equal(x, [1000, 1300])
    np.testing.assert_allclose(slopes, [[2, 2], [-1, -1]])
    np.testing.assert_allclose(deviations[:, 0], values[:, :5].std(axis=1))
    x, slopes, _ = _rolling_diagnostics(steps[:4], values[:, :4])
    assert not len(x)
    assert slopes.shape == (2, 0)
    with pytest.raises(ValueError, match="严格递增"):
        _rolling_diagnostics(np.asarray([0., 0.]), np.ones((1, 2)))


def test_plotting_prefers_dense_history_without_fabricating_legacy_points() -> None:
    seed: dict[str, Any] = {"evaluation_history": [
        {"environment_steps": 0, "mean_metrics_since_previous_evaluation": {}},
        {"environment_steps": 1000, "mean_metrics_since_previous_evaluation": {"loss": 9.}},
    ]}
    steps, values = _metric_series(seed, "loss")
    np.testing.assert_array_equal(steps, [1000])
    np.testing.assert_array_equal(values, [9.])
    seed["training_history"] = [
        {"environment_steps": 100, "metrics": {"loss": 3.}},
        {"environment_steps": 200, "metrics": {"loss": 5.}},
    ]
    steps, values = _metric_series(seed, "loss")
    np.testing.assert_array_equal(steps, [100, 200])
    np.testing.assert_array_equal(values, [3., 5.])
    assert len(_metric_series(seed, "missing")[0]) == 0


@pytest.mark.parametrize("name", ["maac", "mappo"])
def test_dense_audit_does_not_change_training_updates_or_parameters(name: str) -> None:
    pytest.importorskip("mpe2")
    dense = BenchmarkSettings(
        train_episodes=5, eval_episodes=1, num_envs=1, horizon=2,
        replay_batch_size=2, hidden_dim=8, eval_interval_steps=6,
        metrics_interval_steps=2,
    )
    sparse = replace(dense, metrics_interval_steps=6)

    def run(settings: BenchmarkSettings):
        if name == "mappo":
            return _train_mappo(7, settings, torch.device("cpu"), None)
        return _train_off_policy(name, 7, settings, torch.device("cpu"), None)

    left, left_model = run(dense)
    right, right_model = run(sparse)
    assert [p["environment_steps"] for p in left["training_history"]] == [2, 4, 6, 8, 10]
    assert [p["environment_steps"] for p in right["training_history"]] == [6, 10]
    assert [p["environment_steps"] for p in left["evaluation_history"]] == [0, 6, 10]
    assert left["updates"] == right["updates"]
    assert left["optimizer_steps"] == right["optimizer_steps"]
    for key, value in left_model.state_dict().items():
        torch.testing.assert_close(value, right_model.state_dict()[key], rtol=0, atol=0)
    for a, b in zip(left["evaluation_history"], right["evaluation_history"], strict=True):
        assert a["evaluation_returns"] == b["evaluation_returns"]
        for metric, value in a["mean_metrics_since_previous_evaluation"].items():
            assert value == pytest.approx(
                b["mean_metrics_since_previous_evaluation"][metric], rel=1e-5, abs=1e-6
            )
