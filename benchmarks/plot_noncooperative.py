"""为 MPE2 非合作博弈基准生成可复现的算法对比图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROLE_COLORS = {"adversary": "#0072B2", "good_team": "#D55E00"}
ROLE_LABELS = {"adversary": "Adversary", "good_team": "Good team"}
ACTION_GROUPS = {
    "discrete": ("maac", "mappo"),
    "continuous": ("maddpg", "masac"),
}


def _passed_results(report: dict[str, Any]) -> list[dict[str, Any]]:
    results = [result for result in report["results"] if result["status"] == "passed"]
    if not results:
        raise ValueError("报告中没有成功完成的算法")
    for result in results:
        if not result.get("seeds"):
            raise ValueError(f"{result['algorithm']} 缺少随机种子结果")
        if not all(seed.get("evaluation_history") for seed in result["seeds"]):
            raise ValueError(f"{result['algorithm']} 缺少定期评估历史")
    return results


def _confidence_interval(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.zeros(values.shape[1:], dtype=np.float64)
    interval = 1.96 * values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    return np.asarray(interval, dtype=np.float64)


def _save(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    output = output_dir / name
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def _plot_learning_curves(
    results: list[dict[str, Any]], output_dir: Path, group: str
) -> Path:
    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(6 * len(results), 5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for axis, result in zip(axes.flat, results, strict=False):
        histories = [seed["evaluation_history"] for seed in result["seeds"]]
        steps = np.asarray(
            [checkpoint["environment_steps"] for checkpoint in histories[0]],
            dtype=np.int64,
        )
        for role, linestyle in (("adversary", "-"), ("good_team", "--")):
            values = np.asarray(
                [
                    [checkpoint["mean_return_by_role"][role] for checkpoint in history]
                    for history in histories
                ],
                dtype=np.float64,
            )
            mean = values.mean(axis=0)
            interval = _confidence_interval(values)
            axis.plot(
                steps,
                mean,
                color=ROLE_COLORS[role],
                linestyle=linestyle,
                marker="o",
                markersize=3,
                label=ROLE_LABELS[role],
            )
            axis.fill_between(
                steps,
                mean - interval,
                mean + interval,
                color=ROLE_COLORS[role],
                alpha=0.15,
            )
        axis.axhline(0.0, color="0.55", linewidth=0.8)
        axis.set_title(result["algorithm"].upper())
        axis.grid(alpha=0.2)
    for axis in axes.flat:
        axis.set_xlabel("Environment steps")
    axes.flat[0].set_ylabel("Evaluation return")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"MPE2 Simple Adversary: {group.title()} Learning Curves", y=1.01)
    fig.tight_layout()
    return _save(fig, output_dir, f"learning_curves_{group}.png")


def _plot_final_returns(
    results: list[dict[str, Any]], output_dir: Path, group: str
) -> Path:
    algorithms = [result["algorithm"].upper() for result in results]
    positions = np.arange(len(results), dtype=np.float64)
    width = 0.36
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for offset, role in ((-width / 2, "adversary"), (width / 2, "good_team")):
        values = np.asarray(
            [
                [
                    seed["evaluation_history"][-1]["mean_return_by_role"][role]
                    for seed in result["seeds"]
                ]
                for result in results
            ],
            dtype=np.float64,
        )
        means = values.mean(axis=1)
        errors = np.asarray(
            [
                _confidence_interval(row[:, np.newaxis])[0]
                for row in values
            ]
        )
        bars = axis.bar(
            positions + offset,
            means,
            width,
            yerr=errors,
            capsize=4,
            color=ROLE_COLORS[role],
            alpha=0.85,
            label=ROLE_LABELS[role],
        )
        axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    axis.axhline(0.0, color="0.45", linewidth=0.9)
    axis.set_xticks(positions, algorithms)
    axis.set_ylabel("Final evaluation return (mean and 95% CI across seeds)")
    axis.set_title(f"Final {group.title()} Policy Performance by Role")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return _save(fig, output_dir, f"final_returns_{group}.png")


def _plot_return_distributions(
    results: list[dict[str, Any]], output_dir: Path, group: str
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, role in zip(axes, ("adversary", "good_team"), strict=True):
        distributions: list[list[float]] = []
        for result in results:
            samples: list[float] = []
            for seed in result["seeds"]:
                episodes = np.asarray(
                    seed["evaluation_history"][-1]["evaluation_returns"],
                    dtype=np.float64,
                )
                if role == "adversary":
                    samples.extend(episodes[:, 0].tolist())
                else:
                    samples.extend(episodes[:, 1:].mean(axis=1).tolist())
            distributions.append(samples)
        boxes = axis.boxplot(
            distributions,
            tick_labels=[result["algorithm"].upper() for result in results],
            patch_artist=True,
            showmeans=True,
        )
        for box in boxes["boxes"]:
            box.set_facecolor(ROLE_COLORS[role])
            box.set_alpha(0.65)
        axis.axhline(0.0, color="0.45", linewidth=0.9)
        axis.set_title(ROLE_LABELS[role])
        axis.set_xlabel("Algorithm")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Final evaluation episode return")
    fig.suptitle(f"Final {group.title()} Policy Return Distributions")
    fig.tight_layout()
    return _save(fig, output_dir, f"return_distributions_{group}.png")


def _plot_loss_stability(
    results: list[dict[str, Any]], output_dir: Path, group: str
) -> Path:
    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(6 * len(results), 5),
        sharex=True,
        squeeze=False,
    )
    for axis, result in zip(axes.flat, results, strict=False):
        series: list[np.ndarray] = []
        steps: np.ndarray | None = None
        for seed in result["seeds"]:
            history = [
                checkpoint
                for checkpoint in seed["evaluation_history"]
                if "loss" in checkpoint["mean_metrics_since_previous_evaluation"]
            ]
            steps = np.asarray(
                [checkpoint["environment_steps"] for checkpoint in history],
                dtype=np.int64,
            )
            values = np.asarray(
                [
                    checkpoint["mean_metrics_since_previous_evaluation"]["loss"]
                    for checkpoint in history
                ],
                dtype=np.float64,
            )
            series.append(values)
            axis.plot(steps, values, color="0.65", alpha=0.5, linewidth=1.0)
        stacked = np.stack(series)
        assert steps is not None
        axis.plot(
            steps,
            stacked.mean(axis=0),
            color="#009E73",
            marker="o",
            markersize=3,
            linewidth=2.0,
            label="Seed mean",
        )
        axis.axhline(0.0, color="0.55", linewidth=0.8)
        axis.set_title(result["algorithm"].upper())
        axis.grid(alpha=0.2)
    for axis in axes.flat:
        axis.set_xlabel("Environment steps")
    axes.flat[0].set_ylabel("Mean total loss per interval")
    fig.suptitle(f"{group.title()} Optimization Stability Across Seeds")
    fig.tight_layout()
    return _save(fig, output_dir, f"loss_stability_{group}.png")


def _plot_gradient_entropy_stability(
    results: list[dict[str, Any]], output_dir: Path, group: str
) -> Path:
    """绘制分 optimizer 梯度，以及离散熵或 SAC alpha 等稳定性诊断。"""

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), squeeze=False)
    candidates = (
        "actor_gradient_norm",
        "critic_gradient_norm",
        "temperature_gradient_norm",
        "entropy_ratio",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "alpha",
    )
    for axis, result in zip(axes.flat, results, strict=True):
        histories = [
            [
                checkpoint
                for checkpoint in seed["evaluation_history"]
                if checkpoint["mean_metrics_since_previous_evaluation"]
            ]
            for seed in result["seeds"]
        ]
        available = [
            metric
            for metric in candidates
            if histories
            and histories[0]
            and metric in histories[0][0]["mean_metrics_since_previous_evaluation"]
        ]
        for metric in available:
            values = np.asarray(
                [
                    [
                        checkpoint["mean_metrics_since_previous_evaluation"][metric]
                        for checkpoint in history
                    ]
                    for history in histories
                ],
                dtype=np.float64,
            )
            steps = np.asarray(
                [checkpoint["environment_steps"] for checkpoint in histories[0]],
                dtype=np.int64,
            )
            axis.plot(steps, values.mean(axis=0), marker="o", label=metric)
        axis.set_title(result["algorithm"].upper())
        axis.set_xlabel("Environment steps")
        axis.set_ylabel("Interval mean")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, frameon=False)
    fig.suptitle(f"{group.title()} Gradient and Policy Stability")
    fig.tight_layout()
    return _save(fig, output_dir, f"gradient_entropy_stability_{group}.png")


def _plot_cross_play(cross_play: dict[str, Any], output_dir: Path) -> Path:
    group = cross_play["action_kind"]
    labels = cross_play["adversary_policies"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, key, title in (
        (axes[0], "adversary_returns", "Adversary return"),
        (axes[1], "good_team_returns", "Good-team return"),
    ):
        values = np.asarray(cross_play[key], dtype=np.float64)
        image = axis.imshow(values, cmap="coolwarm", aspect="auto")
        axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(labels)), labels)
        axis.set_xlabel("Good-team policy source")
        axis.set_ylabel("Adversary policy source")
        axis.set_title(title)
        if len(labels) <= 6:
            for row in range(values.shape[0]):
                for column in range(values.shape[1]):
                    axis.text(
                        column,
                        row,
                        f"{values[row, column]:.1f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
        fig.colorbar(image, ax=axis, shrink=0.8)
    fig.suptitle(f"{group.title()} Cross-play by Role")
    fig.tight_layout()
    return _save(fig, output_dir, f"cross_play_{group}.png")


def _plot_training_efficiency(
    results: list[dict[str, Any]], output_dir: Path
) -> Path:
    algorithms = [result["algorithm"].upper() for result in results]
    times = np.asarray(
        [[seed["wall_clock_seconds"] for seed in result["seeds"]] for result in results],
        dtype=np.float64,
    )
    throughputs = np.asarray(
        [
            [seed["environment_steps"] / seed["wall_clock_seconds"] for seed in result["seeds"]]
            for result in results
        ],
        dtype=np.float64,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, values, title, ylabel in (
        (axes[0], times, "Wall-clock Cost", "Seconds"),
        (axes[1], throughputs, "Environment Throughput", "Environment steps / second"),
    ):
        means = values.mean(axis=1)
        errors = _confidence_interval(values.T)
        bars = axis.bar(
            algorithms,
            means,
            yerr=errors,
            capsize=4,
            color="#CC79A7",
            alpha=0.8,
        )
        axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Training Efficiency at Equal Environment-Step Budget")
    fig.tight_layout()
    return _save(fig, output_dir, "training_efficiency.png")


def generate_plots(report_path: Path, output_dir: Path) -> list[Path]:
    """按动作空间生成学习、终局、分布、稳定性、cross-play 与效率图。"""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = _passed_results(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for group, names in ACTION_GROUPS.items():
        grouped = [result for result in results if result["algorithm"] in names]
        if not grouped:
            continue
        outputs.extend(
            (
                _plot_learning_curves(grouped, output_dir, group),
                _plot_final_returns(grouped, output_dir, group),
                _plot_return_distributions(grouped, output_dir, group),
                _plot_loss_stability(grouped, output_dir, group),
                _plot_gradient_entropy_stability(grouped, output_dir, group),
            )
        )
    outputs.extend(
        _plot_cross_play(cross_play, output_dir)
        for cross_play in report.get("cross_play", [])
    )
    outputs.append(_plot_training_efficiency(results, output_dir))
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 MPE2 非合作博弈基准对比图")
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results/plots"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for output in generate_plots(args.report, args.output_dir):
        print(output.resolve())


if __name__ == "__main__":
    main()
