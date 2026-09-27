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
from matplotlib import font_manager  # noqa: E402
from scipy.stats import t as student_t  # type: ignore[import-untyped]  # noqa: E402

ROLE_COLORS = {"adversary": "#0072B2", "good_team": "#D55E00"}
ROLE_LABELS = {"adversary": "Adversary", "good_team": "Good team"}
ACTION_GROUPS = {
    "discrete": ("maac", "mappo"),
    "continuous": ("maddpg", "masac"),
}

ALGORITHM_COLORS = {"maac": "#0072B2", "mappo": "#D55E00",
                    "maddpg": "#009E73", "masac": "#CC79A7"}


def _configure_fonts() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _metric_series(seed: dict[str, Any], metric: str) -> tuple[np.ndarray, np.ndarray]:
    """优先使用密集训练历史；旧报告只使用真实存在的稀疏评估区间。"""
    if "training_history" in seed:
        points = [(p["environment_steps"], p["metrics"])
                  for p in seed["training_history"]]
    else:
        points = [(p["environment_steps"], p["mean_metrics_since_previous_evaluation"])
                  for p in seed["evaluation_history"]]
    measured = [(step, values[metric]) for step, values in points if metric in values]
    return (
        np.asarray([p[0] for p in measured], dtype=np.float64),
        np.asarray([p[1] for p in measured], dtype=np.float64),
    )


def _plot_metric(axis: Any, result: dict[str, Any], metric: str) -> None:
    """展示每个种子的原始区间值，以及共同采样步上的均值；不插值。"""
    series = [_metric_series(seed, metric) for seed in result["seeds"]]
    if not all(len(steps) for steps, _ in series):
        axis.text(0.5, 0.5, "未记录此指标 / unavailable", ha="center",
                  va="center", transform=axis.transAxes)
        return
    common = sorted(set.intersection(*(set(steps.tolist()) for steps, _ in series)))
    for steps, values in series:
        axis.plot(steps, values, color="0.65", alpha=0.5, linewidth=0.8)
    if common:
        rows = [dict(zip(steps.tolist(), values.tolist(), strict=True))
                for steps, values in series]
        mean = np.asarray([[row[step] for step in common] for row in rows]).mean(axis=0)
        axis.plot(common, mean, color=ALGORITHM_COLORS[result["algorithm"]],
                  linewidth=1.8, label="种子均值")
    axis.set_xlabel("环境交互步数")
    axis.set_title(f"{result['algorithm'].upper()} · {metric}")
    axis.grid(alpha=0.2)
    if "gradient_norm" in metric or "loss" in metric or metric == "alpha":
        axis.set_yscale("symlog", linthresh=1e-3)


def _evaluation_arrays(
    result: dict[str, Any], role: str,
) -> tuple[np.ndarray, np.ndarray]:
    histories = [seed["evaluation_history"] for seed in result["seeds"]]
    steps = np.asarray([p["environment_steps"] for p in histories[0]], dtype=np.float64)
    if any([p["environment_steps"] for p in h] != steps.tolist() for h in histories):
        raise ValueError("种子的评估步数不一致，不能用插值冒充共同评估")
    values = np.asarray([[p["mean_return_by_role"][role] for p in h] for h in histories])
    return steps, values


def _rolling_diagnostics(
    steps: np.ndarray, values: np.ndarray, window: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """values 为 [seed, evaluation]；计算完整窗口的斜率/时间标准差。

    斜率单位为每 1000 环境步的回报变化。不补点，不回填首个窗口，不判定纳什收敛。
    """
    if window < 2 or values.ndim != 2 or values.shape[1] != len(steps):
        raise ValueError("滚动诊断需要 window>=2 和 [seed,evaluation] 数据")
    if not np.isfinite(steps).all() or not np.isfinite(values).all():
        raise ValueError("滚动诊断不接受 NaN/Inf")
    if np.any(np.diff(steps) <= 0):
        raise ValueError("评估步数必须严格递增")
    slopes, deviations = [], []
    for end in range(window, len(steps) + 1):
        x = steps[end - window:end] / 1000.0
        x = x - x.mean()
        y = values[:, end - window:end]
        slopes.append((y * x).sum(axis=1) / np.square(x).sum())
        deviations.append(y.std(axis=1))
    if not slopes:
        empty = np.empty((values.shape[0], 0))
        return steps[:0], empty, empty.copy()
    return steps[window - 1:], np.stack(slopes, axis=1), np.stack(deviations, axis=1)


def _plot_return_comparison(results: list[dict[str, Any]], output_dir: Path, group: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, role in zip(axes, ("adversary", "good_team"), strict=True):
        for result in results:
            steps, values = _evaluation_arrays(result, role)
            color = ALGORITHM_COLORS[result["algorithm"]]
            for row in values:
                axis.plot(steps, row, color=color, alpha=0.2, linewidth=0.7)
            mean, interval = values.mean(axis=0), _confidence_interval(values)
            axis.plot(steps, mean, color=color, marker="o", markersize=2,
                      label=result["algorithm"].upper())
            axis.fill_between(steps, mean - interval, mean + interval, color=color, alpha=0.12)
        axis.set_title("对手回报" if role == "adversary" else "协作方回报")
        axis.set_xlabel("环境交互步数（含未训练的 step 0）")
        axis.set_ylabel("评估每局累计回报；越高越好")
        axis.legend()
        axis.grid(alpha=0.2)
    fig.suptitle(f"{group} · 算法学习曲线对比（细线为各 seed；阴影为近似 95% CI）")
    fig.tight_layout()
    return _save(fig, output_dir, f"return_comparison_{group}.png")


def _plot_convergence(results: list[dict[str, Any]], output_dir: Path, group: str) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row, role in enumerate(("adversary", "good_team")):
        for result in results:
            steps, values = _evaluation_arrays(result, role)
            x, slopes, deviations = _rolling_diagnostics(steps, values)
            if not len(x):
                for axis in axes[row]:
                    axis.text(0.5, 0.5, "不足 5 个评估点，无法计算", ha="center",
                              transform=axis.transAxes)
                continue
            for axis, data in zip(axes[row], (slopes, deviations), strict=True):
                for seed_values in data:
                    axis.plot(x, seed_values, color=ALGORITHM_COLORS[result["algorithm"]],
                              alpha=0.2, linewidth=0.7)
                axis.plot(x, data.mean(axis=0), label=result["algorithm"].upper(),
                          color=ALGORITHM_COLORS[result["algorithm"]])
        for column, axis in enumerate(axes[row]):
            axis.set_title(f"{ROLE_LABELS[role]} · 最近 5 个真实评估点")
            axis.set_ylabel("回报变化 / 1000 步" if column == 0 else "窗口内回报标准差")
            axis.set_xlabel("环境交互步数")
            axis.axhline(0, color="0.6", linewidth=0.7)
            axis.grid(alpha=0.2)
            if axis.get_legend_handles_labels()[0]:
                axis.legend()
    fig.suptitle(f"{group} · 收敛趋势诊断：斜率与波动趋近零不等于策略有效或达到均衡")
    fig.tight_layout()
    return _save(fig, output_dir, f"convergence_diagnostics_{group}.png")


def _plot_optimizer_losses(results: list[dict[str, Any]], output_dir: Path, group: str) -> Path:
    fig, axes = plt.subplots(2, len(results), figsize=(6 * len(results), 8), squeeze=False)
    for column, result in enumerate(results):
        metrics = ("policy_loss", "value_loss") if result["algorithm"] == "mappo" else (
            "actor_loss", "critic_loss"
        )
        for row, metric in enumerate(metrics):
            _plot_metric(axes[row, column], result, metric)
            axes[row, column].set_ylabel("区间平均损失（symlog）")
    fig.suptitle(f"{group} · actor 与 critic 分开审计；不同算法的损失不可直接排名")
    fig.tight_layout()
    return _save(fig, output_dir, f"optimizer_losses_{group}.png")


def _plot_progress_timing(results: list[dict[str, Any]], output_dir: Path, group: str) -> Path:
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4), squeeze=False)
    for axis, result in zip(axes.flat, results, strict=True):
        for seed in result["seeds"]:
            points = seed.get("training_history", seed["evaluation_history"])
            steps = np.asarray([0] + [p["environment_steps"] for p in points])
            elapsed = np.asarray([0.0] + [p["elapsed_seconds"] for p in points])
            dt, ds = np.diff(elapsed), np.diff(steps)
            valid = (dt > 0) & (ds > 0)
            axis.plot(steps[1:][valid], ds[valid] / dt[valid], label=f"seed {seed['seed']}")
        axis.set_title(result["algorithm"].upper())
        axis.set_xlabel("环境交互步数")
        axis.set_ylabel("区间步数 / 墙钟秒（含区间内评估开销）")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    fig.suptitle(f"{group} · 运行吞吐随训练进展的变化")
    fig.tight_layout()
    return _save(fig, output_dir, f"progress_timing_{group}.png")


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
    interval = (student_t.ppf(0.975, values.shape[0] - 1)
                * values.std(axis=0, ddof=1) / np.sqrt(values.shape[0]))
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
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), squeeze=False)
    for axis, result in zip(axes.flat, results, strict=True):
        _plot_metric(axis, result, "loss")
        axis.set_ylabel("总损失（symlog；细线为各 seed）")
    fig.suptitle(f"{group} · 训练损失历史；总损失不能用于跨算法排名")
    fig.tight_layout()
    return _save(fig, output_dir, f"loss_stability_{group}.png")


def _plot_gradient_entropy_stability(
    results: list[dict[str, Any]], output_dir: Path, group: str
) -> Path:
    """不同尺度的指标各自使用纵轴；绝不让大梯度遮住 KL/entropy。"""
    candidates = (
        "actor_gradient_norm", "critic_gradient_norm", "temperature_gradient_norm",
        "entropy_ratio", "approx_kl", "clip_fraction", "explained_variance", "alpha",
    )
    available = [metric for metric in candidates
                 if any(len(_metric_series(seed, metric)[0])
                        for result in results for seed in result["seeds"])]
    if not available:
        available = ["actor_gradient_norm"]
    fig, axes = plt.subplots(len(available), len(results),
                             figsize=(6 * len(results), 2.5 * len(available)), squeeze=False)
    for column, result in enumerate(results):
        for row, metric in enumerate(available):
            _plot_metric(axes[row, column], result, metric)
            axes[row, column].set_ylabel(metric)
    fig.suptitle(f"{group} · 优化与策略诊断（独立坐标；均值不能反推裁剪率）")
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


def _plot_agent_diagnostics(
    results: list[dict[str, Any]], output_dir: Path, group: str
) -> Path:
    families = (("value", "abs_max"), ("target", "abs_max"),
                ("value_rmse", ""), ("entropy", "mean"), ("alpha", "mean"))
    fig, axes = plt.subplots(len(families), len(results),
                             figsize=(6 * len(results), 14), squeeze=False)
    for column, result in enumerate(results):
        agents = len(result["seeds"][0]["agent_ids"])
        for row, (prefix, suffix) in enumerate(families):
            axis = axes[row, column]
            present = False
            for agent in range(agents):
                metric = f"{prefix}_agent_{agent}" + (f"_{suffix}" if suffix else "")
                histories = [_metric_series(seed, metric) for seed in result["seeds"]]
                for index, (steps, values) in enumerate(histories):
                    if len(steps):
                        present = True
                        axis.plot(steps, values, color=f"C{agent}", alpha=0.55,
                                  label=f"agent {agent}" if index == 0 else None)
            axis.set_title(f"{result['algorithm'].upper()} · {prefix}")
            axis.set_ylabel("原始尺度（各 seed；非归一化 loss）")
            axis.set_xlabel("环境步")
            if prefix in ("value", "target", "value_rmse", "alpha"):
                axis.set_yscale("symlog", linthresh=1.)
            if present:
                axis.legend(fontsize=8)
            else:
                axis.text(.5, .5, "未记录 / 不适用", transform=axis.transAxes, ha="center")
    fig.suptitle(f"{group} · 逐智能体原始尺度诊断；归一化损失不能掩盖价值漂移")
    fig.tight_layout()
    return _save(fig, output_dir, f"agent_diagnostics_{group}.png")


def _plot_clip_rates(
    results: list[dict[str, Any]], output_dir: Path, group: str
) -> Path:
    fig, axes = plt.subplots(3, len(results), figsize=(6 * len(results), 9), squeeze=False)
    for column, result in enumerate(results):
        for row, optimizer in enumerate(("actor", "critic", "temperature")):
            _plot_metric(axes[row, column], result, f"{optimizer}_gradient_clip_rate")
            axes[row, column].set_ylim(-.02, 1.02)
            axes[row, column].set_ylabel(f"{optimizer} 实际裁剪步数占比")
    fig.tight_layout()
    return _save(fig, output_dir, f"gradient_clip_rates_{group}.png")


def _plot_fixed_opponents(
    results: list[dict[str, Any]], output_dir: Path, group: str, mode: str
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)
    for row, role in enumerate(("adversary", "good_team")):
        for column, opponent in enumerate(("random", "untrained")):
            axis = axes[row, column]
            for result in results:
                histories = []
                sampled_steps: list[int] = []
                for seed in result["seeds"]:
                    points = seed["evaluation_history"]
                    key = f"{opponent}/{mode}/{role}"
                    if not all(key in p.get("fixed_opponent_evaluations", {}) for p in points):
                        continue
                    steps = [p["environment_steps"] for p in points]
                    values = []
                    for point in points:
                        samples = np.asarray(point["fixed_opponent_evaluations"][key])
                        values.append(float(samples[:, 0].mean() if role == "adversary"
                                            else samples[:, 1:].mean()))
                    if sampled_steps and steps != sampled_steps:
                        raise ValueError("固定对手评估的 seed 采样步数不一致")
                    sampled_steps = steps
                    histories.append(values)
                    axis.plot(steps, values, color=ALGORITHM_COLORS[result["algorithm"]],
                              alpha=.2, linewidth=.8)
                if histories:
                    array = np.asarray(histories)
                    mean, ci = array.mean(0), _confidence_interval(array)
                    axis.plot(sampled_steps, mean, label=result["algorithm"].upper(),
                              color=ALGORITHM_COLORS[result["algorithm"]])
                    axis.fill_between(sampled_steps, mean-ci, mean+ci, alpha=.15,
                                      color=ALGORITHM_COLORS[result["algorithm"]])
            axis.set_title(f"训练角色 {role} / 固定对手 {opponent}")
            axis.set_xlabel("环境步")
            axis.set_ylabel("该角色回报（越高越好）")
            if axis.lines:
                axis.legend()
    fig.suptitle(f"{group} · {mode} · 固定对手学习曲线（均值与 Student-t 95% 区间）")
    fig.tight_layout()
    return _save(fig, output_dir, f"fixed_opponents_{group}_{mode}.png")


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


def _write_plot_guide(
    results: list[dict[str, Any]], output_dir: Path, outputs: list[Path],
) -> None:
    """将实际审计覆盖率与读图限制随图片一起交付，旧记录绝不补造。"""
    coverage = []
    for result in results:
        for seed in result["seeds"]:
            history = seed["evaluation_history"]
            steps = [point["environment_steps"] for point in history]
            coverage.append({
                "algorithm": result["algorithm"], "seed": seed["seed"],
                "evaluation_points": len(history),
                "training_points": len(seed.get("training_history", [])),
                "max_evaluation_gap_steps": max(np.diff(steps).tolist(), default=0),
                "dense_training_history_available": "training_history" in seed,
                "convergence_assessment": "not_established",
            })
    (output_dir / "plot_audit.json").write_text(
        json.dumps({"sampling_coverage": coverage, "interpolated_points": 0,
                    "convergence_note": "斜率/波动趋零不能证明策略有效或博弈均衡"},
                   ensure_ascii=False, indent=2), encoding="utf-8",
    )
    lines = [
        "# 基准图表阅读与审计说明", "",
        "图片只使用报告中真实存在的数据；连线仅辅助阅读，不是新增评估。", "",
        "| 算法 | seed | 评估点（含 step 0） | 密集训练诊断点 | 最大评估间隔 |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['algorithm']} | {row['seed']} | {row['evaluation_points']} | "
        f"{row['training_points']} | {row['max_evaluation_gap_steps']} |" for row in coverage
    )
    lines.extend([
        "", "## 推荐阅读顺序", "",
        "1. return_comparison：同动作空间、同角色的算法叠加曲线；细线是各 seed。",
        "2. convergence_diagnostics：最近 5 个真实评估点的回报斜率与时间标准差。",
        "3. optimizer_losses：actor 与 critic 损失分开，symlog 轴保留正负及大数值。",
        "4. gradient_entropy_stability：每项指标独立纵轴，查看梯度、熵、KL、alpha。",
        "5. progress_timing：逐区间墙钟吞吐，包含区间内评估开销，不等同于纯训练速度。",
        "", "## 解释限制", "",
        "- 回报越高越好，负回报越接近 0 越好。双方收益分别评估，不合成总分。",
        "- 回报阴影是跨 seed 的 Student-t 95% 区间；少量种子下不应解释成显著性证明。",
        "- 斜率接近 0 也可能是策略停滞；需要同时看回报、波动和优化诊断。",
        "- 自博弈双方同时变化；这些曲线不能替代固定对手测试或证明纳什均衡。",
        "- 旧报告没有密集训练历史时，只能显示评估间隔的均值，无法恢复中间尖峰。",
        "- 旧 gradient_clip_rate 来自均值阈值判断，不是真实裁剪率，本次不绘制。",
        "- 没记录的指标显示为缺失，不补零。少于 5 个评估点不计算滚动趋势。",
        "", "## 图片", "",
    ])
    lines.extend(f"- [{path.name}]({path.name})" for path in outputs)
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_plots(report_path: Path, output_dir: Path) -> list[Path]:
    """按动作空间生成学习、终局、分布、稳定性、cross-play 与效率图。"""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    _configure_fonts()
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
                _plot_return_comparison(grouped, output_dir, group),
                _plot_convergence(grouped, output_dir, group),
                _plot_optimizer_losses(grouped, output_dir, group),
                _plot_progress_timing(grouped, output_dir, group),
            )
        )
        if any(seed.get("gradient_clip_rate_available", False)
               for result in grouped for seed in result["seeds"]):
            outputs.extend((_plot_agent_diagnostics(grouped, output_dir, group),
                            _plot_clip_rates(grouped, output_dir, group)))
        if any(point.get("fixed_opponent_evaluations")
               for result in grouped for seed in result["seeds"]
               for point in seed["evaluation_history"]):
            outputs.extend(_plot_fixed_opponents(grouped, output_dir, group, mode)
                           for mode in ("deterministic", "stochastic"))
    outputs.extend(
        _plot_cross_play(cross_play, output_dir)
        for cross_play in report.get("cross_play", [])
    )
    outputs.append(_plot_training_efficiency(results, output_dir))
    _write_plot_guide(results, output_dir, outputs)
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
