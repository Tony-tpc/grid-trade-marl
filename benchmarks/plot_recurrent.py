"""绘制循环 MAPPO 对照的真实逐 seed 历史，不插值或拼接缺失数据。"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t  # type: ignore[import-untyped]

KINDS = ("mlp", "gru", "lstm")


def _curves(
    rows: list[dict[str, Any]],
    history: str,
    fields: Sequence[tuple[str, Callable[[dict[str, Any]], float | None]]],
    title: str,
    target: Path,
) -> None:
    fig, axes = plt.subplots(
        math.ceil(len(fields) / 2), 2, figsize=(12, 3.8 * math.ceil(len(fields) / 2)), squeeze=False
    )
    for axis, (label, read) in zip(axes.flat, fields, strict=False):
        for kind in KINDS:
            selected = [row for row in rows if row["variant_id"] == kind]
            if not selected or any(read(p) is None for row in selected for p in row[history]):
                continue
            steps = [p["environment_steps"] for p in selected[0][history]]
            if any([p["environment_steps"] for p in row[history]] != steps for row in selected):
                raise ValueError("曲线采样点不一致，禁止插值冒充完整历史")
            values = np.asarray([[read(p) for p in row[history]] for row in selected], dtype=float)
            mean = values.mean(0)
            ci = (
                (
                    student_t.ppf(0.975, len(selected) - 1)
                    * values.std(0, ddof=1)
                    / np.sqrt(len(selected))
                )
                if len(selected) > 1
                else np.zeros_like(mean)
            )
            (line,) = axis.plot(steps, mean, label=f"{kind.upper()} (n={len(selected)})")
            for sample in values:
                axis.plot(steps, sample, color=line.get_color(), alpha=0.2, linewidth=0.7)
            if len(selected) > 1:
                axis.fill_between(steps, mean - ci, mean + ci, color=line.get_color(), alpha=0.15)
        axis.set(xlabel="Joint environment steps", ylabel=label)
        axis.grid(alpha=0.25)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
        else:
            axis.text(
                0.5, 0.5, "Not applicable / not recorded", ha="center", transform=axis.transAxes
            )
    for axis in list(axes.flat)[len(fields) :]:
        axis.set_visible(False)
    fig.suptitle(title + "\nThin: individual seeds; band: 95% Student-t CI (few seeds)")
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def _metric(name: str) -> Callable[[dict[str, Any]], float | None]:
    return lambda point: point["metrics"].get(name)


def plot(report: dict[str, Any], directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    groups = {
        "loss": ["policy_loss", "value_loss"],
        "ppo": ["entropy_ratio", "approx_kl", "clip_fraction", "rollout_explained_variance"],
        "gradients": [
            "actor_gradient_norm",
            "critic_gradient_norm",
            "actor_gradient_clip_rate",
            "critic_gradient_clip_rate",
        ],
        "recurrent_state": ["policy_hidden_norm", "policy_cell_norm", "padding_fraction"],
    }
    for task in report["tasks"]:
        rows = [row for row in report["runs"] if row["task"] == task]
        if task == "memory":
            fields = [
                (
                    "Success rate",
                    lambda p: float((np.asarray(p["evaluation_returns"]) + 1).mean() / 2),
                ),
                ("Individual return (mean)", lambda p: float(np.mean(p["evaluation_returns"]))),
            ]
        else:
            fields = [
                (
                    "Adversary return",
                    lambda p: float(np.asarray(p["evaluation_returns"])[:, 0].mean()),
                ),
                (
                    "Good-agent mean return",
                    lambda p: float(np.asarray(p["evaluation_returns"])[:, 1:].mean()),
                ),
            ]
        path = directory / f"{task}_learning.png"
        _curves(
            rows, "evaluation_history", fields, f"{task}: learning, not convergence proof", path
        )
        files.append(path)
        for name, metrics in groups.items():
            path = directory / f"{task}_{name}.png"
            _curves(
                rows,
                "training_history",
                [(m, _metric(m)) for m in metrics],
                f"{task}: {name}",
                path,
            )
            files.append(path)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for axis, (label, extract) in zip(
            axes,
            [
                (
                    "End-to-end env steps/s (includes evaluation)",
                    lambda r: r["environment_steps"] / r["total_seconds"],
                ),
                ("Trainable parameter count", lambda r: r["parameters"]),
                ("Optimizer steps", lambda r: r["optimizer_steps"]),
            ],
            strict=True,
        ):
            data = [[extract(r) for r in rows if r["variant_id"] == kind] for kind in KINDS]
            axis.bar(
                KINDS,
                [np.mean(v) for v in data],
                yerr=[np.std(v, ddof=1) if len(v) > 1 else 0 for v in data],
                capsize=4,
            )
            axis.set_title(label, fontsize=9)
        fig.tight_layout()
        path = directory / f"{task}_efficiency.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        files.append(path)

        fig, axis = plt.subplots(figsize=(9, 4))
        stages = ("collection", "transfer", "update", "evaluation")
        for i, kind in enumerate(KINDS):
            data = [
                np.mean([r["profile"].get(stage, 0) for r in rows if r["variant_id"] == kind])
                for stage in stages
            ]
            axis.bar(np.arange(4) + i * 0.25, data, 0.25, label=kind)
        axis.set_xticks(np.arange(4) + 0.25, stages)
        axis.set_ylabel("Host cumulative seconds (overlapping; NOT additive)")
        axis.legend()
        fig.tight_layout()
        path = directory / f"{task}_timing.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        files.append(path)

    if "cross_play" in report:
        cross = report["cross_play"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for axis, role in zip(axes, ("adversary", "good_team"), strict=True):
            matrix = np.asarray(cross[role])
            im = axis.imshow(matrix, cmap="viridis", aspect="equal")
            axis.set_xticks(range(len(matrix)), cross["labels"], rotation=75)
            axis.set_yticks(range(len(matrix)), cross["labels"])
            axis.set(xlabel="Good-team source", ylabel="Adversary source", title=role)
            fig.colorbar(im, ax=axis, shrink=0.8)
        fig.tight_layout()
        path = directory / "cross_play.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        files.append(path)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        rows = [r for r in report["runs"] if r["task"] == "mpe"]
        for axis, role in zip(axes, ("adversary", "good_team"), strict=True):
            distributions: list[np.ndarray] = []
            for kind in KINDS:
                returns = np.concatenate(
                    [r["heldout_returns"] for r in rows if r["variant_id"] == kind]
                )
                distributions.append(
                    returns[:, 0] if role == "adversary" else returns[:, 1:].mean(-1)
                )
            axis.boxplot(distributions, tick_labels=KINDS)
            axis.set_title(f"Held-out {role} episode returns")
        fig.suptitle(
            "Descriptive pooled distributions; episodes are not independent training seeds"
        )
        fig.tight_layout()
        path = directory / "heldout_returns.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = plot(json.loads(args.report.read_text(encoding="utf-8")), args.output)
    print(f"Generated {len(result)} plots in {args.output}")


if __name__ == "__main__":
    main()
