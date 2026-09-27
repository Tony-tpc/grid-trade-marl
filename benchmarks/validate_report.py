"""机器可读的基准产物验收；完整性、数值和学习健康分开报告。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def validate_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    settings = report["settings"]
    budget = settings["target_environment_steps"] or (
        settings["train_episodes"] * settings["num_envs"] * settings["horizon"]
    )
    errors: list[str] = []
    warnings: list[str] = []
    runs = []
    for result in report["results"]:
        name = result["algorithm"]
        if name == "qmix":
            if result["status"] != "expected_incompatible" or not result.get("reason"):
                errors.append("QMIX 缺少预期不兼容解释")
            continue
        if result["status"] != "passed":
            errors.append(f"{name}: {result.get('failures', result.get('error'))}")
            continue
        if sorted(s["seed"] for s in result["seeds"]) != sorted(report["seeds"]):
            errors.append(f"{name}: seed 集合不完整或重复")
        for seed in result["seeds"]:
            label = f"{name}/{seed['seed']}"
            runs.append((name, seed["seed"]))
            if not seed["execution_complete"] or seed["environment_steps"] != budget:
                errors.append(f"{label}: 未完成预算")
            steps = [point["environment_steps"] for point in seed["evaluation_history"]]
            interval = report["audit_schedule"]["eval_interval_steps"]
            expected = sorted(set([0, *range(interval, budget + 1, interval), budget]))
            if steps != expected:
                errors.append(f"{label}: 评估历史缺失")
            for point in seed["training_history"]:
                if not all(np.isfinite(v) for v in point["metrics"].values()):
                    errors.append(f"{label}: 训练指标非有限")
            for point in seed["evaluation_history"]:
                arrays = [point["evaluation_returns"],
                          *point.get("fixed_opponent_evaluations", {}).values()]
                if not all(np.isfinite(np.asarray(array)).all() for array in arrays):
                    errors.append(f"{label}: 评估非有限")
                if settings["detailed_evaluation"] and len(
                    point.get("fixed_opponent_evaluations", {})
                ) != 9:
                    errors.append(f"{label}: 固定对手评估不完整")
            checkpoint = seed.get("checkpoint")
            if not checkpoint or not Path(checkpoint["path"]).is_file():
                errors.append(f"{label}: checkpoint 缺失")
                continue
            checkpoint_path = Path(checkpoint["path"])
            if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != checkpoint["sha256"]:
                errors.append(f"{label}: checkpoint 校验值不匹配")
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if not all(torch.isfinite(tensor).all() for tensor in state["algorithm"].values()):
                errors.append(f"{label}: 模型/统计非有限")
            if state["optimization"]["optimizer_step_count"] != seed["optimizer_steps"]:
                errors.append(f"{label}: optimizer-step 数不匹配")
            if seed["optimization_health"]["status"] == "warning":
                warnings.append(f"{label}: 优化警报，不能称为完全稳定")
    with path.with_suffix(".csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    csv_runs = [(row["algorithm"], int(row["seed"])) for row in rows]
    if sorted(csv_runs) != sorted(runs):
        errors.append("CSV 与 JSON 的算法/seed 集合不一致")
    return {"execution_and_artifacts_passed": not errors, "validated_runs": len(runs),
            "errors": errors, "optimization_warnings": warnings,
            "learning_evaluation": "必须另看固定对手曲线，数值有限不证明收敛"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    result = validate_report(args.report)
    output = args.report.with_suffix(".validation.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
