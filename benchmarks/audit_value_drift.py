"""在旧异常 checkpoint 上做 critic-only 固定批次干预，不覆盖模型或宣称单一因果。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from marl.algorithms import MAAC, MADDPG, MASAC
from marl.config import algorithm_config_from_dict
from marl.experiment import build_experiment
from marl.returns import ValueTargetEstimator
from marl.runtime import TensorReplayBuffer


class FixedTarget:
    """仅供诊断：拦截 target estimator，不复制算法的 critic 数学。"""

    def __init__(self, estimator: ValueTargetEstimator, rewards_only: bool) -> None:
        self.estimator = estimator
        self.rewards_only = rewards_only
        self.target: Tensor | None = None

    def estimate(self, rewards: Tensor, next_values: Tensor, terminated: Tensor) -> Tensor:
        if self.target is None:
            target = (rewards if self.rewards_only else
                      self.estimator.estimate(rewards, next_values, terminated))
            self.target = target.detach().clone()
        return self.target


def audit_checkpoint(path: Path, steps: int = 100) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = algorithm_config_from_dict(state["config"])
    replay_state = state["replay"]
    spec = replay_state["spec"]
    replay = TensorReplayBuffer(spec, replay_state["capacity"], pin_memory=False)
    # 显式 CPU 诊断副本，旧模型仅加载参数而非伪装成精确续训。
    replay.load_state_dict({**replay_state, "pin_memory": False})
    batch = replay.sample(128, np.random.default_rng(731))
    interventions = {}
    for mode in ("reward_supervision", "fixed_bootstrap_target", "moving_bootstrap_target"):
        experiment = build_experiment(spec, config, device="cpu", seed=731)
        algorithm = experiment.algorithm
        assert isinstance(algorithm, (MAAC, MADDPG, MASAC))
        algorithm.load_state_dict(state["algorithm"])
        runtime = experiment.trainer.optimization
        for name, optimizer in runtime.optimizers.items():
            optimizer.load_state_dict(state["optimization"]["optimizers"][name])
        runtime.update_count = state["optimization"]["update_count"]
        runtime.sync_metrics = True
        if mode != "moving_bootstrap_target":
            algorithm.return_estimator = FixedTarget(
                algorithm.return_estimator, mode == "reward_supervision"
            )
        trace = []
        for step in range(steps + 1):
            # 相同随机样本（MAAC/MASAC 的 bootstrap 采样），排除 RNG 波动。
            torch.manual_seed(731)
            bundle = algorithm.compute_critic_loss_bundle(batch)
            trace.append({"step": step, **{
                name: float(value.detach()) for name, value in bundle.terms.items()
            }})
            if step < steps:
                algorithm.optimize(runtime.optimizer("critic"), bundle,
                                   runtime.max_grad_norm("critic"),
                                   parameters=runtime.parameters("critic"))
                if mode == "moving_bootstrap_target":
                    runtime.finish(algorithm.target_pairs())
        interventions[mode] = trace
    return {
        "checkpoint": str(path.resolve()), "steps": steps, "batch_size": 128,
        "actor_frozen": True, "optimizer_restored": True, "seed": 731,
        "interventions": interventions,
        "interpretation": (
            "reward_supervision 仅测试拟合有限固定标签能力，不是正确的长期训练目标；"
            "固定/移动 bootstrap 对照局限于这一批数据和局部 critic-only 更新，"
            "不能单独推断完整自博弈训练的失稳因果。"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = [audit_checkpoint(path, args.steps) for path in args.checkpoints]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
