"""从新 adapter 训练、保存、重建和续训；在仓库根目录运行，输出到忽略目录。"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from cue_environment import CueEnvironment
from torch import nn

from marl.algorithms import MAACConfig, MAPPOConfig
from marl.envs import EnvironmentSpec
from marl.experiment import build_experiment
from marl.models import LSTMBackboneConfig, MLPBackboneConfig
from marl.modules.critic import AttentionQConfig
from marl.modules.policy import IndependentDiscreteConfig
from marl.runtime import SyncVectorEnv
from marl.training.off_policy import OffPolicyCollector, OffPolicyTrainer, ReplayConfig
from marl.training.on_policy import OnPolicyTrainer, PPOUpdateConfig, RolloutConfig


# 与 examples/custom_component.py 相同的 Python-only 扩展方式。
@dataclass(frozen=True)
class TanhValueConfig:
    """教学用外部 critic 配置；输入 state [...,S]，输出 value [...,N]。"""

    hidden_dim: int = 16
    kind: str = "tutorial_tanh_value"

    def __post_init__(self) -> None:
        """无显式输入；校验配置宽度，非法值抛 ValueError。"""
        if type(self.hidden_dim) is not int or self.hidden_dim < 1:
            raise ValueError("hidden_dim 必须为正整数")

    def build(self, spec: EnvironmentSpec) -> nn.Sequential:
        """输入环境 spec；返回注册了全部参数、输出逐智能体 value 的网络。"""
        return nn.Sequential(
            nn.Linear(spec.state_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, spec.num_agents),
        )


def main() -> None:
    """输入 CLI 参数；输出每轮 JSON 指标和 checkpoint，并实际恢复后再更新一轮。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("mappo", "maac"), default="mappo")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recurrent", action="store_true")
    parser.add_argument("--custom-critic", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/doc-tutorial"))
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds 必须为正")
    if args.algorithm != "mappo" and (args.recurrent or args.custom_critic):
        parser.error("recurrent/custom-critic 仅在本例 MAPPO 分支使用")
    torch.set_num_threads(1)
    vector = SyncVectorEnv([CueEnvironment(), CueEnvironment()])
    config: MAPPOConfig | MAACConfig
    if args.algorithm == "mappo":
        from marl.modules.critic import CentralizedValueConfig

        backbone = (
            LSTMBackboneConfig(hidden_dim=16)
            if args.recurrent
            else MLPBackboneConfig(hidden_dims=(16,), output_dim=16)
        )
        config = MAPPOConfig(
            policy=IndependentDiscreteConfig(backbone=backbone),
            critic=(
                TanhValueConfig(hidden_dim=16)
                if args.custom_critic
                else CentralizedValueConfig(
                    backbone=MLPBackboneConfig(hidden_dims=(16,), output_dim=16)
                )
            ),
            rollout=RolloutConfig(horizon=8),
            update=PPOUpdateConfig(
                epochs=1,
                mini_batch_size=16,
                sequence_length=4 if args.recurrent else None,
            ),
        )
    else:
        config = MAACConfig(
            policy=IndependentDiscreteConfig(
                backbone=MLPBackboneConfig(hidden_dims=(16,), output_dim=16)
            ),
            critic=AttentionQConfig(hidden_dim=16, attention_heads=4),
            replay=ReplayConfig(capacity=128, batch_size=8),
        )
    experiment = build_experiment(vector, config, device="cpu", seed=args.seed)

    def train_round(
        trainer: OnPolicyTrainer | OffPolicyTrainer,
        model: nn.Module,
        round_index: int,
    ) -> dict[str, float]:
        """输入 trainer/模型/轮次；输出有限指标，采集 seed 从基础 seed 和轮次确定。"""
        seeds = [args.seed + 2 * round_index + index for index in range(2)]
        if isinstance(trainer, OnPolicyTrainer):
            metrics = trainer.train_rollout(seeds)
        else:
            # experiment.algorithm 的具体类型提供 act；trainer.algorithm 协议仅要求 update。
            from marl.algorithms.maac import MAAC

            assert isinstance(model, MAAC)
            collector = OffPolicyCollector(vector, model)
            metrics = {}
            for transitions in collector.rollout(seeds):
                trainer.record_many(transitions)
                if trainer.ready:
                    metrics = trainer.update()
        if not metrics or not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError("本轮未产生有效训练指标")
        print(json.dumps({"round": round_index, "loss": metrics["loss"]}))
        return metrics

    for round_index in range(args.rounds):
        train_round(experiment.trainer, experiment.algorithm, round_index)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / f"{args.algorithm}.pt"
    experiment.trainer.save_checkpoint(checkpoint)
    # 外部 seed 调度不由 trainer 保存；实验层单独记录下一轮位置和 CLI 选择。
    progress = args.output / f"{args.algorithm}-progress.json"
    progress.write_text(
        json.dumps(
            {
                "next_round": args.rounds,
                "seed": args.seed,
                "algorithm": args.algorithm,
                "recurrent": args.recurrent,
                "custom_critic": args.custom_critic,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    restored = build_experiment(vector, config, device="cpu", seed=args.seed)
    restored.trainer.load_checkpoint(checkpoint)
    for name, value in experiment.algorithm.state_dict().items():
        torch.testing.assert_close(value, restored.algorithm.state_dict()[name], rtol=0, atol=0)
    assert (
        restored.trainer.optimization.optimizer_step_count
        == experiment.trainer.optimization.optimizer_step_count
    )
    next_round = json.loads(progress.read_text(encoding="utf-8"))["next_round"]
    train_round(restored.trainer, restored.algorithm, next_round)
    print(f"saved={checkpoint}; restored and updated successfully")


if __name__ == "__main__":
    main()
