"""使用安全 YAML config 运行完整 MAPPO rollout/GAE/PPO 更新。

该示例仍使用合成的 P2P 电能交易曲线，只用于验证通用训练链，不代表论文复现结果。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from marl.algorithms import MAPPOConfig
from marl.config import load_algorithm_config
from marl.envs import build_energy_trading_adapter, load_environment_config
from marl.experiment import build_experiment
from marl.runtime import SyncVectorEnv, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="组合式 MAPPO YAML config 示例")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "algorithms" / "mappo.yaml",
    )
    parser.add_argument(
        "--environment",
        type=Path,
        default=(
            Path(__file__).parent
            / "configs"
            / "environments"
            / "energy_trading.yaml"
        ),
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if min(args.rounds, args.num_envs) < 1:
        parser.error("rounds 和 num-envs 必须大于 0")

    torch.manual_seed(args.seed)
    environment_config = load_environment_config(args.environment)
    environment = SyncVectorEnv(
        [
            build_energy_trading_adapter(environment_config)
            for _ in range(args.num_envs)
        ]
    )
    config = load_algorithm_config(args.config)
    device = resolve_device(args.device)
    if not isinstance(config, MAPPOConfig):
        parser.error("--config 必须选择 MAPPO 配置")
    experiment = build_experiment(
        environment, config, device=device, seed=args.seed,
    )
    trainer = experiment.trainer

    for round_index in range(args.rounds):
        seeds = [args.seed + round_index * args.num_envs + i for i in range(args.num_envs)]
        metrics = trainer.train_rollout(seeds)
        summary = " ".join(
            f"{name}={value:.4f}"
            for name, value in metrics.items()
            if name in {"loss", "policy_loss", "value_loss", "entropy", "approx_kl"}
        )
        print(f"round={round_index + 1} {summary}")


if __name__ == "__main__":
    main()
