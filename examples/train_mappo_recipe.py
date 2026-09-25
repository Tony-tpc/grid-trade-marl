"""使用安全 YAML recipe 运行完整 MAPPO rollout/GAE/PPO 更新。

该示例仍使用合成的 P2P 电能交易曲线，只用于验证通用训练链，不代表论文复现结果。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from marl.algorithms import MAPPO
from marl.envs import ActionKind, EnergyTradingAdapter, EnergyTradingConfig, EnergyTradingEnv
from marl.recipes import load_algorithm_recipe
from marl.runtime import SyncVectorEnv, resolve_device
from marl.training import OnPolicyTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="组合式 MAPPO YAML recipe 示例")
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path(__file__).parent / "configs" / "mappo.yaml",
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if min(args.rounds, args.agents, args.num_envs) < 1:
        parser.error("rounds、agents 和 num-envs 必须大于 0")

    torch.manual_seed(args.seed)
    environment = SyncVectorEnv(
        [
            EnergyTradingAdapter(
                EnergyTradingEnv(EnergyTradingConfig(num_agents=args.agents)),
                ActionKind.DISCRETE,
            )
            for _ in range(args.num_envs)
        ]
    )
    recipe = load_algorithm_recipe(args.recipe)
    device = resolve_device(args.device)
    algorithm = MAPPO.from_recipe(environment.spec, recipe).to(device)
    trainer = OnPolicyTrainer.from_recipe(
        environment,
        algorithm,
        recipe,
        device=device,
        generator=torch.Generator().manual_seed(args.seed),
    )

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
