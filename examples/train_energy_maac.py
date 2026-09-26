"""个体收益 P2P 博弈：环境 -> 适配器 -> MARLBatch -> MAAC。

运行：.venv\\Scripts\\python.exe examples\\train_energy_maac.py --episodes 5
使用合成曲线与 54 个量化动作，仅演示训练接口，不代表论文数值复现。
每个 round 同时采集 num-envs 条 episode，--episodes 表示 round 数。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import cast

import numpy as np
import torch

from marl.algorithms import MAAC, default_maac_recipe
from marl.envs import (
    ActionKind,
    EnergyTradingAdapter,
    EnergyTradingConfig,
    EnergyTradingEnv,
    Transition,
)
from marl.recipes import ComponentRecipe
from marl.runtime import SyncVectorEnv, resolve_device
from marl.training import OffPolicyTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="个体收益电能交易博弈中的 MAAC 训练示例")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", choices=("none", "bf16"), default="none")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.episodes, args.agents, args.batch_size, args.num_envs, args.hidden_dim) < 1:
        parser.error("episodes、agents、batch-size、num-envs、hidden-dim 必须大于零")
    if args.hidden_dim % 4:
        parser.error("hidden-dim 必须能被 MAAC 的 4 个 attention heads 整除")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    vector_env = SyncVectorEnv(
        [
            EnergyTradingAdapter(
                EnergyTradingEnv(EnergyTradingConfig(num_agents=args.agents)),
                ActionKind.DISCRETE,
            )
            for _ in range(args.num_envs)
        ]
    )
    device = resolve_device(args.device)
    recipe = replace(
        default_maac_recipe(),
        policy=ComponentRecipe(
            "independent_discrete", {"hidden_dim": args.hidden_dim}
        ),
        critic=ComponentRecipe(
            "attention_q",
            {"hidden_dim": args.hidden_dim, "attention_heads": 4},
        ),
        experience=ComponentRecipe(
            "replay", {"capacity": 5000, "batch_size": args.batch_size}
        ),
        update=ComponentRecipe(
            "off_policy_update",
            {
                "learning_rate": 3e-4,
                "max_grad_norm": 10.0,
                "amp_dtype": "bf16" if args.amp == "bf16" else None,
            },
        ),
    )
    algorithm = MAAC.from_recipe(vector_env.spec, recipe).to(device)
    trainer = OffPolicyTrainer.from_recipe(
        vector_env.spec,
        algorithm,
        recipe,
        device=device,
        rng=rng,
    )
    print(f"device={device} amp={args.amp} num_envs={args.num_envs}")

    for episode in range(args.episodes):
        current = vector_env.reset(
            [args.seed + episode * args.num_envs + i for i in range(args.num_envs)]
        )
        individual_returns = np.zeros((args.num_envs, vector_env.spec.num_agents))
        episode_cost = np.zeros(args.num_envs)
        epsilon = max(0.1, 0.8 - 0.7 * episode / max(args.episodes - 1, 1))
        metrics: dict[str, float] | None = None
        while not all(step.done for step in current):
            masks = np.stack([cast(np.ndarray, step.action_mask) for step in current])
            explore = rng.random(args.num_envs) < epsilon
            actions = np.empty((args.num_envs, vector_env.spec.num_agents), dtype=np.int64)
            if not explore.all():
                observations = torch.as_tensor(
                    np.stack([step.observations for step in current])
                ).to(device, non_blocking=True)
                action_mask = torch.as_tensor(masks).to(device, non_blocking=True)
                with torch.inference_mode():
                    actions[:] = algorithm.act(
                        observations, action_mask=action_mask
                    ).cpu().numpy()
            for random_env_index in np.flatnonzero(explore):
                actions[random_env_index] = [
                    rng.choice(np.flatnonzero(mask))
                    for mask in masks[random_env_index]
                ]

            next_steps = vector_env.step(actions)
            for env_index, next_step in enumerate(next_steps):
                trainer.record(
                    Transition(current[env_index], actions[env_index], next_step)
                )
                individual_returns[env_index] += next_step.rewards
                episode_cost[env_index] += next_step.info["community_energy_cost"]
            current = next_steps

            if trainer.ready:
                metrics = trainer.update()

        loss_text = (
            f"{metrics['loss']:.3f}" if metrics is not None else "no_update"
        )
        print(
            f"round={episode + 1} episodes={args.num_envs} steps={vector_env.spec.horizon} "
            f"mean_individual_returns={individual_returns.mean(axis=0).round(2).tolist()} "
            f"mean_community_cost={episode_cost.mean():.2f} loss={loss_text}"
        )


if __name__ == "__main__":
    main()
