"""个体收益 P2P 博弈：环境 -> 适配器 -> MARLBatch -> MAAC。

运行：.venv\\Scripts\\python.exe examples\\train_energy_maac.py --episodes 5
使用合成曲线与 54 个量化动作，仅演示训练接口，不代表论文数值复现。
每个 round 同时采集 num-envs 条 episode，--episodes 表示 round 数。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from marl.algorithms import MAACConfig
from marl.config import load_algorithm_config
from marl.envs import (
    Transition,
    build_energy_trading_adapter,
    load_environment_config,
)
from marl.experiment import build_experiment
from marl.runtime import SyncVectorEnv, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="个体收益电能交易博弈中的 MAAC 训练示例")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "algorithms" / "maac.yaml",
    )
    parser.add_argument(
        "--environment",
        type=Path,
        default=(Path(__file__).parent / "configs" / "environments" / "energy_trading.yaml"),
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="在完整更新边界保存 schema 5 网络、布局、optimizer 和 replay",
    )
    args = parser.parse_args()
    if min(args.episodes, args.num_envs) < 1:
        parser.error("episodes 和 num-envs 必须大于零")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    environment_config = load_environment_config(args.environment)
    vector_env = SyncVectorEnv(
        [build_energy_trading_adapter(environment_config) for _ in range(args.num_envs)]
    )
    device = resolve_device(args.device)
    config = load_algorithm_config(args.config)
    if not isinstance(config, MAACConfig):
        parser.error("--config 必须选择 MAAC 配置")
    experiment = build_experiment(
        vector_env,
        config,
        device=device,
        seed=args.seed,
    )
    algorithm = experiment.algorithm
    trainer = experiment.trainer
    print(f"device={device} num_envs={args.num_envs}")

    for episode in range(args.episodes):
        current = vector_env.reset(
            [args.seed + episode * args.num_envs + i for i in range(args.num_envs)]
        )
        individual_returns = np.zeros((args.num_envs, vector_env.spec.num_agents))
        episode_cost = np.zeros(args.num_envs)
        epsilon = max(0.1, 0.8 - 0.7 * episode / max(args.episodes - 1, 1))
        metrics: dict[str, float] | None = None
        while not all(step.done for step in current):
            available_masks = [step.action_mask for step in current]
            if any(mask is None for mask in available_masks):
                raise ValueError("MAAC 环境必须提供 action mask")
            masks = np.stack([mask for mask in available_masks if mask is not None])
            explore = rng.random(args.num_envs) < epsilon
            actions = np.empty((args.num_envs, vector_env.spec.num_agents), dtype=np.int64)
            if not explore.all():
                observations = torch.as_tensor(
                    np.stack([step.observations for step in current])
                ).to(device, non_blocking=True)
                action_mask = torch.as_tensor(masks).to(device, non_blocking=True)
                with torch.inference_mode():
                    actions[:] = algorithm.act(observations, action_mask=action_mask).cpu().numpy()
            for random_env_index in np.flatnonzero(explore):
                actions[random_env_index] = [
                    rng.choice(np.flatnonzero(mask)) for mask in masks[random_env_index]
                ]

            next_steps = vector_env.step(actions)
            for env_index, next_step in enumerate(next_steps):
                trainer.record(Transition(current[env_index], actions[env_index], next_step))
                individual_returns[env_index] += next_step.rewards
                episode_cost[env_index] += next_step.info["community_energy_cost"]
            current = next_steps

            if trainer.ready:
                metrics = trainer.update()

        loss_text = f"{metrics['loss']:.3f}" if metrics is not None else "no_update"
        print(
            f"round={episode + 1} episodes={args.num_envs} steps={vector_env.spec.horizon} "
            f"mean_individual_returns={individual_returns.mean(axis=0).round(2).tolist()} "
            f"mean_community_cost={episode_cost.mean():.2f} loss={loss_text}"
        )

    if args.checkpoint is not None:
        trainer.save_checkpoint(args.checkpoint)
        print(f"checkpoint={args.checkpoint} updates={trainer.optimization.update_count}")


if __name__ == "__main__":
    main()
