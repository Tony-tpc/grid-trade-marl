"""个体收益 P2P 博弈：环境 -> 适配器 -> MARLBatch -> MAAC。

运行：.venv\\Scripts\\python.exe examples\\train_energy_maac.py --episodes 5
使用合成曲线与 54 个量化动作，仅演示训练接口，不代表论文数值复现。
"""

from __future__ import annotations

import argparse
from collections import deque

import numpy as np
import torch

from marl.algorithms import MAAC
from marl.envs import (
    ActionKind,
    EnergyTradingAdapter,
    EnergyTradingConfig,
    EnergyTradingEnv,
    Transition,
    algorithm_config_from_env,
    transitions_to_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="个体收益电能交易博弈中的 MAAC 训练示例")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.episodes < 1 or args.agents < 1 or args.batch_size < 1:
        parser.error("episodes、agents、batch-size 必须大于零")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    adapter = EnergyTradingAdapter(
        EnergyTradingEnv(EnergyTradingConfig(num_agents=args.agents)), ActionKind.DISCRETE
    )
    # 环境适配器决定 N/O/A；MAAC 本身不关心电力市场或设备的内部实现。
    algorithm = MAAC(algorithm_config_from_env("maac", adapter.spec, hidden_dim=64))
    optimizer = torch.optim.Adam(algorithm.parameters(), lr=3e-4)
    replay: deque[Transition] = deque(maxlen=5000)

    for episode in range(args.episodes):
        current = adapter.reset(seed=args.seed + episode)
        individual_returns = np.zeros(adapter.spec.num_agents)
        episode_cost = 0.0
        epsilon = max(0.1, 0.8 - 0.7 * episode / max(args.episodes - 1, 1))
        while not current.done:
            assert current.action_mask is not None
            if rng.random() < epsilon:
                actions = np.asarray(
                    [rng.choice(np.flatnonzero(mask)) for mask in current.action_mask],
                    dtype=np.int64,
                )
            else:
                observations = torch.as_tensor(current.observations).unsqueeze(0)
                mask = torch.as_tensor(current.action_mask).unsqueeze(0)
                with torch.no_grad():
                    actions = algorithm.act(observations, action_mask=mask)[0].cpu().numpy()

            next_step = adapter.step(actions)
            replay.append(Transition(current, actions, next_step))
            individual_returns += next_step.rewards
            episode_cost += float(next_step.info["community_energy_cost"])
            current = next_step

            if len(replay) >= args.batch_size:
                indices = rng.choice(len(replay), size=args.batch_size, replace=False)
                batch = transitions_to_batch([replay[int(i)] for i in indices])
                metrics = algorithm.optimize(batch, optimizer)

        loss_text = f"{metrics['loss']:.3f}" if len(replay) >= args.batch_size else "no_update"
        print(
            f"episode={episode + 1} steps={adapter.spec.horizon} "
            f"individual_returns={individual_returns.round(2).tolist()} "
            f"community_cost={episode_cost:.2f} loss={loss_text}"
        )


if __name__ == "__main__":
    main()
