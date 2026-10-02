"""真实 OPSD 日实验：固定 UC 或同步连续 MAPPO，原始诊断与评估分开保存。"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from marl.algorithms import MAPPO, MAPPOConfig
from marl.config import config_to_dict, load_algorithm_config
from marl.experiment import build_experiment
from marl.training import OnPolicyTrainer
from reproduction.sn_mappo import DailyDemandAdapter, DemandProfiles, DemandResponseConfig


def evaluate(algorithm: MAPPO, environment: DailyDemandAdapter, seed: int) -> dict:
    """独立确定性日评估；循环状态持续传递，并恢复模型 train/eval 标志。"""
    modes = {module: module.training for module in algorithm.modules()}
    algorithm.eval()
    policy_state, value_state = None, None
    step = environment.reset(seed)
    rewards = np.zeros(environment.spec.num_agents)
    pg, der_curtailment, residual, daily_debt = [], 0.0, 0.0, 0.0
    try:
        while not step.done:
            with torch.inference_mode():
                out = algorithm.sample(
                    torch.tensor(step.observations).unsqueeze(0),
                    torch.tensor(step.state).unsqueeze(0), deterministic=True,
                    policy_state=policy_state, value_state=value_state,
                )
            policy_state, value_state = out.policy_state, out.value_state
            step = environment.step(out.actions[0].cpu().numpy())
            rewards += step.rewards
            pg.append(step.info['flows']['pb'])
            der_curtailment += step.info['flows']['pa']*.25
            residual = max(residual, step.info['projection_residual'])
            if environment.spec.horizon == 96:
                daily_debt = max(abs(x) for x in step.info['debt_kwh'])
    finally:
        for module, mode in modes.items():
            module.training = mode
    power = np.asarray(pg)
    return {'returns': rewards.tolist(), 'grid_kwh': float(power.sum()*.25),
            'grid_peak_kw': float(power.max()),
            'grid_par': float(power.max()/power.mean()) if power.mean() else 0.0,
            'grid_tv_kw': float(np.abs(np.diff(power)).sum()),
            'curtailment_kwh': float(der_curtailment), 'projection_residual': residual,
            'terminal_debt_kwh': daily_debt}


def run(args: argparse.Namespace, seed: int) -> dict:
    """
    以指定 seed 构造实验、执行预算内训练并保存逐次诊断、独立评估、配置和
    checkpoint；不覆盖已有实验。
    """
    def profiles(split: str) -> DemandProfiles:
        """读取指定 train/validation/test 分割，选择消费者列并保留 DER 插补标记和训练尺度。"""
        p = DemandProfiles.load(Path(args.data)/f'{split}.npz')
        return replace(p, load_kw=p.load_kw[:, :args.consumers],
                       load_scale=p.load_scale[:args.consumers],
                       imputed=np.column_stack([p.imputed[:, :args.consumers], p.imputed[:, -1]]))

    fixed = (0.0, 0.0, 1.0, 0.0) if args.mode == 'fixed' else None
    physical = DemandResponseConfig()
    train = DailyDemandAdapter(profiles('train'), physical, fixed)
    validation = DailyDemandAdapter(profiles('validation'), physical, fixed)
    test = DailyDemandAdapter(profiles('test'), physical, fixed)
    config = load_algorithm_config(args.algorithm)
    if not isinstance(config, MAPPOConfig):
        raise ValueError('此入口要求 MAPPO 配置')
    experiment = build_experiment(train, config, seed=seed)
    trainer = experiment.trainer
    assert isinstance(trainer, OnPolicyTrainer)
    root = Path(args.output)/f'{args.mode}_{args.consumers}c_seed{seed}'
    root.mkdir(parents=True, exist_ok=True)
    if (root/'training.jsonl').exists():
        raise FileExistsError(f'拒绝覆盖已有实验：{root}')
    rng = np.random.default_rng(seed)
    (root/'manifest.json').write_text(json.dumps({
        'seed': seed, 'algorithm': config_to_dict(config), 'environment': asdict(physical),
        'source_sha256': train.profiles.source_sha256, 'mode': args.mode,
        'training_days': train.starts, 'target_physical_steps': args.steps,
        'evaluation_interval': args.eval_every, 'consumers': args.consumers,
    }, indent=2), encoding='utf-8')
    start, physical_steps, next_evaluation = time.perf_counter(), 0, 0
    with (root/'training.jsonl').open('w', encoding='utf-8') as diagnostics:
        with (root/'evaluation.jsonl').open('w', encoding='utf-8') as evaluation:
            while physical_steps < args.steps:
                if physical_steps >= next_evaluation:
                    row = {'physical_steps': physical_steps, 'split': 'validation',
                           **evaluate(experiment.algorithm, validation, seed=191)}
                    evaluation.write(json.dumps(row)+'\n')
                    evaluation.flush()
                    next_evaluation = physical_steps + args.eval_every
                metrics = trainer.train_rollout([int(rng.integers(0, 2**31-1))])
                physical_steps += train.spec.horizon
                diagnostics.write(json.dumps({'physical_steps': physical_steps, **metrics})+'\n')
                diagnostics.flush()
            final = {'physical_steps': physical_steps, 'split': 'test',
                     **evaluate(experiment.algorithm, test, seed=191)}
            evaluation.write(json.dumps(final)+'\n')
    trainer.save_checkpoint(root/'checkpoint.pt')
    (root/'run_state.json').write_text(json.dumps({
        'physical_steps': physical_steps, 'day_rng_state': rng.bit_generator.state,
    }, indent=2), encoding='utf-8')
    final['seconds'] = time.perf_counter()-start
    (root/'summary.json').write_text(json.dumps(final, indent=2), encoding='utf-8')
    print(json.dumps({'run': str(root), **final}), flush=True)
    return final


def main() -> None:
    """解析脚本命令行并执行文档所述入口；输出写入显式指定的本地目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='runs/sn_mappo/data')
    parser.add_argument('--output', default='runs/sn_mappo/fixed_uc')
    parser.add_argument('--algorithm',
                        default=Path(__file__).parent/'configs/mappo_continuous.yaml')
    parser.add_argument('--mode', choices=['fixed', 'synchronous'], default='fixed')
    parser.add_argument('--consumers', type=int, choices=[1, 3], default=3)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--steps', type=int, default=20_000)
    parser.add_argument('--eval-every', type=int, default=5000)
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1:
        parser.error('steps/eval-every 必须为正')
    torch.set_num_threads(1)
    for seed in args.seeds:
        run(args, seed)


if __name__ == '__main__':
    main()
