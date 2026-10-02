"""顺序 1+3 实验：默认仅一轮数值验收；求解不达标时记录并停止长训练。"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from marl.algorithms import SNMAPPO, SNMAPPOConfig
from marl.config import config_to_dict, load_algorithm_config
from marl.core.recurrent import RecurrentState
from marl.envs.sequential import SequentialEnvironment
from marl.experiment import build_experiment
from reproduction.sn_mappo import DailyDemandAdapter, DemandProfiles, DemandResponseEnv


def evaluate(algorithm: SNMAPPO, environment: SequentialEnvironment, seed: int) -> dict:
    """确定性完整事件评估；保持三个角色 GRU 状态，恢复所有模块 train/eval 标志。"""
    modes = {m: m.training for m in algorithm.modules()}
    algorithm.eval()
    models = (algorithm.leader, algorithm.coordinator, algorithm.followers)
    ps: list[RecurrentState] = [None, None, None]
    vs: list[RecurrentState] = [None, None, None]
    step = environment.reset(seed)
    rewards = np.zeros(environment.spec.num_agents)
    grid, residual, curtailment = [], 0.0, 0.0
    try:
        while not step.done:
            with torch.inference_mode():
                state = torch.tensor(step.state).unsqueeze(0)
                leader = models[0].sample(torch.tensor(step.observations[:1]).unsqueeze(0),
                                           state, deterministic=True,
                                           policy_state=ps[0], value_state=vs[0])
                committed = environment.commit_leader(leader.actions[0, 0].numpy())
                state = torch.tensor(committed.state).unsqueeze(0)
                public = torch.tensor(committed.observations[1:]).unsqueeze(0)
                plan = models[1].sample(public.mean(-2, keepdim=True), state, deterministic=True,
                                        policy_state=ps[1], value_state=vs[1])
                obs = torch.cat([public, plan.actions.expand(-1, public.shape[1], -1)], -1)
                follower = models[2].sample(obs, state, deterministic=True,
                                            policy_state=ps[2], value_state=vs[2])
            for i, out in enumerate((leader, plan, follower)):
                ps[i], vs[i] = out.policy_state, out.value_state
            step = environment.settle(follower.actions[0].numpy())
            rewards += step.rewards
            grid.append(step.info['flows']['pb'])
            residual = max(residual, step.info['projection_residual'])
            curtailment += step.info['flows']['pa']*.25
    finally:
        for module, mode in modes.items():
            module.training = mode
    power = np.asarray(grid)
    return {'steps': len(grid), 'returns': rewards.tolist(),
            'grid_kwh': float(power.sum()*.25), 'grid_tv_kw': float(np.abs(np.diff(power)).sum()),
            'grid_par': float(power.max()/power.mean()) if power.mean() else 0,
            'curtailment_kwh': float(curtailment), 'projection_residual': residual,
            'terminal_energy_kwh': step.info['energy_kwh'],
            'terminal_debt_kwh': max(abs(x) for x in step.info['debt_kwh'])}


def run(args: argparse.Namespace, seed: int) -> None:
    """
    以指定 seed 构造实验、执行预算内训练并保存逐次诊断、独立评估、配置和
    checkpoint；不覆盖已有实验。
    """
    config = load_algorithm_config(args.algorithm)
    if not isinstance(config, SNMAPPOConfig):
        raise ValueError('需要 SNMAPPOConfig')
    config = replace(config, implicit=not args.first_order)
    if args.damping is not None:
        config = replace(config, response=replace(config.response, damping=args.damping))
    data = Path(args.data)
    train = DailyDemandAdapter(DemandProfiles.load(data/'train.npz'))
    validation = DailyDemandAdapter(DemandProfiles.load(data/'validation.npz'))
    test = DemandResponseEnv(DemandProfiles.load(data/'test.npz'))
    experiment = build_experiment(train, config, seed=seed)
    label = 'first_order' if args.first_order else f'implicit_mu{config.response.damping:g}'
    root = Path(args.output)/f'{label}_seed{seed}'
    root.mkdir(parents=True, exist_ok=True)
    if (root/'training.jsonl').exists():
        raise FileExistsError(f'拒绝覆盖已有实验：{root}')
    (root/'manifest.json').write_text(json.dumps({
        'seed': seed, 'config': config_to_dict(config), 'environment': train.checkpoint_context,
        'target_steps': args.steps, 'evaluation_interval': args.eval_every,
        'acceptance': 'Each implicit solve must pass actual residual <= configured tolerance.',
    }, indent=2), encoding='utf-8')
    start, next_evaluation, accepted = time.perf_counter(), 0, True
    with (root/'training.jsonl').open('w', encoding='utf-8') as log:
        with (root/'evaluation.jsonl').open('w', encoding='utf-8') as evaluation:
            while experiment.trainer.physical_steps < args.steps:
                if experiment.trainer.physical_steps >= next_evaluation:
                    evaluation.write(json.dumps({
                        'physical_steps': experiment.trainer.physical_steps, 'split': 'validation',
                        **evaluate(experiment.algorithm, validation, 191),
                    })+'\n')
                    evaluation.flush()
                    next_evaluation = experiment.trainer.physical_steps+args.eval_every
                metrics = experiment.trainer.train_cycle()
                log.write(json.dumps(metrics)+'\n')
                log.flush()
                if metrics['leader/response_success'] != 1:
                    accepted = False
                    break  # 真失败不继续烧长训练预算，也不把缺失修正换成一阶成功。
            month = evaluate(experiment.algorithm, test, 191)
            evaluation.write(json.dumps({'physical_steps': experiment.trainer.physical_steps,
                                         'split': 'test_month', **month})+'\n')
    experiment.trainer.save_checkpoint(root/'checkpoint.pt')
    result = {'seed': seed, 'physical_steps': experiment.trainer.physical_steps,
              'accepted': accepted, 'seconds': time.perf_counter()-start, 'month': month}
    (root/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'run': str(root), **result}), flush=True)


def main() -> None:
    """解析脚本命令行并执行文档所述入口；输出写入显式指定的本地目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', default=Path(__file__).parent/'configs/sn_mappo.yaml')
    parser.add_argument('--data', default='runs/sn_mappo/data')
    parser.add_argument('--output', default='runs/sn_mappo/sequential')
    parser.add_argument('--steps', type=int, default=864)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--eval-every', type=int, default=5000)
    parser.add_argument('--first-order', action='store_true')
    parser.add_argument('--damping', type=float)
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1:
        parser.error('steps/eval-every 必须为正')
    torch.set_num_threads(1)
    for seed in args.seeds:
        run(args, seed)


if __name__ == '__main__':
    main()
