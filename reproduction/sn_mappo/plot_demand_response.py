"""绘制真实诊断和独立评估；散点评估，滚动均值仅用于训练诊断趋势。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def plot_run(root: Path) -> None:
    """
    读取一个实验目录的 training/evaluation
    JSONL，按实际物理步绘图并保存
    diagnostics.png；评估仅散点，训练趋势为尾随均值。
    """
    import matplotlib  # type: ignore[import-untyped]
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]

    rows = [json.loads(line) for line in (root/'training.jsonl').read_text().splitlines()]
    evaluation = [json.loads(line) for line in (root/'evaluation.jsonl').read_text().splitlines()]
    x = np.array([row['physical_steps'] for row in rows])
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    val = [r for r in evaluation if r['split'] == 'validation']
    for agent in range(len(val[0]['returns'])):
        axes[0, 0].scatter([r['physical_steps'] for r in val], [r['returns'][agent] for r in val],
                           label=f'Agent {agent}', s=20)
    axes[0, 0].set_title('Independent validation return (observed points)')
    axes[0, 0].legend(fontsize=7)
    keys = list(rows[0])
    targets = [
        (axes[0, 1], 'policy_loss', 'Actor policy loss'),
        (axes[0, 2], 'value_loss', 'Critic value loss'),
        (axes[1, 0], 'approx_kl', 'Approximate KL'),
        (axes[1, 1], 'entropy', 'Policy entropy'),
        (axes[1, 2], 'response_residual' if any('response_residual' in k for k in keys)
         else 'actor_gradient_norm', 'Solve residual / actor gradient norm'),
    ]
    for axis, suffix, title in targets:
        for key in keys:
            if key == suffix or key.endswith('/'+suffix):
                y = np.array([r[key] for r in rows])
                line, = axis.plot(x, y, alpha=.25, linewidth=.7)
                window = min(20, len(rows))
                smooth = np.convolve(y, np.ones(window)/window, mode='valid')
                axis.plot(x[window-1:], smooth, label=key, color=line.get_color(), linewidth=1.4)
        if suffix == 'response_residual':
            axis.axhline(1e-4, color='red', linestyle='--', label='Acceptance 1e-4')
            axis.set_yscale('log')
        axis.set_title(title)
        axis.legend(fontsize=6)
    for axis in axes.flat:
        axis.set_xlabel('Actual physical training steps')
        axis.grid(alpha=.2)
    fig.suptitle(root.name+' | faint: raw updates; solid: trailing mean (up to 20 updates)')
    fig.savefig(root/'diagnostics.png', dpi=160)
    plt.close(fig)


def main() -> None:
    """解析脚本命令行并执行文档所述入口；输出写入显式指定的本地目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', default='runs/sn_mappo')
    args = parser.parse_args()
    for path in Path(args.root).rglob('training.jsonl'):
        if (path.parent/'evaluation.jsonl').exists():
            plot_run(path.parent)
            print(path.parent/'diagnostics.png')


if __name__ == '__main__':
    main()
