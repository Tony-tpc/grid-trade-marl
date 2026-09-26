# 非合作博弈基准

## 环境选择审计

Gymnasium 定义单智能体环境接口，本身不提供统一的多智能体博弈 API。Farama 官方将
Gymnasium 风格的多智能体环境放在 PettingZoo；原 Multi-Agent Particle Environment
当前由独立的 MPE2 包维护。因此本基准使用 MPE2 的
[simple_adversary_v3](https://mpe2.farama.org/environments/simple_adversary/)，
而不是把单智能体 Gym 环境伪装成博弈。

官方资料：

- [PettingZoo Parallel API](https://pettingzoo.farama.org/api/parallel/)：同步提交所有
  存活智能体动作，返回逐智能体观测、奖励、termination 与 truncation。
- [MPE2 环境总览](https://mpe2.farama.org/mpe2/)：明确列出 adversarial 场景。
- [simple_adversary_v3](https://mpe2.farama.org/environments/simple_adversary/)：
  一个 adversary 与多个 good agents，支持相同场景的离散/连续动作。

### 已满足

| 论文/工程要求 | MPE2 与仓库对应 |
|---|---|
| 问题定义 | adversary 与 good-agent 团队目标冲突，属于一般和博弈 |
| 智能体 | 固定数量；默认 1 个 adversary、2 个 good agents |
| 动作 | 离散 Discrete(5) 或连续 Box(0,1,[5]) |
| 奖励 | 逐智能体 individual reward，不跨阵营平均 |
| 数据 | MAPPO fresh rollout；其余适用算法使用 replay |
| 终止 | max_cycles 作为 truncation，保留 bootstrap |
| state | 官方全局 state 直接供集中式 critic |

### 可复用但需配置

- MAAC、MAPPO 使用离散版本；MADDPG、MASAC 使用连续版本。
- adversary 的 8 维观测在 adapter 中右侧补零到 10 维，以满足当前同构网络接口。
- 连续策略的标准输出为 [-1,1]，adapter 映射到环境要求的 [0,1]。
- 智能体数、观测维、动作维和 state 维仍全部从 EnvironmentSpec 注入。

### 缺失或有偏差

- QMIX 数学上要求共享团队奖励，不能用于该 individual-reward 非合作场景。基准验证
  它明确拒绝环境，并标记为 expected_incompatible；这不是训练失败。
- 当前固定形状训练链不支持 episode 中部分智能体提前退出。adapter 对此显式报错。
- 短基准不证明策略达到纳什均衡、收敛或优于 baseline，只检查数值与接口稳定性。

## 稳定性与输出完整性

每个适用算法、每个 seed 必须同时满足以下条件，才记为 numerically_stable：

1. 至少完成一次参数更新；
2. 所有 loss、诊断指标和梯度范数均为有限值；
3. 更新后的全部模型参数均为有限值；
4. 固定策略评估输出每个智能体的有限 episode return。

output_complete 还要求每个算法提供其应有指标：

| 算法 | 必需指标 |
|---|---|
| MAAC | loss、critic_loss、actor_loss、entropy、gradient_norm |
| MAPPO | loss、policy_loss、value_loss、entropy、approx_kl、clip_fraction、gradient_norm、explained_variance |
| MADDPG | loss、critic_loss、actor_loss、gradient_norm |
| MASAC | loss、critic_loss、actor_loss、alpha_loss、alpha、gradient_norm |
| QMIX | expected_incompatible 原因 |

## 运行

完整对比使用相同环境交互预算、3 个随机种子和固定评估场景：

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark_noncooperative.py `
  --device auto --seeds 7 17 29 --train-episodes 100 `
  --num-envs 1 --eval-episodes 10 --eval-interval 10 `
  --output benchmark-results\mpe2_simple_adversary_full.json
.\.venv\Scripts\python.exe benchmarks\plot_noncooperative.py `
  benchmark-results\mpe2_simple_adversary_full.json `
  --output-dir benchmark-results\plots
```

这里每个算法、每个随机种子都使用 2,500 个环境 step。评估使用同一组固定 seed，
避免曲线差异来自评估关卡随机性。

绘图命令生成五张图片：

- `learning_curves.png`：对手和协作方的跨 seed 学习曲线与 95% 置信区间；
- `final_returns.png`：各算法最终策略的分角色平均回报；
- `return_distributions.png`：最终策略在评估 episode 上的回报分布；
- `loss_stability.png`：各 seed 的区间平均 loss 与总体趋势；
- `training_efficiency.png`：相同交互预算下的耗时和环境吞吐率。

快速检查可缩短为：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[benchmark]"
.\.venv\Scripts\python.exe benchmarks\benchmark_noncooperative.py --device auto
```

默认使用两个 seed、每个算法 8 个短训练 episode，并写入
`benchmark-results/mpe2_simple_adversary.json`。该目录是运行产物，不进入 Git。
可用 `--algorithms`、`--seeds`、`--train-episodes`、`--eval-episodes`、
`--horizon` 和 `--hidden-dim` 调整规模。
