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
| MAAC | loss、critic_loss、actor_loss、entropy_ratio、actor/critic_gradient_norm |
| MAPPO | loss、policy/value_loss、entropy_ratio、approx_kl、clip_fraction、explained_variance、actor/critic_gradient_norm |
| MADDPG | loss、critic_loss、actor_loss、actor/critic_gradient_norm |
| MASAC | loss、critic_loss、actor_loss、alpha、actor/critic/temperature_gradient_norm |
| QMIX | expected_incompatible 原因 |

报告不会跨动作空间选“总冠军”。离散组只比较 MAAC 与 MAPPO，连续组只比较
MADDPG 与 MASAC；每组分别给出 adversary 与 good-team 的跨 seed 均值、标准差、
95% 置信区间和排名。每次训练在 step 0 先评估未训练策略，之后按固定环境步间隔评估。

同动作空间 cross-play 将算法/seed A 的 adversary actor 与算法/seed B 的两个
good-agent actors 组合，在统一 seed 上评估；报告保留 adversary 和 good-team 两张矩阵，
不把冲突双方的回报合成单一分数。

## 运行

论文级对比使用每算法 5 个随机种子、每 seed 100,000 环境步和固定评估场景：

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark_noncooperative.py `
  --device auto --seeds 7 17 29 41 53 --environment-steps 100000 `
  --num-envs 4 --horizon 25 --eval-episodes 10 --eval-interval-steps 10000 `
  --output benchmark-results\mpe2_simple_adversary_full.json
.\.venv\Scripts\python.exe benchmarks\plot_noncooperative.py `
  benchmark-results\mpe2_simple_adversary_full.json `
  --output-dir benchmark-results\plots
```

100,000 必须能被 `num_envs*horizon` 整除，脚本会拒绝不精确的预算，避免算法间
交互量不同。评估与 cross-play 使用统一固定 seed，避免曲线差异来自评估关卡随机性。
JSON 同目录还会生成 CSV 摘要和 `<report>_artifacts/checkpoints/` 下的最终 checkpoint；
每个 seed 记录配置快照、MPE2 版本、CUDA/GPU、update/optimizer-step 数与 SHA-256。

绘图命令生成十三张图片：

- `learning_curves_{discrete,continuous}.png`：含 step 0 的分组学习曲线；
- `final_returns_{discrete,continuous}.png`：分动作空间、分角色终局回报与 95% CI；
- `return_distributions_{discrete,continuous}.png`：分组终局回报分布；
- `loss_stability_{discrete,continuous}.png`：区间平均 loss；
- `gradient_entropy_stability_{discrete,continuous}.png`：actor/critic 梯度、裁剪相关指标、
  离散熵比例、PPO 诊断与 SAC alpha；
- `cross_play_{discrete,continuous}.png`：对手与协作方分开的 cross-play 热力图；
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
