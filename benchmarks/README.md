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
  --num-envs 4 --horizon 25 --eval-episodes 10 `
  --output benchmark-results\mpe2_simple_adversary_full.json
.\.venv\Scripts\python.exe benchmarks\plot_noncooperative.py `
  benchmark-results\mpe2_simple_adversary_full.json `
  --output-dir benchmark-results\plots
```

100,000 必须能被 `num_envs*horizon` 整除，脚本会拒绝不精确的预算，避免算法间
交互量不同。评估与 cross-play 使用统一固定 seed，避免曲线差异来自评估关卡随机性。
JSON 同目录还会生成 CSV 摘要和 `<report>_artifacts/checkpoints/` 下的最终 checkpoint；
每个 seed 记录配置快照、MPE2 版本、CUDA/GPU、update/optimizer-step 数与 SHA-256。

绘图命令生成二十一张图片（两个动作组齐全时），以及中文读图说明和数据覆盖清单：

- `learning_curves_{discrete,continuous}.png`：含 step 0 的分组学习曲线；
- `final_returns_{discrete,continuous}.png`：分动作空间、分角色终局回报与 95% CI；
- `return_distributions_{discrete,continuous}.png`：分组终局回报分布；
- `loss_stability_{discrete,continuous}.png`：区间平均 loss；
- `gradient_entropy_stability_{discrete,continuous}.png`：actor/critic 梯度、裁剪相关指标、
  离散熵比例、PPO 诊断与 SAC alpha；
- `cross_play_{discrete,continuous}.png`：对手与协作方分开的 cross-play 热力图；
- `training_efficiency.png`：相同交互预算下的耗时和环境吞吐率。
- `return_comparison_{discrete,continuous}.png`：按角色叠加不同算法的学习曲线，
  同时显示每个 seed，避免在分离的子图中难以比较。
- `convergence_diagnostics_{discrete,continuous}.png`：最近 5 个真实评估点的回报
  斜率（每 1000 步变化）与时间标准差；仅观察趋势，不能证明学习成功或纳什均衡。
- `optimizer_losses_{discrete,continuous}.png`：actor/policy 与 critic/value 损失分开，
  使用 symlog 坐标显示负 loss 与数量级变化，不跨算法比较 loss 大小。
- `progress_timing_{discrete,continuous}.png`：各 seed 的逐区间墙钟吞吐，帮助识别
  何时变慢；包含区间内评估开销，不冒充纯 optimizer 性能。
- `README.md` 与 `plot_audit.json`：读图方法、每个 seed 的实际采样点数、最大评估
  间隔和旧报告缺少的数据。图表不插值补造点，缺失指标不补零。

### 审计采样与收敛诊断

训练指标与策略评估分别采样：`--metrics-interval-steps` 默认一个完整 rollout
（4 环境 × 25 步 = 100 环境步）；`--eval-interval-steps` 默认按总预算划分约 100 个
区间并向上对齐完整 rollout。两参数都可显式设置，必须是 `num_envs*horizon` 的倍数。
评估边界与训练终点也会 flush 诊断，因此不会漏掉最后不足一个区间的数据。

| 训练预算 | 默认训练诊断点 | 默认策略评估点（含 step 0） |
|---|---:|---:|
| 10,000 步 | 100 | 101 |
| 100,000 步 | 1,000 | 101 |

`training_history` 保存独立的训练指标序列、环境步、update/optimizer-step 计数和时间。
评估区间指标按实际 algorithm update 次数加权，避免 warm-up 和末尾短区间被等权处理。
增加训练诊断不执行额外模型前向；更频繁的策略评估会增加运行时间，需结合吞吐图审查。

有 artifact 目录时，每个 seed 的训练/评估事件逐点写入
`<report>_artifacts/diagnostics/<algorithm>_seed_<seed>.jsonl`，异常退出后仍可审计已完成区间。
该 JSONL 是诊断证据，不是可以恢复 optimizer 的 checkpoint。

旧 100k 报告只有 11 个评估点，无法恢复真实的密集历史；新绘图程序会如实显示其稀疏性。
不将曲线插值、平滑或重复评估最终 checkpoint 当作历史训练数据。
少于 5 个评估点时，滚动图明确显示数据不足。斜率趋近零也可能是策略停滞，必须结合
原始回报、逐 seed 波动、critic 误差与固定对手测试解释。

回报阴影使用跨训练 seed 的 Student-t 95% 区间，小样本下不作为显著性证明。
旧报告的 `gradient_clip_rate` 不能恢复；新版从实际 optimizer-step 事件计数，
区间合并时保留计数与峰值，不用平均梯度反推裁剪率。

## 稳定性改造与审计

- `--layer-norm`：策略与 critic 隐藏层启用 LayerNorm；默认关闭。
- `--normalize-targets`：逐智能体 Welford target 统计，同时缩放预测和标签用于
  critic 误差；最小标准差 1。网络公开输出、bootstrap、actor Q 仍为原始单位。
  这不是 PopArt，不能据 normalized loss 变小宣称稳定。
- MAPPO 每份 fresh rollout 更新一次统计，所有 epoch 固定；MASAC 两个 critic
  每次更新共用一次统计。统计随模型保存，诊断 loss 默认不修改统计。
- tanh-Gaussian 使用 PyTorch 稳定 Jacobian；entropy 表示随机样本的
  `-log_prob`，与原始高斯解析熵区分。确定性动作的 surprisal 不代表熵。
- 新报告拆分执行完整性、数值有限性、优化警报和学习评估；保留旧
  `numerically_stable` 字段仅作有限值兼容，不代表价值不漂移或策略收敛。
- 逐智能体记录 V/Q、target、TD error 的均值/峰值/非有限事件，以及 RMSE、
  bias、explained variance。高裁剪率或低离散熵持续三个诊断区间触发警报。
  价值绝对值 10,000 为本基准经验警报，阈值保存在 settings，不是理论边界。
  非有限诊断在最近日志边界中止该运行；不是逐 GPU kernel 同步检查。
- 固定随机对手和固定未训练对手分别替换一侧；共同场景种子与训练分离，
  同时输出确定性/随机评估。评估使用独立 RNG 上下文，不消耗训练 RNG。
- 新版完整报告额外生成逐智能体尺度、真实裁剪率和固定对手学习曲线，
  四算法完整报告共 **29 张图**。旧报告按已有字段生成，不补造历史。
- trainer/optimizer checkpoint schema 升级到 3；旧 schema 不静默续训，
  可显式加载 algorithm 权重作诊断。关闭稳定化开关时原网络权重结构不变。

本轮 2k 步 × seeds 7/17 × 四算法 × 四候选（32 runs）全部运行完整、数值有限。
基准、仅 LayerNorm、仅 target 标准化、组合四组分别保持同一学习率、batch 和 UTD。
组合组未触发当前短程警报，选为 10k×3 候选，不修改库默认配置：
MAAC 原始 value/target 峰值从约 37 降至 17，MADDPG 从约 51 降至 24；
MAPPO 两 seed 的持续高裁剪警报在组合组中未出现。2k 太短，不能证明长期漂移已消除。

异常旧 checkpoint 的 critic-only 干预见 `audit_value_drift.py`。
固定奖励标签可降低大 Q，说明并非完全无法拟合有限监督；移动 bootstrap 下 MAAC
100 次局部更新的 Q 峰值约 176 万升至 455 万，同时 TD loss 下降。
这直接说明 TD loss 下降不保证原始价值稳定，但只是一批数据的局部干预，
不能排除状态外推、策略变化和超参数的共同影响。

快速检查可缩短为：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[benchmark]"
.\.venv\Scripts\python.exe benchmarks\benchmark_noncooperative.py --device auto
```

默认使用两个 seed、每个算法 8 个短训练 episode，并写入
`benchmark-results/mpe2_simple_adversary.json`。该目录是运行产物，不进入 Git。
可用 `--algorithms`、`--seeds`、`--train-episodes`、`--eval-episodes`、
`--horizon` 和 `--hidden-dim` 调整规模。

## 运行路径性能验收

`benchmark_runtime.py` 在同一进程中比较正确性修复后的逐环境基线与批量 collector，
以及 MAPPO 逐 minibatch 搬运与整批 rollout 搬运：

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark_runtime.py `
  --device cuda --iterations 10 --num-envs 4 --horizon 25 `
  --mappo-repeats 5 --enforce `
  --output benchmark-results\runtime_acceptance.json
```

`--enforce` 要求离策略采集吞吐至少提升 20%、MAPPO 吞吐不低于旧路径的 95%、
峰值显存增长不超过 25%，并检查 MAPPO 更新后参数在 `1e-5` 容差内等价。
2026-09-26 在 RTX 4060 Laptop GPU / torch 2.14.0+cu130 上的本地验收为：

- 离策略采集 `2.67×`，峰值显存 `1.0001×`；
- MAPPO 数据路径 `1.97×`，峰值显存 `1.058×`，最大参数差 `4.33e-7`。

这些数值是本机结果，不作为其他机器的固定性能承诺。当前系统找不到 MSVC `cl.exe`，
因此没有启用 `torch.compile`；BF16 硬件可用但仍保持默认关闭，避免在未完成算法级
精度对照前改变数值容差。
