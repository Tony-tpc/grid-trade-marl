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
  原始价值绝对值 10,000 为本基准经验安全阈值，保存在 settings，不是理论边界。
  超过该阈值或非有限诊断在最近日志边界中止该运行；不是逐 GPU kernel 同步检查。
  持续低熵/高裁剪率保留为警报，不自动修改算法或静默重启。
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

## 并行训练、恢复与更新调度

完整验证预算为四算法各 10,000 环境步 × seeds 7、17、29，QMIX 仍记录为预期
不兼容。不自动扩大到 100k×5。示例（默认 UTD=1，不通过减少更新冒充提速）：

```powershell
.\.venv\Scripts\python.exe -m benchmarks.benchmark_noncooperative `
  --algorithms maac mappo maddpg masac qmix --seeds 7 17 29 `
  --environment-steps 10000 --layer-norm --normalize-targets `
  --device cpu --torch-threads 1 --workers 4 `
  --output benchmark-results/revised_10k_x3.json
.\.venv\Scripts\python.exe -m benchmarks.validate_report benchmark-results/revised_10k_x3.json
.\.venv\Scripts\python.exe -m benchmarks.plot_noncooperative `
  benchmark-results/revised_10k_x3.json --output-dir benchmark-results/plots_revised_10k_x3
```

- `--workers` 默认 1，标准库 ProcessPoolExecutor + Windows spawn 按算法×seed
  分任务；多进程每个 Torch 线程固定 1。单张 CUDA 卡只允许 1 worker。
- worker 各自拥有 diagnostics JSONL、progress JSON、checkpoint 和失败 JSON。
  进度包含环境步、更新计数、墙钟 ETA、线程数和源码哈希；ETA 是当前任务的
  平均速度估计，不是剩余整个任务池的确定结束时间。
- cross-play 按需加载 checkpoint，不把所有完成模型长期留在显卡上。
- `--resume` 必须使用同一输出/产物目录及配置。完成任务验证源码哈希、配置、
  完整标记和 checkpoint SHA256 后才跳过；未完成任务从 resume checkpoint 恢复。
- resume checkpoint 在完整 rollout 的评估边界保存，跨过默认 1000 环境步间隔时
  保存，终点强制保存；不整除时顺延到下一评估边界。保存 trainer/replay/optimizer、
  Torch 与探索 RNG、episode、评估历史、训练诊断、累计时间和调度余数。
  不恢复 episode 中途环境内部状态。旧诊断日志归档为 before-resume 文件，
  当前日志从 checkpoint 中真实保存的历史重建，避免重复历史混入曲线。
- 旧最终 checkpoint 缺少运行层状态，不能冒充精确续训；只能显式诊断/评估。

`--learning-starts` 默认 replay batch_size；`--train-every-transitions` 和
`--gradient-steps` 默认均为 1。计数单位是联合 transition，不乘智能体数量。
三组调度消融分别使用：原 warm-up/batch32/UTD1、warm-up1024/batch32/UTD1、
warm-up1024/batch128/UTD0.25。最后一项改变了训练工作量，不能计作等价实现收益。

本机 500 步 × 四算法 × 两 seed 的 worker 校准，1/2/4 workers 整批墙钟分别为
44.55/47.88/39.15 秒（4 workers 约 1.14×），包含进程启动与汇总，不能推断为
线性四倍提速。10k×3 选择 4 workers、CPU 每进程 1 线程。
三组 2k×2 离策略调度消融每任务分别执行 1969/977/244 次算法更新，整批耗时
134.30/76.30/40.16 秒；均未触发短程健康警报，但工作量不同，不属于等价性能对照。
主验证保留原 warm-up、batch32 和 UTD1，没有自动采用少更新的配置。

### 2026-09-27 的 10k×3 验收

MAAC、MAPPO、MADDPG、MASAC × seeds 7/17/29 共 12 个任务完成；QMIX 预期不兼容。
总墙钟 1088.30 秒（含密集评估、保存和 cross-play），不是旧 100k×5 同预算速度对照。
每任务 100 个训练诊断点、101 个真实评估点、201 条 JSONL 事件和一个最终 checkpoint；
四算法原始 value/target 峰值分别约 60.78、40.24、100.61、63.36，数值全部有限，
当前阈值下没有健康警报。MAAC 等价值曲线仍在上升，**不能据此宣称长期稳定或收敛**。
离散/连续各有 6×6 双方收益 cross-play 矩阵，共生成 29 张图和中文阅读说明。

固定随机对手、确定性评估中，相比 step 0 的 adversary 平均回报增量分别为
21.08、23.26、10.88、9.34；good-team 分别为 17.43、17.79、2.81、3.61。
仅三个训练 seed，good-team 的配对增量 Student-t 95% 区间均跨零，连续算法还存在
单 seed 退步；不能只凭正的均值宣布普遍改善，也不跨动作空间排名。

本次每个评估点包含 10 种组合、各 4 局、每局 25 步，故每任务另有约 101,000 个
评估环境步（不计入 10,000 训练步预算，也不含最终 cross-play）。密集审计会增加时间。
产物保存在忽略目录 `benchmark-results/revised_10k_x3*` 和 `plots_revised_10k_x3/`。

## 端到端性能验收

2026-09-27 接口审计补充：MAPPO 同一 rollout 内普通训练指标改为按 mini-batch
样本数加权，梯度仍按真实更新次数统计。修正不改变训练参数更新；上述历史实验产物
没有被重写，不能把本次源码修正当成那些实验已重新执行。整批价值拟合请优先查看
`rollout_explained_variance*`，具体审计见 [代码说明](../docs/code_audit_2026-09-27.md)。

新增完整训练测量复用非合作基准，包含真实采集、replay、算法更新和诊断评估，
完整预热后至少重复 5 次，记录中位数、范围、实际 optimizer steps、显存和最终权重：

```powershell
.\.venv\Scripts\python.exe -m benchmarks.benchmark_runtime --full-training `
  --device cpu --threads 1 --steps 200 --repeats 5 `
  --output benchmark-results/performance/current_cpu.json
.\.venv\Scripts\python.exe -m benchmarks.benchmark_runtime --profile-algorithm masac `
  --device cpu --threads 1 --steps 200 --output benchmark-results/performance/masac_profile.json
```

`--reference` 可对照同配置、同更新数的基线报告，最终权重容差为
`atol=1e-5, rtol=1e-4`。CPU 1/2/4 线程和 CUDA 分开校准；不改变通用
`--device auto` 的含义。profile 同时保存标准 pstats 和函数级阶段摘要；
累计耗时有嵌套，不能相加；CUDA 下 cProfile 是主机调度时间，不是 kernel 用时。
GPU 遥测为可用时的结束快照，缺失标 unavailable，不用于事后猜测全部慢因。

保留的实现优化：MAAC 内置独立 actor 利用 grad=None 隔离动量，外部策略保留安全
回退；MASAC 温度只在 actor 更新后采样 log-prob，不再运行 critic；CUDA replay
使用私有 pinned staging 与复制完成事件，公开 sample 仍独立拥有结果；MAPPO
直接保留 CPU 环境张量和动作，避免无用的 CPU→GPU→CPU；设备指标批量累计。
没有引入新 UpdatePlan、第二套 trainer 更新循环或自定义 attention kernel。
foreach Adam 实验没有确认收益，已撤回；compile/BF16 未启用。

2026-09-27 最终配对测量以稳定性提交 `052faf0` 为旧实现，每侧独立常驻进程、完整
预热、200 个训练环境步，五次重复交替 old/new 顺序，使用相同种子和更新次数。
末次 seed 11 的四算法模型/统计在 CPU 和 CUDA 上均通过上述参数容差检查。
下面是旧/新耗时中位数之比（大于 1 表示更快）：

| 算法 | CPU 单线程 | CUDA | CUDA 峰值显存比（新/旧） |
|---|---:|---:|---:|
| MAAC | 0.989× | 1.112× | 0.9992× |
| MAPPO | 1.009× | 1.031× | 0.9985× |
| MADDPG | 0.952× | 1.043× | 0.9991× |
| MASAC | 1.078× | 1.004× | 0.9984× |

离策略三算法几何平均为 CPU **1.005×**、CUDA **1.052×**，**未达到 20% 提速目标**。
MAPPO 回退/显存门槛通过。原始结果保留五次耗时范围：例如 MASAC CUDA 的旧/新
耗时范围分别为 6.57～12.76 / 5.90～11.03 秒，波动明显，不把几个百分点包装成稳健收益。
等价实现收益、worker 总吞吐收益、减少更新的配置收益必须分开阅读。
短程 CPU cProfile 中 MAAC/MASAC 的公共 backward/裁剪/optimizer 路径累计约
2.73/3.85 秒，replay 采样约 0.044/0.042 秒；主要剩余成本仍是大量小批次网络更新。
这些带 profiler 开销的函数累计时间有嵌套，不能相加或当作生产吞吐。

## 历史 collector-only 结果（不代表完整训练提速）

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
