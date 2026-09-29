# 从零训练、调参、评估与续训

[首页](README.md) · [新环境](environments.md) · [模块与算法扩展](extensions.md) · [训练器](trainers.md)

<a id="start"></a>
## 安装与第一次运行

在仓库根目录打开 PowerShell。已有 `.venv` 时直接使用，不必重装。新机器需要
Python 3.10 或更高版本，按项目依赖安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch, marl; print(torch.__version__); print(torch.cuda.is_available())"
.\.venv\Scripts\python.exe examples\train_mappo.py --rounds 1 --num-envs 2 --device cpu
.\.venv\Scripts\python.exe examples\train_energy_maac.py --episodes 1 --num-envs 8 --device cpu
```

示例使用合成能源数据，仅验证训练链。MAAC 的 episodes 参数实际表示并行采集 round，
不是总 episode 数；总数约为 episodes × num-envs。`num-envs` 是同步策略批次，环境仍在
同一进程逐个 step，并非自动开同等数量 CPU 进程。`device=auto` 只按 CUDA 可用性选择，
不保证 GPU 更快；先 CPU 小规模跑通，再用相同真实环境步数比较端到端吞吐。

上述能源命令使用默认 MLP。需要验收 48 步历史 LSTM 时，使用独立的
[maac_energy_history.yaml](../examples/configs/algorithms/maac_energy_history.yaml)，
完整命令、预期 update 数量和 Python 单次更新示例见[网络教程](networks.md)。

不依赖能源任务的完整教学脚本：

```powershell
.\.venv\Scripts\python.exe doc\examples\train_walkthrough.py --algorithm mappo --rounds 2
.\.venv\Scripts\python.exe doc\examples\train_walkthrough.py --algorithm maac --rounds 2
.\.venv\Scripts\python.exe doc\examples\train_walkthrough.py --recurrent --custom-critic
```

### Python 中构造并训练

```python
from marl.algorithms import MAPPOConfig
from marl.config import load_algorithm_config
from marl.envs import build_energy_trading_adapter, load_environment_config
from marl.experiment import build_experiment

config = load_algorithm_config("examples/configs/algorithms/mappo.yaml")
assert isinstance(config, MAPPOConfig)
env = build_energy_trading_adapter(
    load_environment_config("examples/configs/environments/energy_trading.yaml")
)
experiment = build_experiment(env, config, device="cpu", seed=42)
metrics = experiment.trainer.train_rollout([42])
print(metrics)
```

需要并行采集时为每个环境单独创建 adapter，再封装 SyncVectorEnv；不能把同一可变
adapter 重复放 N 次。MAPPO 必须传可交互环境；离策略可只传 spec，再由外部 collector
record/record_many 供给 transition。完整离策略循环见 [教学脚本](examples/train_walkthrough.py)。

<a id="tuning"></a>
## 配置与调参流程

算法 YAML schema_version=2，环境配置独立。复制完整算法模板后修改，不要把 MAPPO
的 advantage/rollout 和 SAC 的 replay/target_update 混在同一份万能配置中。
完整默认值、字段范围和含义见 [配置全集](../examples/configs/README.md)。

```powershell
Copy-Item examples\configs\algorithms\mappo.yaml examples\configs\algorithms\mappo_trial.yaml
.\.venv\Scripts\python.exe examples\train_mappo.py --config examples\configs\algorithms\mappo_trial.yaml --rounds 2 --num-envs 2 --device cpu
```

以上复制/编辑是你开始新实验时执行，本说明书不会自动改写模板。参数也可用冻结 dataclass
的 replace 派生，不修改原配置：

```python
from dataclasses import replace
from marl.algorithms import MAPPOConfig
base = MAPPOConfig()
trial = replace(base, update=replace(base.update, actor_learning_rate=1e-4, epochs=2))
```

### 常用参数影响

下表是实验排查顺序，不是针对所有任务的最优参数承诺。一次只改变一组机制，保留基线，
固定训练预算、评估场景和种子集合。

| 参数位置 | 作用 | 如何观察 |
|---|---|---|
| update.actor_learning_rate / critic_learning_rate | 两类参数的单步速度 | KL、梯度裁剪率、原始 value 误差和回报 |
| update.learning_rate | 专用学习率为 None 时的回退 | 设置专用值后，不再随公共值改变 |
| update.max_grad_norm / 专用上限 | 限制梯度范数 | 持续高裁剪率需查目标尺度，不能只调大阈值 |
| loss.clip_ratio | PPO 策略比率裁剪 | approx_kl、clip_fraction、回报一起看 |
| update.epochs / mini_batch_size | 同一 rollout 复用次数、每次样本量 | 更新力度、方差、吞吐；不等于新增交互 |
| rollout.horizon / num-envs | 每份 fresh 数据 T×B | 必须适配 episode，早结束目前报错 |
| advantage.gamma / gae_lambda | 折扣和优势偏差/方差 | 回报时间尺度、value 标签与终止处理 |
| loss.entropy_coefficient | 离散策略探索强度 | 熵与合法动作数；过低可能早早确定化 |
| loss.value_coefficient / td_coefficient | 价值目标权重 | 日志 raw loss 与加权贡献有区别 |
| loss.normalize_targets | 逐智能体目标尺度 | 原单位 V/Q/target/RMSE；归一化 loss 下降不证明稳定 |
| replay.capacity / batch_size | 历史覆盖与每次抽样量 | ready、旧数据比例、内存占用 |
| value_target.gamma | TD bootstrap 折扣 | 真实终止屏蔽、截断保留 |
| target_update.tau 或 interval | 目标跟随速度 | 参数过快/过慢变化与 critic 漂移 |
| MASAC initial_alpha / target_entropy / temperature_learning_rate | 连续探索和温度自适应 | alpha、熵、动作饱和比例 |
| policy/critic.backbone | 编码后的表示容量；MAPPO 可使用跨步 GRU/LSTM | 层宽、耗时、梯度与记忆任务 |
| encoder.temporal / 环境 history_steps | 窗口时序网络 / 原始窗口长度 W | 历史排列、填充值、局部编码梯度 |
| update.sequence_length（仅 MAPPO） | 跨步循环 backbone 的连续反传片段长度 | padding 比例；仅有历史 encoder 时必须为 null |

专用 max_grad_norm=None 表示回退公共值，不是单独关闭该角色裁剪；公共也为 None 时才
关闭。soft tau 越小 target 越慢，hard interval 以算法 update_count 而非环境步计数。
MAPPO 当前 update 配置没有 AMP 开关；离策略 amp_dtype=bf16 要求支持 BF16 的 CUDA。

### 调参执行顺序

1. 固定 seed，小网络 CPU 跑一次采集/更新，核对奖励和形状。
2. 记录原始基线配置、依赖版本、Git 版本、真实环境步和 optimizer-step 数。
3. 先检查观测/奖励尺度、mask 和终止，再调整学习率、梯度上限、目标尺度。
4. 其次调整 PPO epoch/clip/entropy，或 replay 调度/target 更新，避免同时改所有项。
5. 输入已含历史窗口时配置 `encoder`；需要跨环境步记忆时使用 MAPPO 循环 `backbone`。
   两者可组合，但窗口 W 与 `update.sequence_length` 不同，见[网络教程](networks.md#time)。
6. 对候选配置使用多个训练种子和相同独立评估种子；最终测试集不参与反复调参。

<a id="metrics"></a>
## 指标怎么读

| 指标 | 意义 | 容易误读之处 |
|---|---|---|
| 各 agent episode return | 环境原始累计收益 | 不把对抗双方平均成胜率 |
| policy_loss / actor_loss | 优化策略的 surrogate | 允许为负，跨算法绝对值不可比较 |
| value_loss / critic_loss | 对 return/TD target 的误差 | 可能使用尺度化，不等于原单位误差 |
| entropy | 随机程度 | 连续 -log_prob 不是离散最大熵；需区分随机/确定性评估 |
| approx_kl / clip_fraction | PPO 策略变化幅度 | 不是“是否收敛”的单一指标 |
| explained_variance | 价值解释 target 方差的程度 | 低方差 target 下不稳定；看原始误差 |
| gradient_norm | 裁剪前范数 | 平均值不能反推每次裁剪事件 |
| *_gradient_clip_rate | 实际裁剪 step 比例 | 与学习率、目标尺度、阈值一起解释 |
| rollout_* value 指标 | 整份旧 rollout 的 value/return 诊断 | 与更新中 mini-batch 指标不是同一时刻 |
| update_count / optimizer_step_count | 算法轮次 / 实际优化次数 | 两者都不是环境步数 |

训练诊断和固定策略评估是两条时间序列。保存真实步数、原始点和逐 seed 曲线；回报、
actor loss、critic loss 分图或分轴。滚动平均只辅助看趋势，不替代原始数据，不补造旧历史。
曲线平、数值有限、学习有效与达到博弈均衡是不同结论。

## 基准、图表与运行目录

需要 MPE2 和绘图时安装项目可选依赖；小预算命令使用整除 T×B 的环境步数：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[benchmark]"
.\.venv\Scripts\python.exe benchmarks\benchmark_noncooperative.py --algorithms maac mappo --device cpu --seeds 7 17 29 --environment-steps 1000 --num-envs 2 --horizon 25 --eval-episodes 4 --eval-interval-steps 100 --metrics-interval-steps 50 --output benchmark-results\tutorial.json
.\.venv\Scripts\python.exe benchmarks\plot_noncooperative.py benchmark-results\tutorial.json --output-dir benchmark-results\tutorial-plots
```

这是接口与日志练习预算，不是学习收敛验收。MADDPG/MASAC 在同场景连续动作组比较，
QMIX 因个体奖励不兼容会被明确标记。基准有自身构造/覆盖配置的逻辑，不会自动使用你
刚修改的能源训练 YAML。并发、resume、间隔和图表说明以 [基准手册](../benchmarks/README.md)
为准。教学脚本输出目录默认 `benchmark-results/doc-tutorial`，与正式实验隔离。

## 保存与恢复

```python
experiment.trainer.save_checkpoint("benchmark-results/my-run/checkpoint.pt")
restored = build_experiment(env, config, device="cpu", seed=42)
restored.trainer.load_checkpoint("benchmark-results/my-run/checkpoint.pt")
next_metrics = restored.trainer.train_rollout([43])  # MAPPO：外部记录并恢复下一轮 seed
```

必须在完整 update 后保存。当前 trainer schema 为 5，检查配置及 `input_spec` 中的历史/
节点布局、静态邻接，再检查模型参数和 buffer；旧 schema 或配置/布局不一致会拒绝加载。
调参应作为新实验，不能暗中冒充相同
轨迹续训。模型权重 alone 不包含 optimizer 动量、replay、RNG 或进度。实验代码还需记录
下一 round/seed、采样器 RNG、活跃环境的处理办法；当前 trainer 不保存环境状态。
基准脚本自有恢复清单，普通示例不自动提供相同能力。

## 常见错误定位

| 现象 | 先查 |
|---|---|
| replay 样本不足 | ready、batch_size、是否真正 record，先采集再 update |
| on-policy rollout 已消费 | 是否重复 update 同一对象或 to 后还用旧对象 |
| horizon 之前结束 | 环境真实 episode 长度，当前不支持部分 reset |
| 全部动作非法 | adapter mask、终止观测、padding 占位动作 |
| 循环策略每步像失忆 | 是否使用 act 丢状态，是否保存/回传 h/c |
| config 旧/未知字段 | YAML schema=2；五算法使用嵌套网络配置，MAAC 另有 embedding，QMIX 另有超网络字段 |
| 历史/图网络构建失败 | adapter 的 observation_history / state_layout、邻接尺寸及网络使用位置 |
| checkpoint 不兼容 | trainer schema=5；config/input_spec/参数/buffer，不能静默部分恢复 |
| GPU 更慢 | CPU step、拷贝、微小网络、频繁指标同步；比较端到端 |
| loss 有限但回报不升 | 奖励符号/尺度、探索、对手变化、策略评估与 baseline |

<a id="verification"></a>
## 修改后的验证

先执行与变更有关的测试，完整成果再跑门禁；不以长训练替代固定张量单测。

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp=.pytest_cache/refactor-tmp
.\.venv\Scripts\python.exe -m ruff check marl examples tests
.\.venv\Scripts\python.exe -m mypy marl examples tests
.\.venv\Scripts\python.exe -m pip check
git diff --check
.\.venv\Scripts\python.exe doc\tools\build_api.py --check
.\.venv\Scripts\python.exe doc\tools\check_docs.py
```

数学/语义：[individual_rewards](../tests/test_individual_rewards.py)、[ppo_components](../tests/test_ppo_components.py)、
[maac_attention](../tests/test_maac_attention.py)。生命周期：[on_policy](../tests/test_on_policy.py)、
[off_policy](../tests/test_off_policy.py)、[recurrent_mappo](../tests/test_recurrent_mappo.py)。
配置/扩展：[typed_config](../tests/test_typed_config.py)、[experiment](../tests/test_experiment.py)。
性能迁移：[migration_equivalence](../tests/test_migration_equivalence.py)、[speed_runtime](../tests/test_speed_runtime.py)。
历史与图网络：[network_configuration](../tests/test_network_configuration.py)，覆盖窗口语义、
因果 mask、图聚合、梯度归属、五算法更新和下一次更新恢复一致性。
