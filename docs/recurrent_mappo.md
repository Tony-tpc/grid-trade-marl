# MAPPO 的 MLP / GRU / LSTM 配置与循环训练

## 实现与数学语义审计

| 要求 | 处理 |
|---|---|
| PPO clip、entropy、value 公式 | 复用原有 objective，不另写循环版公式 |
| 每个智能体独立 actor | 保留 ModuleList 独立参数，隐藏状态增加明确的 N 维 |
| 集中式 critic | 一份 state 编码输出 N 个 value，actor/critic 不共享参数或状态 |
| LSTM h/c | 使用原生 nn.LSTM；双状态同形状、同 dtype/device |
| 经验来源 | 只使用 fresh rollout，禁止离策略重放 |
| GAE | 保留逐智能体语义；先估计，再分块/补齐 |
| 更新 | MAPPO.update 编排，BaseMARLAlgorithm.optimize 唯一 backward/step 入口 |
| 时间方向 | 单向 RNN；不使用未来观测、不提供 bidirectional/projection/dropout |
| TBPTT | 采样策略的 detached 初态，片段内反传；无 burn-in，初态不按新参数重算 |
| 环境生命周期 | 每次 rollout reset；中途结束报错，不隐式 autoreset |
| 恢复 | 仅完整 rollout 更新后的边界；不保存活跃环境或半段 rollout |

参考接口语义：[PyTorch LSTM](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html)、
[Recurrent PPO 状态推理](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)。
不引入 SB3/TorchRL 依赖，也不复制其 trainer。

## Python：配置和构造

```python
from marl.algorithms import MAPPOConfig
from marl.experiment import build_experiment
from marl.models import LSTMBackboneConfig, MLPBackboneConfig
from marl.modules.policy import IndependentDiscreteConfig
from marl.modules.critic import CentralizedValueConfig
from marl.training import PPOUpdateConfig
from marl.envs import MemoryCueAdapter

config = MAPPOConfig(
    policy=IndependentDiscreteConfig(
        backbone=LSTMBackboneConfig(hidden_dim=64, num_layers=1),
    ),
    critic=CentralizedValueConfig(
        backbone=MLPBackboneConfig(hidden_dims=(128, 128), output_dim=64),
    ),
    update=PPOUpdateConfig(epochs=2, mini_batch_size=64, sequence_length=10),
)
experiment = build_experiment(MemoryCueAdapter(), config, seed=7)
metrics = experiment.trainer.train_rollout([7])
```

actor/critic 独立选择 MLP/GRU/LSTM 的 9 种组合均使用 MAPPO，没有额外算法名称、
UpdatePlan 或优化基类。外部无状态 critic 可以继续是 nn.Sequential。

MLP 的中间层列表不包括最后特征层；旧 hidden_dim=64 只对应新 output_dim=64，
保持原有 (128,128) 中间层、ReLU 和 LayerNorm 放置。GRU/LSTM 暂无 LayerNorm 选项。

## YAML 与不兼容迁移

算法 YAML 升级至 schema_version: 2，环境 YAML 仍为 1。完整默认值见算法模板。
网络使用 policy/critic.backbone；未知字段和旧顶层 hidden_dim/layer_norm 会报错，
不同时保留两套读法。MAAC 的离散 policy 配置同步迁移，但拒绝循环网络。

```yaml
schema_version: 2
algorithm: mappo
policy:
  kind: independent_discrete
  backbone:
    kind: lstm
    hidden_dim: 64
    num_layers: 1
critic:
  kind: centralized_value
  backbone:
    kind: mlp
    hidden_dims: [128, 128]
    output_dim: 64
    layer_norm: false
update:
  epochs: 2
  mini_batch_size: 64
  sequence_length: 10
```

训练 checkpoint 升级为 schema 4，旧 schema 不自动迁移。加载前校验配置和参数
名称/形状/dtype；错误不会先修改网络或 RNG。旧实验保留，使用审计基线 commit
724487d 的代码复现，不把模型权重单独加载当作精确续训。

## 张量、状态与连续片段

K=层数，B=环境/序列数，T=时间，N=智能体数，H=隐藏维度。

| 边界 | 观测/state | 状态 |
|---|---|---|
| Actor 单步 | [B,N,O] | [K,B,N,H] |
| Actor 序列 | [B,T,N,O] | [K,B,N,H] |
| Critic 单步 | [B,S] | [K,B,H] |
| Critic 序列 | [B,T,S] | [K,B,H] |

GRU 状态是 Tensor，LSTM 是 (h,c)，MLP 是 None。状态不是 distribution_params。
MARLModelOutput 的 policy_state/value_state 是处理当前输入之后的状态；rollout
保存的是处理当前输入之前的状态。初始 None 表示零状态。

在线循环策略使用 sample(..., policy_state=..., value_state=...)，并把返回状态传回
下一步。仅需要动作时可用 algorithm.policy.act(..., hidden_state=...) 并保存其
policy_state；MAPPO.act 的 Tensor-only 接口会拒绝循环 actor，防止每步失忆。

sequence_length=null 在循环网络中使用整个 rollout；纯 MLP 不设置非空值。
长度必须 <= horizon 且 <= mini_batch_size。mini_batch_size 按时间位置计数，不乘 N：
长度 16、mini_batch_size 64 表示每次最多 4 条片段。horizon=25 时每个环境分成
16+9，第二条右补 7 个位置。每 epoch 打乱片段，不打乱片段内部时间。

先按原始 [T,B,N] 计算 GAE/return，再分块。序列初态来自对应环境与片段起点，
并在片段边界 detach；padding 从所有目标、诊断、target 统计及加权计数排除。
Padding action mask 提供合法占位动作，不能先造出无效分布再期望 mask 消掉 NaN。
纯 MLP 仍使用原有展平 mini-batch，既有网络参数布局与固定更新结果保持等价。

## 终止、评估与恢复

terminated 不 bootstrap；truncated 使用最终观测与携带的 critic 状态 bootstrap，
不能先 reset 再估值。当前每份 rollout 都重置环境，因此下一份状态也重置。
不支持部分环境中途 reset；不承诺环境步数小于 episode 长度时续接被截断的历史。

评估和 cross-play 每个来源、每个 episode 独立持有状态；不混用两个来源的 h/c。
评估结束恢复每个子模块原始 train/eval 标志。确定性评估不采样训练 RNG；原有随机
对手评估在外层 fork_rng 中隔离随机数。循环状态仅归本次评估，不写回训练对象。

保存 checkpoint 时应完成此次更新。恢复模型、独立 optimizer、更新次数、实际
optimizer-step 次数、normalization buffers、Torch RNG 和 mini-batch generator；
调用方仍需恢复下一次 reset seeds。基准保存自己的运行参数与 seed 调度。

## 固定预算验收和图表解释

```powershell
.\.venv\Scripts\python.exe -m benchmarks.benchmark_recurrent --output benchmark-results/recurrent_mappo_20k_10k_x3 --device cpu
.\.venv\Scripts\python.exe -m benchmarks.plot_recurrent benchmark-results/recurrent_mappo_20k_10k_x3/report.json --output benchmark-results/recurrent_mappo_20k_10k_x3/plots
```

- 三种 actor、critic 固定 MLP；seed 7/17/29，共 18 个运行。
- 记忆任务各 20k 步，提示仅首帧可见，末步作答，中间动作不能改写环境存储。
  四种联合提示平衡评估；MLP 机会水平 50%，GRU/LSTM 各 seed 阈值 >=80%。
- MPE 各 10k 步，每 100 步固定 8 个场景评估，终局再使用独立 64 个场景；
  记忆任务每 200 步评估。均包含 step 0，总计各 101 个评估点。
- JSON/CSV/checkpoint、逐 rollout 指标和 SHA256 全部保存在忽略目录。
- 曲线细线是原始 seed，粗线是均值，色带是 Student-t 95% CI；3 个 seed 的区间
  可能很宽。loss 不单调不是不收敛的直接证据，数值有限也不是已收敛的证据。
- 对手与 good-agent 回报分开解释；cross-play 行是对手来源、列是 good-agent 来源。
- 效率图包括评估开销；profile 是主机累计调用时间，嵌套项不能相加，也不是 CUDA
  kernel 计时。CPU 运行的 peak_cuda_bytes 为 null，不能当作显存测量结果。
- 架构参数量不同，不能据此宣称等参数量、公平算力预算的论文比较。

## 本轮实际验收结果（2026-09-27）

### 工程与记忆能力

- 审计基线：724487d。CPU/CUDA 均实际执行 9 种 actor/critic 组合，未跳过 CUDA。
- 完整 pytest：217 项通过；Ruff、mypy（81 个源文件）、pip check、
  git diff --check 通过。覆盖多层且 actor/critic 层数、宽度不同的状态，
  不仅覆盖单层同宽配置。
- 序列 padding 改值不改变有效 loss/梯度；完整前向与逐步传状态一致；
  采样 log-prob/value 可重算；参数隔离、真终止/截断、消费保护及损坏
  checkpoint 回滚均有自动测试。
- CUDA 连续化后的 CPU 张量重新 pin，h/c 历史仍留在 CUDA；设备迁移转交一次性
  消费权，不创建第二份可训练的旧 rollout。

| 记忆任务 Actor | seed 7 | seed 17 | seed 29 | 参数量（含 MLP critic） |
|---|---:|---:|---:|---:|
| MLP | 50% | 50% | 50% | 76,102 |
| GRU | 100% | 100% | 100% | 51,910 |
| LSTM | 100% | 100% | 100% | 60,614 |

全部严格使用预定 20,000 步，没有更换 seed、追加训练或挑选最好 checkpoint。
每个运行保存 500 个训练诊断点、101 个评估点、2,000 次实际 optimizer.step。
这一结果证明隐藏状态、时间梯度和循环推理链能解决本专用任务，不代表一般博弈已收敛。

### MPE2 短程对照

下表为每个种子在独立 64 个场景上的终局平均回报，再对 3 个种子取均值 ± 样本标准差。
good-agent 一列是两个 good-agent 的平均，不与对手合并；不是跨角色总分。

| Actor | 对手回报 | good-agent 回报 | 参数量（含 MLP critic） |
|---|---:|---:|---:|
| MLP | -27.71 ± 1.07 | -15.55 ± 1.63 | 108,178 |
| GRU | -25.50 ± 2.23 | -21.16 ± 4.58 | 73,426 |
| LSTM | -24.38 ± 2.17 | -10.21 ± 9.47 | 88,018 |

每个运行严格使用 10,000 步、100 个训练诊断点、101 个评估点、800 次 optimizer.step。
全部训练指标、参数和评估回报有限，JSON/CSV 的 18 个运行与 18 份最终 checkpoint
逐项核对，SHA256 一致；没有把数值有限解释成收敛。尤其 LSTM 的 good-agent 跨种子
差异仍很大，不能用上述均值宣称算法优越性。

终审发现过 cross-play 的 good-agent 来源未回传隐藏状态，已修复并新增 9 种来源
组合的回归测试。**最终 9×9 两个角色矩阵由原有 checkpoint 重新评估生成**，不重训、
不改参数、不改评估 seed。含错误矩阵的旧报告仅留作审计备份，不用于最终图表或结论。

### MLP 性能回归

两侧统一使用 1,000 环境步、80 次 optimizer.step、相同网络/seed/日志频率，
每设备一次预热、5 次测量。正式短实验结束后，直接从 724487d 的源码归档运行旧版，
再运行新版，以降低其他训练负载对回归判断的影响。

| 设备 | 基线中位耗时 | 新版中位耗时 | 新版/基线吞吐 |
|---|---:|---:|---:|
| CPU | 1.190 s | 1.136 s | 1.048× |
| CUDA（RTX 4060 Laptop GPU） | 2.821 s | 2.608 s | 1.082× |

两种设备均满足“吞吐回退不超过 5%”，最终全部 state_dict 张量与基线逐项完全相等。
早期有负载时的两轮测量也保留在 benchmark-results，不只保存最好的数字。
这些短测含明显墙钟波动，仅作为本机回归门禁，不把 4.8%/8.2% 宣称为通用加速。
本小网络与 CPU 环境的短测中 CUDA 更慢，不能据此推广到大模型或 GPU 环境。

### 产物、图表和可追溯性

本机产物目录：benchmark-results/recurrent_mappo_20k_10k_x3。

- report.json / summary.csv：18 个运行的原始结果与汇总；verification.json：
  独立产物审计、SHA/CSV 检查和性能原始重复测量。
- plots/：16 张图；memory/mpe 各含 learning、loss、ppo、gradients、
  recurrent_state、efficiency、timing，另有 heldout_returns 与 cross_play。
- learning 展示原始 seed 曲线、均值和 Student-t 95% CI；t 区间不强制裁剪到
  成功率的 [0,1] 或回报理论范围，因此小样本色带可越界，不能当作新的概率值。
- 主实验在 CPU 上执行；峰值显存为 null 并在图中注明 CPU，不虚构 CUDA 显存。
  CUDA 在 9 组合工程测试与性能回归中验证。短实验启用 cProfile，其耗时不能当作
  不带 profile 的纯训练性能测量。
- training_source.zip、evaluation_source.zip、final_source.zip 分别保存训练、
  修正后 cross-play 和最终验证使用的源码。前两个归档的 SHA256 已与对应报告
  source_sha256 独立核对。训练后加入了输入校验、损坏恢复回滚、CUDA pinned
  存储检查和格式整理；这些没有改变本次 CPU 训练目标函数或更新调度。
- 源码、测试与说明进入完整功能提交；上述原始结果、图像和 checkpoint 均不入 Git。

## 尚未支持的边界

不支持离策略 GRU/LSTM replay、跨 rollout 的环境/状态续接、部分环境 autoreset、
中途 checkpoint、burn-in、双向循环、LSTM projection 或循环 dropout。
半段 rollout 不作为可恢复检查点；调用方在完整更新后恢复下一次 reset seeds。
本次没有启动 100k×5 长实验，也没有为网络变体新增算法类或算法注册名。
