# 网络配置：从观测布局到可训练模型

[手册首页](README.md) · [算法](algorithms.md) · [训练](training.md) ·
[完整默认值与迁移](../docs/network_configuration.md) · [YAML 字段](../examples/configs/README.md)

本页说明当前内置网络的使用契约。网络构造与训练已接通，不代表完成能源论文的全部
数学、数据或实验复现；具体差距见[论文对应表](../docs/network_configuration.md)。

## 先确定配置位置

通常的单智能体网络是 `encoder → backbone → 输出头`。encoder 决定怎样解释输入，
backbone 决定怎样提取后续特征，动作/Q/value 输出头由算法和 EnvironmentSpec 决定尺寸。
集中式网络还需装配节点、动作或 attention；不能把所有组件任意填进 backbone。

| 算法 | policy 配置 | critic / mixer 配置 | 输出 |
|---|---|---|---|
| MAPPO | independent_discrete：encoder、backbone | centralized_value：encoder、graph、backbone | 离散动作及逐体 value |
| MAAC | independent_discrete：encoder、backbone | attention_q：encoder、graph、embedding、backbone、attention_heads | 每个智能体每个候选动作的 Q |
| MADDPG | independent_deterministic：encoder、backbone | independent_centralized_q：encoder、graph、backbone | 连续动作及逐体 Q |
| MASAC | independent_gaussian：encoder、backbone | twin_independent_centralized_q：encoder、graph、backbone | 连续动作、log-prob 及两组逐体 Q |
| QMIX | shared_discrete_q：encoder、backbone | qmix_mixer：encoder、graph、mixing_dim、hyper_hidden_dims、value_backbone | 局部动作 Q 及团队 Q_tot |

所有 encoder 默认 identity，也可使用 history。普通 backbone 默认 MLP；只有 MAPPO
的 policy/critic backbone 可使用跨步 GRU/LSTM。QMIX mixer 没有普通 `backbone` 字段：
编码后的 state 进入权重超网络和独立 value 分支，非负混合权重保持局部 Q 单调性。
`mixing_dim` 与 `value_backbone.output_dim` 分别配置，改变前者不会自动改变后者。

MLP 的 `hidden_dims` 不含最后特征层。`hidden_dims=(), output_dim=32` 仍会创建
Linear → 可选 LayerNorm → ReLU，之后才是算法输出头，并非 identity。
五份默认 YAML 与 Python 默认 Config 一致，历史示例是另外一份实验配置。

<a id="time"></a>
## 两种时间维不能混用

设 W 为单条观测的历史窗口，T 为 rollout 中连续环境步，P 为当前特征数。

| 模式 | 配置 | 一次局部编码 | 状态与训练数据 |
|---|---|---|---|
| 独立窗口 | encoder.kind=history；temporal=GRU/LSTM/Transformer | `[...,O] → [...,W,C] → [...,H] → [...,H+P]` | 每次零初态；transition 可独立抽样 |
| 跨步循环 | MAPPO backbone.kind=GRU/LSTM | `[B,T,E] → [B,T,H]` | 传递 h/c；rollout 连续切片反传 |

历史编码保留所有前置维。例如 MAPPO 的局部输入 `[B,T,O]` 会提取 `[B,T,W,C]`，
分别处理 B×T 个完整窗口；不同 T 位置之间只有后续跨步 backbone 才能传递隐藏状态。
窗口内部参与反传，输出取最后位置；GRU/LSTM 返回的跨窗口状态不被保存。

Transformer 只用于 `encoder.temporal`，有正弦位置编码和因果注意力，dropout 固定为 0。
它按 W 维处理时间，不把 batch 或 agent 维当作时间。历史长度来自布局，不能在算法 YAML
中另填 W/C/N；没有历史布局时构建失败。

只有历史 encoder、后接 MLP 的 MAPPO，`update.sequence_length` 必须为 null，rollout
仍可展平打乱。使用跨步 backbone 时，sequence_length 才表示连续反传片段长度；此时
null 表示整段 rollout。两种机制可组合，详见[循环 MAPPO](../docs/recurrent_mappo.md)。

## adapter 显式声明输入含义

`HistoryLayout.history_indices` 是按时间从旧到新排列的 W 行索引，每行 C 个通道；
`current_indices` 按输出拼接顺序指定其他特征。两者必须完整覆盖观测且互不重叠。
下面是可独立运行的布局与编码例子，输入按通道平铺而不是按时间平铺：

```python
import torch
from marl.core import HistoryLayout
from marl.models import HistoryEncoderConfig, LSTMBackboneConfig

# [x0, x1, x2, y0, y1, y2, current]
layout = HistoryLayout(((0, 3), (1, 4), (2, 5)), (6,))
encoder = HistoryEncoderConfig(
    temporal=LSTMBackboneConfig(hidden_dim=8, num_layers=1)
).build(input_dim=7, layout=layout)
inputs = torch.arange(14, dtype=torch.float32).reshape(2, 7)
outputs = encoder(inputs)
assert outputs.shape == (2, 9)  # H=8，加一项 current
torch.testing.assert_close(outputs[:, -1], inputs[:, 6])
```

adapter 把上述 layout 写入 `EnvironmentSpec.observation_history`，各 actor 才能读取它。
该字段不隐式定义全局 state：MAPPO value 或 QMIX mixer 使用历史/图编码时，还需要
独立的 `StateLayout`。下面的片段接续上例，表示两个节点的 state 都按同样的 7 维排列：

```python
from marl.core import StateLayout
from marl.envs import ActionKind, EnvironmentSpec

spec = EnvironmentSpec(
    num_agents=2, observation_dim=7, action_dim=3, state_dim=14,
    action_kind=ActionKind.DISCRETE, horizon=8,
    observation_history=layout,
    state_layout=StateLayout(
        node_indices=(tuple(range(7)), tuple(range(7, 14))),
        history=layout,
    ),
)
```

StateLayout 的 `node_indices` 用全局 state 索引；其 `history` 使用单节点内的相对索引。
额外全局特征写在 state_layout.current_indices，节点编码后将它们拼接到末尾。
仅有 identity 且 graph=null 时直接保留原始 state 顺序，无需声明节点布局。

## GNN 的位置、邻接和信息边界

开启方式是在 critic 或 mixer 中设置 `graph: {kind: gnn, hidden_dim: 128, num_layers: 2}`。
各节点表示一个智能体。MAAC/MADDPG/MASAC 使用已有的观测 agent 维；MAPPO value 和
QMIX mixer 从 state_layout 取节点，不能仅凭 S 恰好能被 N 整除而猜测布局。

静态邻接来自 `spec.adjacency`，要求有限、非负的 N×N 矩阵，可非对称。
未提供时使用无自环完全图 A，然后统一计算 `D^-1(A+I)`；提供的对角值也会加 1。
归一化矩阵在本次前向各层复用，不随层数重复添加自环。每层把自身特征与邻居加权均值
拼接，再做 Linear+ReLU；单节点同样有效。目前不支持动态或批量通信图。

graph 只处理观测或 state 节点特征，当前动作在 critic 后续分支接入。局部 actor 与
QMIX 局部 Q 不读取邻居。共享参数不等于共享其他节点的输入数据。

## 参数究竟由谁持有

以下说明针对内置组件；外部组件需要自行保证同等信息与梯度契约。

| 组件 | 参数归属 |
|---|---|
| MAPPO/MAAC/MADDPG/MASAC actor | 每个智能体独立 encoder、backbone 和动作头 |
| MAAC critic | 每个智能体独立历史 encoder、own embedding、state-action embedding、Q head；attention 及可选 GNN 在该 critic 内共享 |
| MADDPG critic | N 个独立 Q；每个 Q 内又有 N 个独立局部观测编码器 |
| MASAC critic | 两组独立 Q 集合，合计 2N 个 Q；每个 Q 有自己的 N 个局部编码器 |
| MAPPO value / QMIX mixer 的 StateEncoder | 一个局部历史编码器对 N 个 state 节点共享；各节点仍独立零初态处理窗口 |
| QMIX 局部 Q | 所有智能体共享同一 encoder、backbone、Q head，只输入自身观测 |

同一个冻结 Config 可复用，但每次 build 都创建新模块，不因此共享参数。
actor 与 critic 不共享编码参数，MASAC twin 分支不共享新增参数。target 是独立副本，
不参与梯度更新，由 target update 同步。MAAC 的 own/state-action 分支在一次前向中
复用同一份已编码局部观测，但它们的后续 embedding 权重独立。

## 能源 MAAC：从配置到实际更新

能源 adapter 的 W 来自环境配置 `history_steps`，默认 48。购电价、售电价、PV、需求
共同形成 `[48,4]`，每个局部编码器使用一套四输入 LSTM，不是四套独立 LSTM。
窗口包含当前时刻及其前 W−1 步，日初不足部分重复当天首值。当前特征为时段及其他
10 项设备/环境状态，因此 `O=4W+11=203`，128 维 LSTM 编码后为 `E=128+11=139`。

在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe examples\train_energy_maac.py --config examples\configs\algorithms\maac_energy_history.yaml --environment examples\configs\environments\energy_trading.yaml --episodes 1 --num-envs 1 --device cpu --checkpoint .pytest_cache\energy-history.pt
```

这是 48 次环境步、batch_size=2 的小验收，按脚本每步检查 ready 的调度应完成 47 次
离策略 update。应看到有限 loss 和 checkpoint 的 updates=47，不能仅以构造成功为通过。
CPU 耗时受线程数影响，可在小环境验证前设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。

Python 中可用相同配置完成一次采样和更新；下列代码是可独立运行的完整片段：

```python
import torch
from marl.algorithms import MAACConfig
from marl.config import load_algorithm_config
from marl.envs import Transition, build_energy_trading_adapter, load_environment_config
from marl.experiment import build_experiment

env = build_energy_trading_adapter(
    load_environment_config("examples/configs/environments/energy_trading.yaml")
)
config = load_algorithm_config("examples/configs/algorithms/maac_energy_history.yaml")
assert isinstance(config, MAACConfig)
experiment = build_experiment(env, config, device="cpu", seed=42)
current = env.reset(seed=42)
while not experiment.trainer.ready:
    assert current.action_mask is not None
    with torch.inference_mode():
        actions = experiment.algorithm.act(
            torch.as_tensor(current.observations).unsqueeze(0),
            action_mask=torch.as_tensor(current.action_mask).unsqueeze(0),
        )[0].cpu().numpy()
    following = env.step(actions)
    experiment.trainer.record(Transition(current, actions, following))
    current = following
metrics = experiment.trainer.update()
assert experiment.trainer.optimization.update_count == 1
print(metrics["loss"])
```

示例使用独立 actor/critic LSTM，graph=null。1 层、128 维是工程默认值，不能当作论文
披露参数。合成曲线、首值填充、54 项动作量化、现有 MHA 和小 replay 都是当前边界；
真实跨日历史、混合动作、论文 attention 精确对齐与大规模实验仍需后续实现。

## 函数级输入输出与副作用

| 入口 | 输入 → 输出 | 作用与副作用 |
|---|---|---|
| [HistoryEncoderConfig.build](api/marl-models-encoder.md#historyencoderconfig-build) | input_dim、HistoryLayout → HistoryEncoder | 校验布局，创建时序参数与索引 buffer；缺布局报错 |
| [HistoryEncoder.forward](api/marl-models-encoder.md#historyencoder-forward) | `[...,O] → [...,H+P]` | 重排窗口、取最后输出、拼接当前特征；不保存跨调用状态 |
| [ObservationEncoder.forward](api/marl-modules-encoding.md#observationencoder-forward) | `[...,N,O] → [...,N,E]` | 逐体编码及可选图聚合；不读取动作 |
| [JointActionEncoder.forward](api/marl-modules-encoding.md#jointactionencoder-forward) | `[...,N*(O+A)] → [...,N*(E+A)]` | 输入顺序为全部观测后接全部动作；只编码观测，保留动作梯度 |
| [StateEncoder.forward](api/marl-modules-encoding.md#stateencoder-forward) | `[...,S] → [...,N*E+G]` | 按节点布局取值，共享局部编码，拼接 G 项全局当前特征 |
| [QMixer.forward](api/marl-modules-mixer.md#qmixer-forward) | `[...,N]` 与 `[...,S] → [...,1]` | 检查相同批维，以非负权重混合 Q；不执行优化 |

上述 forward 构建计算图但不调用 backward 或 optimizer.step；训练由算法 update 管理。
序列/布局 buffer 随模型迁移设备并进入 checkpoint。schema 5 记录完整 input_spec，
同尺寸但排列不同的历史也会拒绝恢复，详见[训练器恢复边界](trainers.md)。
