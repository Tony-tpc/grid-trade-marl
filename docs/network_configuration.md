# 五种算法的统一网络配置与历史编码

[网络使用教程](../doc/networks.md) · [手册首页](../doc/README.md) ·
[配置全集](../examples/configs/README.md)

本轮补齐网络可配置性和历史编码训练路径。五种算法的目标函数、更新顺序、逐智能体奖励及
termination/truncation 语义沿用原实现。完整论文复现仍需要混合动作、论文 attention 公式
对齐、真实数据和实验评估。本说明中的 LSTM 层数、隐藏宽度是工程选择。

## 论文要求与仓库能力

| 类别 | 修改前审计 | 本轮结果与边界 |
|---|---|---|
| 问题与奖励 | 已满足：支持一般和博弈及 individual reward | 保留逐智能体 TD、Q 和策略目标；能源环境拒绝 QMIX |
| 智能体 | 已满足：固定 N、同构观测、独立 MAAC actor/critic 局部参数 | MAAC 各家庭局部编码独立；state 节点编码与 QMIX 局部 Q 有明确共享关系，见教程 |
| 历史观测 | 缺失：四组历史平铺输入 MLP | 显式重排 [W,4]，LSTM 最后隐藏输出拼接当前特征，W=48 |
| 网络 | 缺失：连续策略、独立 Q、MAAC 嵌入和 mixer 写死层宽 | Python/YAML 统一嵌套配置；默认 MLP 数值回归保持一致 |
| 动作 | 缺失/有偏差：54 项离散量化或连续阈值近似 | 本轮保留；尚未实现论文的混合 Bernoulli/Gaussian 动作 |
| 数据 | 可复用：独立 transition replay；缺失真实数据管线 | 窗口内反传、窗口间零状态，不需要序列 replay；仍使用合成曲线 |
| return | 已满足当前实现：TD0、true termination 控制 bootstrap | 数学不变；不把一般 MAAC 支持视为论文所有细节已对齐 |
| attention 与目标 | 可复用：候选动作 Q、反事实 baseline、熵项 | 保留现有 MHA；论文 attention 精确公式仍待单独审计与实现 |
| 更新与终止 | 已满足：critic/actor/target 更新、终止与截断分离 | 新编码参数由原 optimizer 管理，target 编码参数不反传 |
| 评估与恢复 | 可复用：小环境测试与 checkpoint | schema 5 验证网络和输入布局，检查恢复后的下一次更新一致 |
| 规模与数据历史 | 缺失：真实跨日历史、大规模 replay 容量优化 | 本轮不处理，不宣称完成论文数值复现或收敛验证 |

## 两类时间维的区别

统一路径为 **输入编码器 → backbone → 算法输出头**。
集中式网络另有节点/动作/attention 装配；QMIX mixer 使用 state 编码加超网络和 value
分支，不提供普通 `mixer.backbone` 字段。

- `encoder.kind: identity` 原样返回输入，无参数。
- `encoder.kind: history` 使用 adapter 声明的窗口，每次调用零初态，不保存跨调用状态。
  先将 `[*B,O]` 重排为 `[*B,W,C]`，取时序网络最后位置的 `[*B,H]`，
  拼接 `current_indices` 指定的当前特征，返回 `[*B,H+K]`。
- `backbone.kind: gru/lstm` 是跨环境步循环模式，仅 MAPPO 支持，需要旧有的
  状态传递、连续序列 mini-batch 与 padding 屏蔽。它可以与 history encoder 组合。

五种算法都可使用 history GRU/LSTM/Transformer。离策略算法拒绝跨环境步循环 backbone。
Transformer 只在 `encoder.kind: history` 下的 `encoder.temporal` 中选择，
使用正弦位置编码、因果 mask、最后时间位置输出，
dropout 固定为 0；不会将 replay batch 或 agent 维当作时间维。
GNN 只在集中式 critic/mixer 的 graph 中选择；actor 不能读取邻居信息。

## Python 与 YAML

```python
from marl.algorithms import MAACConfig
from marl.models import HistoryEncoderConfig, LSTMBackboneConfig, MLPBackboneConfig
from marl.modules.policy import IndependentDiscreteConfig
from marl.modules.critic import AttentionQConfig

history = HistoryEncoderConfig(temporal=LSTMBackboneConfig(hidden_dim=128, num_layers=1))
config = MAACConfig(
    policy=IndependentDiscreteConfig(
        encoder=history,
        backbone=MLPBackboneConfig(hidden_dims=(128, 128), output_dim=128),
    ),
    critic=AttentionQConfig(
        encoder=history,
        embedding=MLPBackboneConfig(hidden_dims=(), output_dim=128),
        backbone=MLPBackboneConfig(hidden_dims=(), output_dim=128),
        attention_heads=4,
    ),
)
```

同一个冻结 Config 可以复用；每次 build 创建新的 nn.Module 和参数，不因复用配置而共享网络。
单个组件内部仍可有规定的共享关系，例如 QMIX 局部 Q 和 StateEncoder 的节点编码。
YAML 仍为 schema 2，例如：

```yaml
schema_version: 2
algorithm: maac
policy:
  kind: independent_discrete
  encoder:
    kind: history
    temporal: {kind: lstm, hidden_dim: 128, num_layers: 1}
  backbone: {kind: mlp, hidden_dims: [128, 128], output_dim: 128, layer_norm: false}
critic:
  kind: attention_q
  encoder:
    kind: history
    temporal: {kind: lstm, hidden_dim: 128, num_layers: 1}
  embedding: {kind: mlp, hidden_dims: [], output_dim: 128, layer_norm: false}
  backbone: {kind: mlp, hidden_dims: [], output_dim: 128, layer_norm: false}
  attention_heads: 4
  graph: null
```

| 字段 | 默认值与含义 |
|---|---|
| encoder | `{kind: identity}`；history 需要布局 |
| encoder.temporal（history） | `{kind: lstm, hidden_dim: 128, num_layers: 1}`；也可 kind: gru |
| encoder.temporal（Transformer） | `{kind: transformer, model_dim: 128, num_heads: 4, num_layers: 2}` |
| MLP backbone | hidden_dims=(128,128)，output_dim=128，layer_norm=False |
| MAAC embedding/backbone | hidden_dims=()，output_dim=128，layer_norm=False，各自可配置 |
| MAAC attention_heads | 4，必须整除 embedding.output_dim；Q 输出维度始终来自 A |
| graph | None；开启为 `{kind: gnn, hidden_dim: 128, num_layers: 2}` |
| QMIX mixing_dim | 32，混合隐层宽度 |
| QMIX hyper_hidden_dims | ()；非空时为超网络逐层宽度，线性输出仍由 N/mixing_dim 决定 |
| QMIX value_backbone | MLP hidden_dims=()、output_dim=32，最后标量输出由 mixer 添加 |

Transformer 的 model_dim 必须能被 num_heads 整除；GRU/LSTM 无额外 dropout 配置，
MLP 的 layer_norm 不适用于它们。QMIX value_backbone.output_dim 不随 mixing_dim 自动变化。

所有宽度/层数为正整数。MLP hidden_dims 和 hyper_hidden_dims 允许空列表。
MLP 每个中间层和最后特征层均使用可选 LayerNorm 及 ReLU；动作/Q/value 输出头不包含在
hidden_dims 中。连续 actor、独立 Q/twin-Q 的默认 MLP 仍为 128→128→128。

旧 policy/critic 顶层 hidden_dim、layer_norm 明确报错，没有兼容别名。
原 `hidden_dim=64` 对普通网络迁移为 `backbone.output_dim=64`，中间层仍为 (128,128)；
MAAC 原隐藏宽度同时迁移到 embedding.output_dim 与 backbone.output_dim，
原 layer_norm 同时迁移到这两个嵌套配置。动作类型和输出维度由算法及 EnvironmentSpec 决定。
ExtensionCatalog 外部组件入口保留，YAML 不导入 Python 类。

## adapter 如何声明布局

布局类型在 [core/layout.py](../marl/core/layout.py)，也从 marl.envs 导出。每个索引必须是整数，
历史/当前特征互不重叠且完整覆盖输入。例子：输入
`[x0,x1,x2,y0,y1,y2,current]` 对应：

```python
from marl.core import HistoryLayout, StateLayout
history = HistoryLayout(((0, 3), (1, 4), (2, 5)), (6,))
state_layout = StateLayout(
    node_indices=(tuple(range(7)), tuple(range(7, 14))),
    history=history,
)
# 在 EnvironmentSpec 中传 observation_history=history、state_layout=state_layout。
```

observation_history 描述每个局部观测；state_layout 独立描述全局 state 的 N 个节点，
可以指定额外全局 current_indices。每个 state 节点的历史结构一致，节点历史网络共享参数；
MAAC/MADDPG/MASAC 的联合观测编码则为每个智能体创建独立局部编码器。
仅观测历史不隐式声明 state 历史。无布局时默认 identity+MLP 可运行；
需要历史或 state 图结构而未声明布局时，构建阶段报错。

静态 adjacency 为不可变非负有限 N×N tuple。默认使用无自环完全图 A；
前向统一采用 `D^-1(A+I)`，adapter 提供的对角权重会再加 1；本次前向所有图层
复用这份归一化邻接，不逐层重复加自环。
每层拼接自身特征和邻居加权均值后做 Linear+ReLU，节点表示智能体。
动态/批量邻接不在本轮范围。GNN 只处理观测或显式 state 节点特征，当前动作随后进入
MAAC 的 state-action embedding 或独立 Q 的联合动作分支。
QMIX 权重继续取绝对值，状态编码不依赖 agent Q，保持逐体 Q 的单调性。

## 能源 48 步 LSTM 示例

[示例配置](../examples/configs/algorithms/maac_energy_history.yaml) 与
[能源环境配置](../examples/configs/environments/energy_trading.yaml) 配合运行：

```powershell
.\.venv\Scripts\python.exe examples\train_energy_maac.py --config examples\configs\algorithms\maac_energy_history.yaml --episodes 1 --num-envs 1 --device cpu --checkpoint .pytest_cache\energy-history.pt
```

能源 adapter 将四组按通道平铺的购电价、售电价、光伏和需求历史重排为 [48,4]。
四个通道共同输入每个局部编码器的一套 LSTM，不是四套独立 LSTM；窗口包含当前时刻
及其前 47 步，W 来自环境 history_steps 而非算法 YAML。
原始 O=203，当前特征 K=11（时刻及 10 项设备/环境状态），LSTM 输出 H=128，
故 actor/critic 的后续观测输入 E=139。每个家庭的 actor、critic 分别训练 LSTM；
MAAC 的自己观测与 state-action embedding 在同一次调用中复用本家庭已编码观测，
但彼此的嵌入权重仍然独立。

LSTM 1 层、128 维、示例 replay 容量 256/batch 2 均为工程验收设置，不能视为论文参数。
现有环境使用合成单日曲线，日初缺少历史时重复当天首值，不是真实跨日历史；
54 项离散动作也是量化近似。此示例验证历史信息参与训练，不是论文结果复现。

## checkpoint 和验证

checkpoint schema 5 同时保存网络配置、参数及 EnvironmentSpec 数据快照，包括全部历史索引、
state 节点索引和静态邻接。加载先检查 schema/config/spec、参数名称/shape/dtype 和布局 buffer，
再加载 RNG、模型、optimizer/replay。旧 schema 4 及更早版本不做静默部分恢复。
这些前置校验不等于任意损坏文件都可事务回滚：off-policy/runtime 恢复不承诺完整回滚，
详见[训练器恢复说明](../doc/trainers.md)。
仅在完整更新边界保存；活跃环境进度、半段 rollout、外部 seed 调度仍由调用方管理。

[网络验收测试](../tests/test_network_configuration.py) 覆盖手算 LSTM、图聚合、时间重排、
因果屏蔽、replay 重排独立性、本地 actor 信息边界、梯度归属和 QMIX 单调性。
五算法分别用三种历史编码器、可选 GNN 完成真实小环境采样、更新与下一次更新精确恢复。
能源专项通过前向 hook 检查 [48,4]，并验证 actor 和 critic 历史参数都发生更新。
固定输入的默认动作、Q/value、loss 继续使用已有迁移基线，而非重新生成基线掩盖变化。

函数级导航：[布局](../doc/api/marl-core-layout.md)、
[窗口编码](../doc/api/marl-models-encoder.md)、
[集中式编码](../doc/api/marl-modules-encoding.md)、
[策略](../doc/api/marl-modules-policy.md)、
[critic](../doc/api/marl-modules-critic.md)、
[mixer](../doc/api/marl-modules-mixer.md)。
