# 配置参考

每个算法一份默认 YAML：`algorithms/{mappo,maac,maddpg,masac,qmix}.yaml`；
`algorithms/maac_energy_history.yaml` 是另外的历史编码验收配置。
环境独立保存在 `environments/energy_trading.yaml` 或
`environments/mpe2_simple_adversary.yaml`。
seed、device、并行环境数和输出路径由脚本/命令行管理。

## 公共规则

算法 YAML 仅 `schema_version: 2` 和 `algorithm` 必填；其余节及字段可省略，
使用对应 Config 的默认值。五份算法模板显式列出全部默认值，并有 Python/YAML 等价测试。
未知字段、错误类型、非有限浮点数、未知组件与环境尺寸字段会被拒绝。
没有 `type/options` 层；字段直接位于对应节中。

环境 schema 仍使用 `schema_version/environment/action_kind/options`。
环境的 `options` 是动力学与设备参数，含义与算法组件配置不同。

| 算法 | 配置节 | 动作要求 |
|---|---|---|
| MAPPO | policy、critic、loss、advantage、rollout、update | discrete |
| MAAC | policy、critic、loss、value_target、replay、update、target_update | discrete |
| MADDPG | policy、critic、loss、value_target、replay、update、target_update | continuous |
| MASAC | policy、critic、loss、value_target、replay、update、target_update | continuous |
| QMIX | policy、mixer、loss、value_target、replay、update、target_update | discrete，且 shared reward |

`algorithm` 决定配置结构；不能只修改名称而保留另一算法的字段。

## 网络：全部内置选项

下表的 kind 是对应算法的默认内置选择，不是各算法可任意互换的字符串。
替换网络需提供满足算法能力要求的配置；库外网络通过显式传入的 ExtensionCatalog 加载。

| 使用位置 | 默认 kind | 可选字段及默认值 |
|---|---|---|
| MAPPO/MAAC policy | independent_discrete | encoder=identity，backbone=MLPBackboneConfig() |
| MADDPG policy | independent_deterministic | encoder=identity，backbone=MLPBackboneConfig() |
| MASAC policy | independent_gaussian | encoder=identity，backbone=MLPBackboneConfig() |
| QMIX policy | shared_discrete_q | encoder=identity，backbone=MLPBackboneConfig() |
| MAPPO critic | centralized_value | encoder=identity，backbone=MLPBackboneConfig()，graph=null |
| MAAC critic | attention_q | encoder=identity，embedding/backbone=单层128维MLP，attention_heads=4，graph=null |
| MADDPG critic | independent_centralized_q | encoder=identity，backbone=MLPBackboneConfig()，graph=null |
| MASAC critic | twin_independent_centralized_q | encoder=identity，backbone=MLPBackboneConfig()，graph=null |
| QMIX mixer | qmix_mixer | encoder=identity，graph=null，mixing_dim=32，hyper_hidden_dims=[]，value_backbone=单层32维MLP |

所有层宽必须为正整数；attention_heads 必须为正整数且整除 embedding.output_dim。
观测维 O、动作维 A、智能体数 N、全局状态维 S 都由环境 spec 注入。

五算法 encoder 均可选择 history，内部 temporal 可为 GRU/LSTM/Transformer；
窗口索引及通道数来自 adapter。graph 仅用于集中式 critic/mixer，不能出现在 actor。
MLP 的 hidden_dims=[] 表示只保留最后特征层。全部网络字段、范围、Python/YAML 示例、
state 节点布局和 checkpoint schema 5 见 [统一网络配置](../../docs/network_configuration.md)。
能源路径示例使用 [maac_energy_history.yaml](algorithms/maac_energy_history.yaml)。

## 数学与训练参数：全部选项

| 节/算法 | 字段与默认值 | 含义与约束 |
|---|---|---|
| loss / MAPPO | clip_ratio=0.2 | PPO 概率比裁剪宽度，(0,1) |
| loss / MAPPO | value_coefficient=0.5 | value MSE 权重，>=0 |
| loss / MAPPO、MAAC | entropy_coefficient=0.01 | 熵奖励系数，>=0 |
| loss / 所有离策略算法 | td_coefficient=1.0 | critic TD MSE 权重，>=0 |
| loss / MAPPO、MAAC、MADDPG、MASAC | normalize_targets=false | 使用逐智能体运行尺度；Q/value 保留原始单位；QMIX 无此字段 |
| loss / MASAC | initial_alpha=0.2 | 初始逐智能体温度，>0 |
| loss / MASAC | target_entropy=null | null 使用 -action_dim；也可填写有限数值 |
| advantage / MAPPO | kind=gae，gamma=0.99 | 折扣因子，[0,1] |
| advantage / MAPPO | lambda=0.95 | GAE 递推系数，[0,1]；Python 名为 gae_lambda |
| advantage / MAPPO | normalize=true | advantage 的 batch 标准化，不改变 return |
| advantage / MAPPO | normalization_scope=per_agent | per_agent 按智能体分别标准化；global 合并样本与智能体；normalize=false 时不标准化 |
| value_target / 离策略 | kind=td0，gamma=0.99 | 一步 target，仅 terminated 阻止 bootstrap |
| rollout / MAPPO | horizon=null | null 使用环境 horizon；指定时为 1..环境 horizon |
| replay / 离策略 | capacity=100000 | ring buffer 最大 transition 数，正整数 |
| replay / 离策略 | batch_size=256 | 每次采样数，1..capacity |
| update / 全部 | learning_rate=0.0003 | Adam 学习率，>0 |
| update / 全部 | max_grad_norm=10.0 | >0 的梯度上限；null 关闭 |
| update / MAPPO、MAAC、MADDPG、MASAC | actor_learning_rate=null、critic_learning_rate=null | 正数为该角色学习率；null 回退 learning_rate |
| update / MAPPO、MAAC、MADDPG、MASAC | actor_max_grad_norm=null、critic_max_grad_norm=null | 正数为该角色裁剪上限；null 回退 max_grad_norm |
| update / MASAC | temperature_learning_rate=null、temperature_max_grad_norm=null | 温度优化器专用值，null 分别回退公共学习率/裁剪上限 |
| update / MAPPO | sequence_length=null | 跨步循环 backbone：null 用整段，正整数指定片段；仅历史 encoder 或纯 MLP 均只用 null |
| update / MAPPO | epochs=4 | 同一 fresh rollout 更新轮数，正整数 |
| update / MAPPO | mini_batch_size=256 | 时间位置数上限；循环路径按整条片段采样 |
| update / MAPPO | value_clip_ratio=null | null 关闭 value clipping；正数启用围绕 old value 的裁剪，使用原始 value 单位 |
| update / 离策略 | amp_dtype=null | null=FP32；bf16 仅支持相应 CUDA 设备 |
| target_update / 离策略 | kind=soft，tau=0.005 | 每次更新后 Polyak 同步，tau∈(0,1] |
| target_update / 离策略 | kind=hard，interval=1 | 可替代 soft；每 interval 次更新完整同步，正整数 |

hard 和 soft 的字段不能混写。MAPPO 不含 target_update。
YAML 可使用 gae_lambda 代替 lambda，但不能同时填写两者。
专用裁剪上限为 null 表示回退，不表示单独关闭该角色裁剪；公共上限也为 null 时才关闭。
这些字段的类型和默认值由 Config 与五份默认 YAML 验证，外部组件可以定义自己的字段。

## 环境参数与列表含义

环境模板逐项列出了 `EnergyTradingConfig` 的所有默认值及注释，包含 agent 数、horizon、
观测历史长度、各设备容量/功率/效率、舒适约束、出行窗口和惩罚系数。
参数范围由 EnergyTradingConfig 校验，完整物理含义见
[环境说明](../../marl/envs/README.md)。

| 列表/元组字段 | 含义 | 长度 |
|---|---|---|
| sa_cycle_kw: [1.5, 1.0] | 一个智能电器运行周期中，每个连续时段的用电功率 kW；不是逐 agent 参数 | 周期时段数，至少 1 |
| has_ev | 每个 agent 是否拥有 EV | null=全部拥有，否则长度 N |
| has_storage | 每个 agent 是否拥有储能 | null=全部拥有，否则长度 N |
| has_hvac | 每个 agent 是否拥有 HVAC | null=全部拥有，否则长度 N |
| has_sa | 每个 agent 是否拥有可移时智能电器 | null=全部拥有，否则长度 N |

这些 YAML 列表在不可变环境配置中保存为 Python tuple。
算法配置当前没有分组/group 元组；不会把环境配置里的列表解释成算法组。
`action_kind` 为 discrete 时使用 54 项动作目录，为 continuous 时使用长度 4 的动作向量。
能源环境输出 individual reward，因此不能直接与 QMIX 配对。

MPE2 simple_adversary 配置支持 `num_good_agents`、`horizon` 和
`dynamic_rescaling`。它同样输出 individual reward；将 `action_kind` 设为
`discrete` 可用于 MAAC/MAPPO，设为 `continuous` 可用于 MADDPG/MASAC。

## MAPPO 的 MLP / GRU / LSTM

MAPPO 的 policy 和 critic 分别使用嵌套 `backbone`；两侧可以独立选择。
`kind: mlp` 使用 `hidden_dims: [128,128]`、`output_dim: 128`、
`layer_norm: false`；`kind: gru/lstm` 只使用 `hidden_dim: 128`、
`num_layers: 1`，不能把 MLP 字段混入循环网络。

MAPPO 等普通 MLP 的旧顶层 `hidden_dim=64` 对应 `backbone.output_dim=64`，中间层仍是
`[128,128]`。MAAC critic 的旧宽度同时对应 embedding.output_dim 和 backbone.output_dim，
旧 layer_norm 也应迁移到这两个配置。旧字段会报错，不保留兼容接口。
算法 YAML 保持 schema 2，环境 schema 仍为 1；trainer checkpoint 独立使用 schema 5。
所有离策略 policy/critic 拒绝跨环境步循环 backbone，因为尚无序列 replay；
它们支持每次从零状态编码完整窗口的 history encoder，不需要序列 replay。

`update.sequence_length: null` 在循环网络中使用整段 rollout；正整数表示连续
片段长度，不得超过 horizon 或 mini_batch_size。没有跨步循环 backbone 时不填写非空值，
即使 encoder 内部使用 LSTM/GRU 也一样。
mini_batch_size 按时间位置计数，不乘智能体数，循环路径包含补齐位置。
完整用法、截断反向传播近似及 checkpoint 迁移见
[循环 MAPPO 说明](../../docs/recurrent_mappo.md)。
