# 配置参考

每个算法一份 YAML：`algorithms/{mappo,maac,maddpg,masac,qmix}.yaml`。
环境独立保存在 `environments/energy_trading.yaml` 或
`environments/mpe2_simple_adversary.yaml`。
seed、device、并行环境数和输出路径由脚本/命令行管理。

## 公共规则

算法 YAML 仅 `schema_version: 1` 和 `algorithm` 必填；其余节及字段可省略，
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
| MAPPO/MAAC policy | independent_discrete | hidden_dim=128 |
| MADDPG policy | independent_deterministic | hidden_dim=128 |
| MASAC policy | independent_gaussian | hidden_dim=128 |
| QMIX policy | shared_discrete_q | hidden_dim=128 |
| MAPPO critic | centralized_value | hidden_dim=128 |
| MAAC critic | attention_q | hidden_dim=128，attention_heads=4 |
| MADDPG critic | independent_centralized_q | hidden_dim=128 |
| MASAC critic | twin_independent_centralized_q | hidden_dim=128 |
| QMIX mixer | qmix_mixer | mixing_dim=32 |

hidden_dim/mixing_dim 必须为正整数；attention_heads 必须为正整数且整除 hidden_dim。
观测维 O、动作维 A、智能体数 N、全局状态维 S 都由环境 spec 注入。

## 数学与训练参数：全部选项

| 节/算法 | 字段与默认值 | 含义与约束 |
|---|---|---|
| loss / MAPPO | clip_ratio=0.2 | PPO 概率比裁剪宽度，(0,1) |
| loss / MAPPO | value_coefficient=0.5 | value MSE 权重，>=0 |
| loss / MAPPO、MAAC | entropy_coefficient=0.01 | 熵奖励系数，>=0 |
| loss / 所有离策略算法 | td_coefficient=1.0 | critic TD MSE 权重，>=0 |
| loss / MASAC | initial_alpha=0.2 | 初始逐智能体温度，>0 |
| loss / MASAC | target_entropy=null | null 使用 -action_dim；也可填写有限数值 |
| advantage / MAPPO | kind=gae，gamma=0.99 | 折扣因子，[0,1] |
| advantage / MAPPO | lambda=0.95 | GAE 递推系数，[0,1]；Python 名为 gae_lambda |
| advantage / MAPPO | normalize=true | advantage 的 batch 标准化，不改变 return |
| value_target / 离策略 | kind=td0，gamma=0.99 | 一步 target，仅 terminated 阻止 bootstrap |
| rollout / MAPPO | horizon=null | null 使用环境 horizon；指定时为 1..环境 horizon |
| replay / 离策略 | capacity=100000 | ring buffer 最大 transition 数，正整数 |
| replay / 离策略 | batch_size=256 | 每次采样数，1..capacity |
| update / 全部 | learning_rate=0.0003 | Adam 学习率，>0 |
| update / 全部 | max_grad_norm=10.0 | >0 的梯度上限；null 关闭 |
| update / MAPPO | epochs=4 | 同一 fresh rollout 更新轮数，正整数 |
| update / MAPPO | mini_batch_size=256 | 展平时间/环境维后的批大小，正整数 |
| update / 离策略 | amp_dtype=null | null=FP32；bf16 仅支持相应 CUDA 设备 |
| target_update / 离策略 | kind=soft，tau=0.005 | 每次更新后 Polyak 同步，tau∈(0,1] |
| target_update / 离策略 | kind=hard，interval=1 | 可替代 soft；每 interval 次更新完整同步，正整数 |

hard 和 soft 的字段不能混写。MAPPO 不含 target_update。
YAML 可使用 gae_lambda 代替 lambda，但不能同时填写两者。

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
