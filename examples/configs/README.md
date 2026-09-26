# YAML 配置目录

每个 YAML 只承担一种职责：

- `algorithms/<algorithm>.yaml`：一份完整算法 recipe；
- `environments/<environment>.yaml`：环境动力学、设备和动作编码；
- seed、device、并行环境数量、输出目录等运行参数由实验入口或命令行管理。

`algorithm` 是类型校验标签，不是可以单独切换的开关。切换算法时应选择另一份完整
算法 YAML，而不是只修改该字符串。算法 YAML 不允许填写 `num_agents`、观测维度、动作
维度或 state 维度；这些值由环境构建后的 `EnvironmentSpec` 注入。

## 算法 YAML 顶层字段

| 字段 | 必填 | 含义 |
|---|---:|---|
| `schema_version` | 是 | 当前只能为 `1` |
| `algorithm` | 是 | `mappo`、`maac`、`maddpg`、`masac` 或 `qmix` |
| `policy` | 是 | 局部观测到动作或局部 Q 的策略拓扑 |
| `critic` | 是 | value、Q、attention critic 或 mixer |
| `objectives` | 是 | 非空目标函数列表；算法装配会检查所需类型 |
| `returns` | 是 | GAE 或 TD target 估计器 |
| `experience` | 是 | fresh rollout 或 replay 配置 |
| `update` | 是 | optimizer、epoch、mini-batch 和梯度裁剪配置 |
| `target_update` | 是 | `none`、`soft` 或 `hard` |

每个组件都包含必填的 `type` 和可选的 `options`。省略某个 option 时使用下表默认值；
未知 option 会立即报错。

## 内置组件的全部可选字段

| 类别/type | 可选字段与默认值 |
|---|---|
| policy `independent_discrete` | `hidden_dim=128` |
| policy `independent_deterministic` | `hidden_dim=128` |
| policy `independent_gaussian` | `hidden_dim=128` |
| policy `shared_discrete_q` | `hidden_dim=128` |
| critic `centralized_value` | `hidden_dim=128` |
| critic `attention_q` | `hidden_dim=128`, `attention_heads=4` |
| critic `independent_centralized_q` | `hidden_dim=128` |
| critic `twin_independent_centralized_q` | `hidden_dim=128` |
| critic `qmix_mixer` | `mixing_dim=32` |
| objective `ppo_clip` | `clip_ratio=0.2` |
| objective `value_mse` | `coefficient=0.5` |
| objective `td_mse` | `coefficient=1.0` |
| objective `entropy` | `coefficient=0.01` |
| objective `counterfactual` | 无 |
| objective `deterministic_policy` | 无 |
| objective `sac_entropy` | `initial_alpha=0.2`, `target_entropy=null` |
| returns `gae` | `gamma=0.99`, `lambda=0.95`, `normalize_advantage=true` |
| returns `td0` | `gamma=0.99` |
| experience `rollout` | `horizon=null`，null 表示环境 horizon |
| experience `replay` | `capacity=100000`, `batch_size=256` |
| update `ppo_update` | `learning_rate=0.0003`, `epochs=4`, `mini_batch_size=256`, `max_grad_norm=10.0` |
| update `off_policy_update` | `learning_rate=0.0003`, `max_grad_norm=10.0`, `amp_dtype=null`；AMP 只允许 `bf16` |
| target `none` | 无 |
| target `soft` | `tau=0.005` |
| target `hard` | `interval=1` |

## 环境 YAML 字段

`schema_version`、`environment`、`action_kind` 和 `options` 均为必填顶层字段。
`options` 内的环境参数都有代码默认值，因此单项均可省略。示例环境文件显式列出全部
可选项，用于充当可阅读模板。`action_kind` 属于环境适配器输出语义：离散算法选择
`discrete`，连续算法选择 `continuous`。

当前 `energy_trading` 环境使用个体奖励，因此不能直接与要求共享团队奖励的 QMIX 配对。
配置加载采用 `yaml.safe_load`，不支持 Python 类路径或动态导入。
