# 接入新环境：从接口到第一次训练

[首页](README.md) · [数据形状](architecture.md) · [模块扩展](extensions.md) · [训练](training.md)

## 两种“新开环境”

若指安装新的 Python 虚拟环境，见 [安装](training.md#start)。若指换一个强化学习任务，
需要写 EnvironmentAdapter。本章处理后者；不把环境维度写死进算法。

## 先确定任务契约

先写明玩家是否固定、观测是否同维、动作离散还是连续、每个奖励的含义、何时真正终止、
何时仅因时间限制截断。当前网络要求固定 N 和同构输入输出，异构观测需在 adapter 明确
补齐并解释语义。当前固定长度采集器不支持中途部分 reset。

| 决策 | 对应字段 | 检查 |
|---|---|---|
| 玩家数和维度 | EnvironmentSpec 的 N/O/A/S | reset/step 全程一致 |
| 动作空间 | ActionKind.DISCRETE/CONTINUOUS | 离散 int64 [N]；连续 float [N,A] |
| 个体还是共享收益 | RewardStructure | SHARED 要求每人奖励相同 |
| 时间长度 | spec.horizon | 不能用一个 done 混淆终止和截断 |
| 全局信息 | EnvironmentStep.state | 只进入集中式 critic/mixer，不泄漏给 actor |
| 历史语义 | spec.observation_history | 显式 [W,C] 特征索引与当前特征索引 |
| state 节点及图 | spec.state_layout、spec.adjacency | 节点数等于 N，固定非负有限邻接 |
| 动作约束 | action_mask | 每个玩家至少一个 True；连续为 None |

`validate_step` 检查形状、有限值、共享奖励一致性与 mask。它并不替代你对动作范围、
所有 dtype、物理约束和数据泄漏的校验。

历史/图布局的可运行声明与形状例子见[网络教程](networks.md)。
普通 MLP 无需声明布局；不能根据 S=N×O 或扁平特征长度猜测时间或节点语义。
窗口必须按真实时间顺序构造，索引校验无法识别业务上的未来数据泄漏；adapter 需明确
窗口是否含当前时刻以及 episode 开始时怎样填充。修改索引会改变 checkpoint input_spec，
即使观测总维度没变，也不能直接按原布局续训。

## 最小实现与运行

完整示例：[cue_environment.py](examples/cue_environment.py)。这是教学用可见提示任务：
每个玩家观察一个二值提示，选择 0/1，匹配自己提示得 +1，否则 -1。奖励独立，固定 horizon
后以 time-limit truncation 结束；没有真实数据、市场机制或论文复现实验的含义。

它逐个实现以下接口：

| 函数 | 输入 | 输出与副作用 |
|---|---|---|
| `__init__` | horizon | 配置 spec、创建 RNG 和状态；不训练 |
| `spec` | 无 | EnvironmentSpec(2,2,2,4,DISCRETE,horizon) |
| `_snapshot` | rewards [2]、结束标记 | validate_step 后的 EnvironmentStep |
| `reset` | 可选 seed | 初始观测，零奖励，清除 episode 状态 |
| `step` | int64 动作 [2] | 先按当前提示算 reward，再推进时间和提示 |

从仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe doc\examples\train_walkthrough.py --algorithm mappo --rounds 2
.\.venv\Scripts\python.exe doc\examples\train_walkthrough.py --algorithm maac --rounds 2
```

示例只要求基本项目依赖。它验证接口、采样、一次更新、有限指标及 checkpoint 恢复，
不以两轮训练证明策略收敛。

## 改成你的真实任务

1. 从原环境获取 observation 和 reward，按固定 agent 顺序整理，记录顺序映射。
2. 明确 state 是原环境全局状态还是观测拼接；不能暗中放入未来收益或测试数据。
3. 在 reset(seed) 中重置所有动态量，并将 seed 传给原环境自己的 RNG。
4. 在 step 中验证动作，映射为原环境动作，执行一步，然后组装下一结果。
5. rewards 读这次动作造成的收益，每个分量有独立测试。
6. 返回前调用 validate_step，离散 mask 包括最终观测所需的合法占位/下一动作语义。
7. Python 直接 `build_experiment(adapter, config)`；多个独立 adapter 包进 SyncVectorEnv。
8. 稳定接口加入 [envs/__init__.py](../marl/envs/__init__.py)，再补相应配置与测试。

连续策略的标准动作常是 [-1,1]。真实控制量若在 [low,high]，在 adapter 中执行
`low + (a+1)/2*(high-low)`，replay 仍保存算法标准动作 a；否则 critic 训练看到的动作与
actor 输出空间不一致。变换的物理范围、可行性和裁剪策略要在 adapter 测试中验证。
离散量化动作的编码表也属于 adapter，不能藏入算法 loss。

## YAML 接入不是改名字即可

当前 [envs/config.py](api/marl-envs-config.md) 分别提供能源和 MPE2 的显式构造入口；
`train_mappo.py --environment` 仍调用能源 adapter 构造器，不能直接给任意环境 YAML
就期望自动切换。新环境可先用 Python 直接构造；需要 YAML 时再在环境层增加其数据配置、
严格解析和显式构造函数，最后在实验入口选用。算法 YAML 保持独立。

例如 MPE2 YAML 应搭配 `build_mpe2_simple_adversary_adapter(config)`，
而不是能源构造器。MPE2 是一个现成外部环境范例：[adapter](api/marl-envs-mpe2_simple_adversary_adapter.md)。
它负责固定 agent 顺序、观测补齐、动作范围转换和终止解释。基准入口见
[benchmarks](../benchmarks/README.md)。MemoryCueAdapter 则用于验证历史信息是否真正被记住，
与本章可见提示教学任务不同。

## 接入验收

- 固定 seed 的 reset/step 可复现，所有 NumPy 数组 shape/dtype/有限值正确。
- mask 对应动作确实可执行，非法动作早报错，连续 action_mask=None。
- reward 分量与手算一致；共享奖励由任务定义，不通过训练器平均伪造。
- terminated 时不 bootstrap，truncated 使用最终观测 bootstrap。
- MARLBatch 的 rewards 来自 next；next_state/next_mask 也与 next_observations 对齐。
- 最小采集更新和 checkpoint 恢复通过，再开始长实验。

可参照 [环境回归](../tests/test_environment_adapter.py)、[MPE2 回归](../tests/test_mpe2_adapter.py)
和 [on-policy 测试](../tests/test_on_policy.py)。环境适配器不能返回 algorithm 专用 advantage
或 old_log_prob；它们应由训练采集链计算。
