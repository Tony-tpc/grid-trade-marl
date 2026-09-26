# 环境适配接口

核心目标：以后更换论文环境，算法 `maac.py`、`maddpg.py`、`mappo.py`、
`masac.py`、`qmix.py` 保持原样。环境层负责把研究问题表达为算法能理解的数据。

## 当前论文环境的建模依据

参考 Ye et al., *A Scalable Privacy-Preserving Multi-Agent Deep Reinforcement Learning
Approach for Large-Scale Peer-to-Peer Transactive Energy Trading*, IEEE Transactions on
Smart Grid, 2021。论文第 II-IV 节说明设备、市场和 Markov game；第 VI 节说明
48 个半小时时段、历史 24 小时观测和数据来源。

本环境实现的对象：

| 对象 | 当前实现 |
|---|---|
| 智能体 | 一户 prosumer 对应一个 agent |
| 局部观测 | 时段、过去价格/负荷/PV、EV/储能电量、EV 可用性、室内外温度、智能电器状态、设备拥有情况 |
| 全局状态 | 按论文设为所有局部观测的拼接；只交给集中式 Critic/Mixer |
| 动作 | EV/储能充放电、HVAC 功率、智能电器启动 |
| 市场 | mid-market-rate (MMR) 本地买卖价，社区差额与供电商交易 |
| 状态转移 | 电池 SoC、室内温度、智能电器不可中断工作周期 |
| 奖励 | 负电费 + 舒适度惩罚 + EV 出行惩罚 + 配网峰值惩罚 |
| episode | 48 个半小时交易时段，对应一天 |

**本论文是个体收益博弈**：每户得到自己的 `r_i`，策略 `pi_i` 追求自己的 `J_i`。
MAAC/MADDPG/MASAC/MAPPO 保留 `[B,N]` 的逐智能体奖励与价值；QMIX 要求
所有智能体共享同一团队奖励，算法配置的 validate 会拒绝将其直接用于本论文环境。

默认外生曲线是合成数据。论文使用的 Ausgrid 负荷/PV、室外温度和零售电价，
以及家庭参数分布，并未包含在本仓库。因此这里只是**论文启发的基础环境**，
不是数值复现。真实曲线可构造 `EnergyProfiles` 注入：

```python
profiles = EnergyProfiles(
    demand_kw=demand,  # [48, N]
    pv_kw=pv,  # [48, N]
    buy_price=buy_price,  # [48]
    sell_price=sell_price,  # [48]
    outdoor_c=outdoor,  # [48]
)
environment = EnergyTradingEnv(EnergyTradingConfig(num_agents=N), profiles)
adapter = EnergyTradingAdapter(environment, ActionKind.DISCRETE)
```

当前 `EnergyTradingConfig` 支持逐家庭设置是否拥有 EV/储能/HVAC/智能电器。
其余设备容量、效率和时间窗口暂为社区共同参数；若研究要精确建模异质家庭，
应只扩展环境参数和转移方程，不改算法对 `MARLBatch` 的读取方式。
本实现没有加密通信或隐私保护协议；集中训练的 global state 会包含各家庭观测。
当前示例适合少量家庭的接口验证，不代表已具备论文 300 户的计算扩展性。

可运行示例把环境参数独立保存在
[`examples/configs/environments/energy_trading.yaml`](../../examples/configs/environments/energy_trading.yaml)。
该文件显式列出 `EnergyTradingConfig` 的全部可选字段；加载时使用 `yaml.safe_load`，
未知顶层字段和未知环境 option 都会立即报错。算法网络、loss、optimizer 和 replay 参数
必须放在独立的算法 YAML 中，不能混入环境文件。

## 适配器边界

`EnvironmentStep` 统一返回：

- `observations [N,O]`：局部观测；
- `state [S]`：训练阶段可用的全局状态；
- `rewards [N]`：每个 agent 的奖励；
- `terminated` / `truncated`：自然终止 / 时间限制；
- `action_mask [N,A]`：离散算法可用动作，连续算法为 `None`。

`Transition(current, actions, next)` 记录一步经验；
`transitions_to_batch([...])` 将多步经验堆叠为 `MARLBatch`。
具体适配器在返回时调用 `validate_step()`，会检查观测、状态、奖励和动作掩码维度。
`MAAC(adapter.spec, MAACConfig())` 从 EnvironmentSpec 注入智能体数、观测、
动作和全局状态维度。训练器和算法不读取 `EnergyTradingEnv` 的内部字段。

若新论文环境的观测编码、奖励公式或结算规则不同，实现新的具体环境与适配器即可。
若动作空间从离散变为连续，需要选择与动作类型匹配的算法；算法配置的 validate 会检测这点。
若新论文需要可变数量智能体、异质动作维度或混合动作分布，则还需扩展通用模型能力；
仅做数据格式转换不足以改变算法的数学定义。

## 换环境的最小模板

```python
class NewPaperAdapter(EnvironmentAdapter):
    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            num_agents=...,
            observation_dim=...,
            action_dim=...,
            state_dim=...,
            action_kind=ActionKind.DISCRETE,
            horizon=...,
        )

    def reset(self, seed=None) -> EnvironmentStep:
        # 调用新环境 reset，并整理为统一形状。
        ...

    def step(self, actions) -> EnvironmentStep:
        # 解码标准动作，调用新环境 step，再统一返回形状。
        ...
```

完成后，现有训练代码仍通过 `adapter.spec` 创建配置，再把 transition 转为
`MARLBatch`。需要注意 `MAPPO` 为 on-policy，还需 rollout/GAE 计算
`old_log_prob`、`advantages` 和 `returns`；环境适配层本身不生成这些算法量。

环境 YAML 使用 `load_environment_config()` 读取，由 `build_energy_trading_adapter()` 构造。
算法和 trainer 使用 `build_experiment(adapter, algorithm_config)` 一次装配。
