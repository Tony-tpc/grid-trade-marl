# MARL Framework

这是一个面向算法研究的本地 MARL 基础框架。结构借鉴 BenchMARL/TorchRL 的分层
思想，但网络、模块、数据协议、更新工具都由本项目实现，后续修改论文算法时不依赖
外部框架内部逻辑。

本 README 面向使用者和首次维护者，说明模块职责以及“应该在哪里修改”。如果要让
自动化 Agent 或后续开发者严格实施一篇新论文，请同时阅读
[`AGENTS.md`](AGENTS.md)：其中包含论文审计、扩展决策、张量语义、测试和 Git 门禁。

## 环境

根目录已创建 Python 3.11 虚拟环境 `.venv`，并安装 `requirements-dev.txt`。PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest
python examples\minimal_usage.py
```

若系统禁止执行激活脚本，可直接使用 `.\.venv\Scripts\python.exe`。

### 依赖

- `torch`：模型、自动求导、优化器（核心必需）。
- `numpy`：环境数据和数值处理。
- `gymnasium`：标准环境接口；不侵入模型/算法层。
- `PyYAML`：后续实验配置。
- `tensorboard`：训练指标。
- `pytest`、`pytest-cov`、`ruff`、`mypy`：开发和验证工具。

当前刻意不安装 `torchrl`、`tensordict`、`benchmarl`、`torch-geometric`。如需对接，
应添加 adapter；算法核心仍保留在本地。

## 完整模块地图

```text
marl/
├── core/
│   ├── batch.py            # 统一训练数据 MARLBatch
│   └── output.py           # 统一模型输出 MARLModelOutput
├── envs/
│   ├── base.py             # EnvironmentSpec/Step/Adapter
│   ├── config.py           # 旧算法的环境配置工厂
│   └── *_adapter.py        # 环境动作与公共动作协议之间的转换
├── models/
│   ├── base.py             # 所有 Backbone 的抽象接口
│   ├── mlp.py              # 向量观测 MLP
│   ├── gru.py              # 部分可观测/序列状态
│   ├── gnn.py              # 本地消息传递，不依赖 PyG
│   └── transformer.py      # agent/time token 建模
├── modules/
│   ├── actor.py            # Backbone + ActionHead
│   ├── critic.py           # 独立/集中式 Critic
│   ├── mixer.py            # VDNMixer / QMixer
│   └── action_head.py      # 离散与 tanh-Gaussian 动作分布
├── algorithms/
│   ├── base.py             # 所有算法共同生命周期
│   ├── maac.py
│   ├── maddpg.py
│   ├── mappo.py
│   ├── masac.py
│   └── qmix.py
├── training/
│   └── on_policy.py        # RolloutBuffer、PPOUpdatePlan、OnPolicyTrainer
├── components.py           # 六类平级组件协议
├── policies.py             # 可复用策略拓扑
├── objectives.py           # PPO/value/entropy 目标与 LossBundle
├── returns.py              # GAE 等 return estimator
├── registry.py             # 显式分类注册表
├── recipes.py              # Python/YAML recipe 和编译校验
├── builtins.py             # 内置组件的严格构造工厂
└── runtime.py              # 设备、向量环境、离策略 replay/运行设施
```

仓库根目录的其他部分：

| 路径 | 含义 |
|---|---|
| `examples/` | 可直接运行的最小训练脚本和 YAML recipe |
| `tests/` | 数学、形状、语义、checkpoint 和端到端回归 |
| `marl/envs/README.md` | 环境与 adapter 的详细说明 |
| `marl/algorithms/README.md` | MARL/PyTorch 概念、张量形状和算法差异 |
| `AGENTS.md` | 后续论文实现和代码审查的强制工程约定 |

### 各层如何协作

```text
EnvironmentAdapter.spec ──> EnvironmentSpec ──> compile recipe
          │                                      │
          ▼                                      ▼
 EnvironmentStep ──> rollout/replay ──> MARLBatch ──> Algorithm
                                                   │
               policy + critic + objectives <──────┤
                                                   ▼
                                    UpdatePlan / optimizer
```

- `EnvironmentSpec` 描述静态能力与尺寸，不包含训练超参数。
- `MARLBatch` 是环境、缓冲区、算法和 trainer 之间的公共数据语言。
- model 只做表示学习，module 负责 Actor/Critic/Mixer 等网络角色。
- component 表达可替换机制，recipe 只选择组件与超参数。
- algorithm 是薄装配类；trainer/update plan 管理数据生命周期和参数更新。

## 修改导航：先判断改哪一层

| 需求 | 首选修改位置 | 通常不需要做的事 |
|---|---|---|
| 调学习率、clip ratio、hidden size | Python/YAML recipe | 新建算法类 |
| 换已有 policy/critic/objective 的组合 | recipe | 复制 `mappo.py` |
| 新增 loss 或正则项 | `objectives.py` + registry | 覆写整套训练循环 |
| 新增 GAE/V-trace/n-step | `returns.py` + registry | 修改环境 |
| 换 MLP 为 GRU/GNN/Transformer | `models/`、policy factory | 修改奖励 |
| 新动作分布或参数共享拓扑 | `modules/action_head.py`、`policies.py` | 把编码写入 trainer |
| 新市场/机器人/博弈环境 | `envs/` + adapter | 修改通用算法 |
| 新 replay/rollout 数据生命周期 | experience source、`training/` | 按算法名硬编码分支 |
| 新 optimizer 更新顺序 | UpdatePlan | 在算法中调用 `optimizer.step()` |
| 全新且无法组合表达的算法 | `algorithms/` | 增加二级算法基类 |

最重要的判断规则是：**只重新组合已有机制时写 recipe；只有出现新的数学机制时才写
组件；只有组件和训练计划都无法表达时才写新的具体算法类。**

## 公共数据约定

| 数据 | 形状 | 说明 |
|---|---|---|
| observations | `[*B,N,O]` | 每个智能体的局部观测 |
| state | `[*B,S]` | 集中训练使用的全局状态 |
| discrete actions | `[*B,N]` | 每个智能体一个动作编号 |
| continuous actions | `[*B,N,A]` | 每个智能体一个动作向量 |
| rewards | `[*B,N]` | 默认保留逐智能体奖励 |
| value/advantage/return | `[*B,N]` | 不跨智能体平均 |
| action mask | `[*B,N,A]` | `True` 表示离散动作合法 |

`*B` 可以包含 batch、并行环境或时间维。自然终止 `terminated` 会阻止 bootstrap；
时间限制 `truncated` 保留当前下一状态的 bootstrap，但不能让 advantage 穿过 episode
边界继续递推。新增算法和环境必须保持这两个标记分离。

## 如何新增环境

新增论文环境时，一般不需要修改算法：

1. 在 `marl/envs/` 实现环境本体或包装第三方环境；
2. 实现 `EnvironmentAdapter.spec/reset/step`；
3. 在 adapter 中完成动作编解码、观测整理和奖励语义转换；
4. 每次返回 `EnvironmentStep` 前调用 `validate_step()`；
5. 明确奖励是 `INDIVIDUAL` 还是 `SHARED`；
6. 增加 reset、step、seed、动作边界、奖励和终止测试。

最小结构：

```python
class NewPaperAdapter(EnvironmentAdapter):
    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            num_agents=self.environment.num_agents,
            observation_dim=self.environment.observation_dim,
            action_dim=self.environment.action_dim,
            state_dim=self.environment.state_dim,
            action_kind=ActionKind.DISCRETE,
            horizon=self.environment.horizon,
            reward_structure=RewardStructure.INDIVIDUAL,
        )

    def reset(self, seed=None) -> EnvironmentStep:
        return self.validate_step(self._convert(self.environment.reset(seed=seed)))

    def step(self, actions) -> EnvironmentStep:
        native_actions = self._decode_actions(actions)
        return self.validate_step(self._convert(self.environment.step(native_actions)))
```

环境尺寸只能来自 `adapter.spec`，不能在 recipe 中手写。异质动作、混合动作或可变数量
智能体不是简单 adapter 转换就能解决的问题，还需要相应 policy/action head 能力。

## 如何新增算法变体

### 情况一：只需要新 recipe

如果需要的组件都已存在，复制并修改
[`examples/configs/mappo.yaml`](examples/configs/mappo.yaml)，不要复制算法源文件。
YAML 使用 `yaml.safe_load`，只能填写显式注册名称，且 schema 当前为版本 1。

### 情况二：需要新组件

以新增 objective 为例：

1. 在 `marl/objectives.py` 实现独立、可测试的目标函数；
2. 在 `marl/builtins.py` 添加严格 factory，拒绝未知 options；
3. 通过 `register_objective()` 注册稳定名称并声明动作/奖励兼容性；
4. 在 recipe 的 `objectives` 中引用该名称；
5. 测试公式、系数、形状、梯度和非法配置。

policy、critic、return estimator、experience source、update plan 和 target update 遵循
同样流程。带参数的组件使用 `nn.Module`；无参数组件优先使用不可变 dataclass。

### 情况三：确实需要新算法

新算法类必须直接继承 `BaseMARLAlgorithm`，不能继承 MAPPO/MADDPG 再覆盖大段 loss，
也不能创建 `BasePPOAlgorithm` 之类的中间继承层。推荐顺序：

1. 先实现并测试缺少的组件；
2. 新算法类只装配 policy、critic、objectives 和算法特有前向语义；
3. 用 `MARLModelOutput` 返回统一输出，用 `LossBundle` 组合 loss；
4. 把 mini-batch、epoch、优化器和 target 更新顺序放入 UpdatePlan；
5. trainer 依赖能力协议，不使用算法名称判断；
6. 用固定小张量对照论文公式或迁移前实现；
7. 增加一个小环境完整训练及 checkpoint 恢复测试。

当前 MAPPO 是组合式参考实现；MAAC、MADDPG、MASAC、QMIX 仍保留旧
`BaseMARLAlgorithm.optimize()` 入口，迁移时应逐个进行，不要一次重写全部算法。

## 新论文接入推荐流程

1. 提取论文中的状态、动作、奖励、网络、损失、数据来源和更新顺序；
2. 写“已满足 / 可配置复用 / 缺失”对照表；
3. 先实现环境 adapter 和固定张量数学测试；
4. 按 recipe → component → trainer → algorithm 的顺序选择最小扩展层；
5. 每个阶段运行定向测试和完整门禁；
6. 最后再做长训练、性能测试和论文指标对比。

不要用“训练能跑”代替公式验证，也不要把单个 seed 的结果称为复现成功。论文缺少的
参数或数据要显式记录假设。

## 环境适配与训练

环境与算法之间使用相同的 adapter 和公共数据协议，但经验生命周期分成两条：

```text
具体环境 -> EnvironmentAdapter -> EnvironmentStep
                   |
                   +-> off-policy: Transition -> TensorReplayBuffer
                   |               -> MARLBatch -> 旧算法 optimize()
                   |
                   +-> on-policy: RolloutBuffer -> GAEEstimator
                                   -> PreparedRollout -> PPOUpdatePlan

adapter.spec -> EnvironmentSpec -> compile_recipe（组合式链路）
                             └-> algorithm_config_from_env（旧算法兼容链路）
```

接口定义在 [`marl/envs/base.py`](marl/envs/base.py)。后续换论文环境时，新增一个
`EnvironmentAdapter` 子类，实现 `spec`、`reset` 和 `step`，即可复用现有算法类和
训练设施。离策略入口可继续使用 `transitions_to_batch()` 或 `TensorReplayBuffer`；
MAPPO 由 `OnPolicyTrainer` 直接采集 fresh rollout。如果新环境的动作类型与某个算法
不匹配，recipe 编译器或旧配置工厂会直接报错。
固定智能体数量、同构观测和动作是目前算法的前提；突破这些前提需要扩展模型能力。

论文式 P2P 电能交易环境位于 [`marl/envs/energy_trading.py`](marl/envs/energy_trading.py)，
专属动作编码位于 [`marl/envs/energy_trading_adapter.py`](marl/envs/energy_trading_adapter.py)。
它模拟 48 个半小时时段、EV、储能、空调、智能电器、MMR 价格、费用、舒适度、
出行和配网峰值惩罚。默认负荷、PV、电价和室外温度为**合成示例曲线**；真实研究
请提供 `EnergyProfiles` 和准确的设备参数，不应把示例训练结果视为论文复现。

原论文的动作同时包含连续功率和二元启动决策。目前适配器提供两种显式近似：

- `ActionKind.DISCRETE`：54 种预设动作组合，供 MAAC/MAPPO 使用；
- `ActionKind.CONTINUOUS`：4 维动作，智能电器以阈值启动，供 MADDPG/MASAC 使用。

这两种编码便于不改算法源码地接入环境，但都不是论文的精确混合动作策略。

运行一个真实的环境交互与参数更新示例：

```powershell
.\.venv\Scripts\python.exe examples\train_energy_maac.py --episodes 5 --agents 3 --num-envs 16 --device auto
```

示例使用合成曲线、ε-greedy 探索、批量环境推理和预分配 Tensor replay buffer，
并打印各家庭平均回报。设备由 `--device auto|cpu|cuda` 选择；使用 NVIDIA GPU 前，
需按 [PyTorch 官方安装说明](https://docs.pytorch.org/get-started/locally/) 在本项目
虚拟环境安装 CUDA 版 PyTorch。`--amp bf16` 可在支持
BF16 的 CUDA 设备上开启混合精度；默认保持 FP32。通用执行组件位于
`marl/runtime.py`，负责设备选择、并行适配器和离策略经验存储。旧的离策略算法仍可调用
`BaseMARLAlgorithm.optimize()`；MAPPO 的采样、GAE、mini-batch 与多 epoch 更新由
`OnPolicyTrainer` 和 `PPOUpdatePlan` 负责。环境适配器只采集与校验数据，不负责梯度更新。
如果已有与 Python、操作系统及架构兼容的 CUDA wheel，也可以用
`python -m pip install --no-index --no-deps --force-reinstall <wheel路径>`
离线安装；安装后请检查 `torch.cuda.is_available()` 并运行训练测试。
小网络、短实验可能受环境步进和 CPU/GPU 往返限制，是否加速应比较实际吞吐。
请先关注输出的 episode 步数、个体奖励和 loss 是否正常，再接入论文数据。环境设计
细节与更换论文环境的示例见 [`marl/envs/README.md`](marl/envs/README.md)。

如果不熟悉 MARL 或 PyTorch，请先阅读
[`marl/algorithms/README.md`](marl/algorithms/README.md)，其中解释了张量形状、CTDE、
Actor/Critic、PyTorch 梯度、target network 和五种算法之间的区别。

## 组合式算法边界

`BaseBackbone` 负责表示学习；`Actor/Critic/Mixer` 负责网络角色。策略拓扑、目标函数、
return 估计、经验来源和更新计划是平级组件，由强类型 Python recipe 或安全 YAML recipe
组装。具体算法仍直接继承 `BaseMARLAlgorithm`，但算法类只选择并校验组件。

五个算法均直接继承 `BaseMARLAlgorithm`，并且是可实例化的具体实现：

- `MAAC`：独立离散 Actor、共享注意力 Critic、逐智能体 Q 和反事实 baseline。
- `MADDPG`：独立连续 Actor、逐智能体集中式 Critic 和 target network。
- `MAPPO`：独立离散 Actor、集中式逐智能体 V 和 clipped PPO objective。
- `MASAC`：独立 tanh-Gaussian Actor、逐智能体 twin-Q 与温度。
- `QMIX`：仅用于共享团队奖励的合作任务，不适用于个体收益博弈。

MAPPO 已迁移到完整的组合式 on-policy 链路。使用 YAML 配置运行合成环境示例：

```powershell
.\.venv\Scripts\python.exe examples\train_mappo_recipe.py --rounds 2
```

同一链路也可以完全由 Python recipe 构造，环境尺寸仍只从 `env.spec` 注入：

```python
from marl.algorithms import MAPPO, MAPPOConfig, default_mappo_recipe
from marl.training import OnPolicyTrainer

recipe = default_mappo_recipe(MAPPOConfig(hidden_dim=64))
algorithm = MAPPO.from_recipe(env.spec, recipe)
trainer = OnPolicyTrainer.from_recipe(env, algorithm, recipe)
metrics = trainer.train_rollout(seeds=[42])
```

新机制通过显式注册表扩展，注册动作发生在加载或编译使用该名称的 recipe 之前：

```python
from marl.envs import ActionKind
from marl.objectives import EntropyObjective
from marl.registry import register_objective


def build_small_entropy(options, spec):
    coefficient = float(options.get("coefficient", 0.005))
    return EntropyObjective(coefficient=coefficient)


register_objective(
    "small_entropy",
    build_small_entropy,
    action_kinds=frozenset({ActionKind.DISCRETE}),
)
```

Python 中也可以使用 `default_mappo_recipe()`，或者通过 `register_policy()`、
`register_objective()` 等显式注册接口添加通用组件。只重新组合现有机制时应新增 recipe；
只有出现新的数学机制时才新增组件，不复制完整算法类。YAML 不能填写 Python 导入路径，
环境尺寸始终由 `EnvironmentSpec` 注入。

MAAC、MADDPG、MASAC、QMIX 暂时保留原更新入口，后续将按相同组件协议逐个迁移。
