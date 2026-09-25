# MARL Framework

这是一个面向算法研究的本地 MARL 基础框架。结构借鉴 BenchMARL/TorchRL 的分层
思想，但网络、模块、数据协议、更新工具都由本项目实现，后续修改论文算法时不依赖
外部框架内部逻辑。

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

## 分层结构

```text
marl/
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
└── algorithms/
    ├── base.py             # 所有算法共同生命周期
    ├── maac.py
    ├── maddpg.py
    ├── mappo.py
    ├── masac.py
    └── qmix.py
```

此外 `marl/core/` 定义跨层使用的 `MARLBatch` 和统一模型输出。

## 环境适配与训练

环境与算法之间现在有一条固定的数据链：

```text
具体环境 -> EnvironmentAdapter -> EnvironmentStep / Transition
          -> transitions_to_batch() / TensorReplayBuffer.sample()
          -> MARLBatch -> algorithm.optimize()
                |
                +-> adapter.spec -> algorithm_config_from_env()
```

接口定义在 [`marl/envs/base.py`](marl/envs/base.py)。后续换论文环境时，新增一个
`EnvironmentAdapter` 子类，实现 `spec`、`reset` 和 `step`，即可复用现有算法类和
`transitions_to_batch()`。如果新环境的动作类型与某个算法不匹配，配置工厂会直接报错。
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
