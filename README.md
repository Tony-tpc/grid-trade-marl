# MARL：可直接阅读和修改的多智能体算法库

配置与算法实现放在一起，组件配置与组件实现放在一起。通常从要修改的算法文件开始，
查看文件顶部的 Config 和算法构造函数，就能找到网络、loss 和训练参数的落点。

```text
算法 YAML → load_algorithm_config() → typed config
                                   ↓
                         build_experiment(env, config)
                                   ↓
                      Algorithm + optimizer + Trainer
```

## 快速运行

在仓库根目录使用项目虚拟环境：

```powershell
.\.venv\Scripts\python.exe examples\train_mappo.py --rounds 1 --num-envs 2 --device cpu
.\.venv\Scripts\python.exe examples\train_energy_maac.py --episodes 1 --num-envs 8 --device cpu
.\.venv\Scripts\python.exe examples\minimal_usage.py
.\.venv\Scripts\python.exe examples\custom_component.py
```

算法配置使用 `--config`，环境配置使用 `--environment`。两份 YAML 独立：
[算法与环境全部字段](examples/configs/README.md)。
能源示例使用合成数据，验证训练链路；论文实验还需要真实数据、baseline 和多 seed 评估。

```python
from marl.algorithms import MAPPOConfig
from marl.config import load_algorithm_config
from marl.envs import build_energy_trading_adapter, load_environment_config
from marl.experiment import build_experiment

env = build_energy_trading_adapter(
    load_environment_config("examples/configs/environments/energy_trading.yaml")
)
config = load_algorithm_config("examples/configs/algorithms/mappo.yaml")
assert isinstance(config, MAPPOConfig)
experiment = build_experiment(env, config, device="cpu", seed=42)
metrics = experiment.trainer.train_rollout([42])
```

`build_experiment` 接受单个 adapter 或 `SyncVectorEnv`。离策略实验还可只传
`EnvironmentSpec`，由外部采样代码调用 trainer 的 `record(transition)` 和 `update()`。
MAPPO 需要可交互的环境来采集 fresh rollout。

## 模块与修改导航

| 位置 | 职责 | 通常何时修改 |
|---|---|---|
| `marl/algorithms/<algorithm>.py` | 算法 Config、loss 参数、组件装配及特有前向公式 | 调整一个算法的组成、增加 loss 接线 |
| `marl/modules/policy.py` | 多智能体策略、参数共享关系、对应 Config | 更换策略拓扑 |
| `marl/modules/actor.py` | 单智能体 backbone + action head，采样和动作评估 | 修改单智能体执行组合 |
| `marl/modules/action_head.py` | 动作分布、mask、log-prob、entropy | 增加动作分布 |
| `marl/modules/critic.py`、`mixer.py` | 价值网络与对应 Config | 更换 critic/mixer 网络 |
| `marl/models/` | MLP、GRU、GNN、Transformer 表示学习 | 换 backbone |
| `marl/objectives.py` | 纯数学目标与 `LossBundle` | 增加可复用目标函数 |
| `marl/returns.py` | GAE、TD0 及配置、估计器能力协议 | 修改 advantage/target 估计 |
| `marl/target_updates.py` | hard/soft target 更新及配置 | 修改目标网络同步 |
| `marl/training/on_policy.py` | RolloutBuffer、PPOUpdatePlan、OnPolicyTrainer | 修改 fresh rollout 与 PPO 更新 |
| `marl/training/off_policy.py` | replay 配置、OffPolicyUpdatePlan、OffPolicyTrainer | 修改离策略更新与 checkpoint |
| `marl/training/gradients.py` | 临时冻结网络参数但保留输入梯度 | 调整 actor/critic 梯度边界 |
| `marl/runtime.py` | 设备选择、向量环境、TensorReplayBuffer | 修改运行设施 |
| `marl/envs/` | 环境、adapter、EnvironmentSpec、独立环境配置 | 数据、奖励、物理规则、动作编码 |
| `marl/core/` | MARLBatch、MARLModelOutput | 修改跨模块数据约定 |
| `marl/config.py` | 安全 YAML 解析与配置序列化 | 新增 YAML 可选组件/算法 |
| `marl/experiment.py` | 唯一实验装配入口 | 新增训练生命周期 |
| `marl/extensions.py` | 外部组件的可选 ExtensionCatalog | 接入库外 Python 组件 |
| `tests/` | 数学、形状、梯度、配置、续训与迁移回归 | 每次机制变更 |

直接算法继承关系保持 `BaseMARLAlgorithm → 具体算法`。trainer 不按算法名称分支。
`returns.py` 是数学估计器，既供 on-policy 也供 off-policy 使用，保留在算法公共层。

[算法概念与张量说明](marl/algorithms/README.md) ·
[环境与 adapter 指南](marl/envs/README.md) · [后续开发约定](AGENTS.md)

## 调参数与替换组件

改超参数可以直接编辑对应算法 YAML，也可以使用不可变 dataclass：

```python
from dataclasses import replace
from marl.algorithms import MAPPOConfig
from marl.algorithms.mappo import MAPPOLossConfig
from marl.modules.policy import IndependentDiscreteConfig
from marl.training.on_policy import PPOUpdateConfig

config = MAPPOConfig(
    policy=IndependentDiscreteConfig(hidden_dim=64),
    loss=MAPPOLossConfig(clip_ratio=0.1),
    update=PPOUpdateConfig(learning_rate=0.001, epochs=2),
)
config = replace(config, update=replace(config.update, mini_batch_size=64))
```

`kind` 选择网络实现，后面的字段直接就是该配置的参数。每个算法有独立 schema：
QMIX 填 `mixer`，MAPPO 填 `advantage/rollout`，离策略算法填
`value_target/replay/target_update`。省略可选字段时使用对应 Config 中的默认值。

环境尺寸不能写进算法 YAML；由 `env.spec` 注入。
`algorithm` 与整份配置结构匹配，切换算法时换成相应 YAML。

### 新增仓库内的网络

1. 在对应 module 中实现 `nn.Module`，实现当前算法实际需要的方法。
2. 在同一文件定义冻结 Config，提供 `kind` 和 `build(spec)`。
3. Python 中直接把配置传给算法 Config 的 policy/critic/mixer 字段。
4. 需要 YAML 时，在 `config.py` 的对应分支显式增加解析选择，更新字段文档。
5. 用数学、形状、梯度和 checkpoint 测试验证。

不要求继承某个具体内置网络。MAPPO 的 policy 需要 `act/evaluate/logits`，
MADDPG/MASAC 的 policy 需要 `act`，QMIX 的 policy 需要 `act/q_values`。
所有可训练网络必须是 `nn.Module` 并把参数挂在自身子模块中。
MADDPG 的 critic 暴露逐智能体 `critics`；MASAC 的 twin-Q 暴露 `first/second`，
算法据此保留各智能体 actor 的梯度边界。

### 新增数学机制或算法

新 objective 放进 `objectives.py`，然后在对应算法的 loss 配置和构造函数中接线。
新 advantage/TD target 放进 `returns.py`，实现对应估计器接口，并在算法配置和
YAML 解析分支中添加选择。更换 backbone 则修改或新增网络 Config 的 `build(spec)`。

这类改动允许修改现有算法的少量组件接线。只有新机制确实需要不同前向或训练生命周期，
才增加一个直接继承 `BaseMARLAlgorithm` 的算法类。新增算法文件同时包含其 Config，
再更新公共导出、`AlgorithmConfig` union、YAML 解析和实验构建入口的类型声明。
离策略算法只要满足现有 trainer 能力即可复用通用更新；新生命周期才新增 trainer 分支。

### 外部组件目录

内置组件不注册。只有希望通过 YAML 使用仓库外组件时才建立 `ExtensionCatalog`。
完整可运行示例见 [custom_component.py](examples/custom_component.py)，其核心是：

```python
catalog.register(
    "critic", "tanh_value",
    parse=read_settings,
    build=TanhValueConfig.build,
    action_kinds=frozenset({ActionKind.DISCRETE}),
)
config = load_algorithm_config("custom_mappo.yaml", catalog=catalog)
experiment = build_experiment(env, config)
```

catalog 在加载 YAML 时绑定扩展配置，`build_experiment` 直接消费已解析配置，
无需再次查目录。parse 负责库外参数的类型、范围和未知字段检查；
build 接收解析后的配置与 EnvironmentSpec，返回满足能力接口的 `nn.Module`。
目录按 policy/critic/mixer 分类，拒绝重名；构造前检查动作和奖励兼容性。
YAML 不能填写 Python 类路径，也不会触发动态导入或隐式全局注册。

## 环境修改

新增 `EnvironmentAdapter`，实现 `spec`、`reset(seed)`、`step(actions)`，
并对返回值调用 `validate_step()`。数据集、物理约束、结算规则和动作编解码留在环境中。
新环境的 YAML 建模与构造入口放在环境层；当前 `envs/config.py` 明确构造能源环境。

- observations：`[*B,N,O]`；state：`[*B,S]`。
- 离散 actions：`[*B,N]`；连续 actions：`[*B,N,A]`。
- rewards/value/advantage/return/log-prob：`[*B,N]`。
- action mask：`[*B,N,A]`，True 表示合法；连续动作没有 mask。
- terminated 阻止 bootstrap；truncated 保留当前 bootstrap，但停止跨 episode 的 GAE 递推。
- shared reward 必须由环境明确声明；QMIX 不能用于一般和个体收益任务。

内置 policy 当前是同构、固定智能体数的 MLP 网络。已有 GRU/GNN/Transformer backbone
不代表序列缓冲、hidden state、混合动作或异构智能体的完整训练链已实现。

## Checkpoint 与验证

```python
experiment.trainer.save_checkpoint("checkpoints/run.pt")
restored = build_experiment(env, config, seed=42)
restored.trainer.load_checkpoint("checkpoints/run.pt")
```

checkpoint 保存模型、optimizer、规范化 config 和更新计数；off-policy 还保存 replay、
NumPy RNG，on-policy 保存 mini-batch generator；两者保存 Torch CPU/CUDA RNG。
配置不一致会拒绝加载。恢复外部组件需要先用同一 catalog 解析配置，代码本身不写入 checkpoint。

MAPPO 在完成一次更新后保存，恢复后按新的 reset seeds 采集下一份 rollout。
checkpoint 不保存采样中的外部环境状态；离策略外部采样器的进度由实验代码管理。

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp=.pytest_cache/refactor-tmp
.\.venv\Scripts\python.exe -m ruff check marl examples tests
.\.venv\Scripts\python.exe -m mypy marl examples tests
git diff --check
```

固定输入回归保存五个算法重构前的动作、value/Q 和 loss；还验证 on-policy checkpoint
恢复后的下一次更新逐张量一致。训练曲线不能替代数学单测。

## API 迁移

旧 recipe、factory、registry、assembly 和 `from_recipe()` 接口已删除，没有兼容层。
使用 `load_algorithm_config()`、算法专用 Config 与 `build_experiment()`。
底层直接构造使用 `MAPPO(env.spec, MAPPOConfig())` 等形式。
环境入口改为 `load_environment_config()`；示例参数改为 `--config`。
旧 checkpoint 的 recipe 元数据不能用于新配置接口。
