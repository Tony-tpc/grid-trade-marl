# 自定义模块、数学目标和算法

[首页](README.md) · [架构](architecture.md) · [训练](training.md) · [API](api/README.md)

<a id="modules"></a>
## 先选最轻的扩展位置

| 需求 | 应修改 | 是否新增算法 |
|---|---|---|
| 学习率、clip、层宽不同 | 算法 Config/YAML | 否 |
| 同样输入输出换表示 | models backbone 或 modules 网络 | 否 |
| 参数共享关系不同 | policy topology | 否 |
| 分布或动作编码不同 | action head + policy + adapter | 先检查契约 |
| 新 penalty、return 估计 | objectives/returns + 配置接线 | 通常否 |
| 市场规则、数据集、物理约束 | environment/adapter | 否 |
| 根本不同的专用前向/更新顺序 | 具体算法的 update | 组合不足时才是 |
| 新数据生命周期 | buffer/sampler/trainer | 不复制已有优化循环 |

### 用内置 Config 换网络

```python
from marl.algorithms import MAPPOConfig
from marl.models import LSTMBackboneConfig, MLPBackboneConfig
from marl.modules.policy import IndependentDiscreteConfig
from marl.modules.critic import CentralizedValueConfig
from marl.training.on_policy import PPOUpdateConfig, RolloutConfig

config = MAPPOConfig(
    policy=IndependentDiscreteConfig(backbone=LSTMBackboneConfig(hidden_dim=32)),
    critic=CentralizedValueConfig(backbone=MLPBackboneConfig(output_dim=32)),
    rollout=RolloutConfig(horizon=8),
    update=PPOUpdateConfig(epochs=2, mini_batch_size=16, sequence_length=4),
)
```

观测 O、动作 A、玩家 N 来自 env.spec，不能再写一份。循环网络还需要状态传递和连续序列，
不是只把 Linear 替换成 LSTM。运行对应教学例子：

```powershell
.\.venv\Scripts\python.exe doc\examples\train_walkthrough.py --algorithm mappo --recurrent
```

五算法共享 encoder → backbone → 输出头约定。历史 GRU/LSTM/Transformer 放在
encoder.temporal；跨环境步 GRU/LSTM backbone 仅 MAPPO 支持。
集中式 critic/mixer 可配置 graph: GNN。窗口与节点尺寸由 EnvironmentSpec 显式声明，
actor 不读取邻居。完整参数、布局例子和能源 48 步验收见
[统一网络配置](../docs/network_configuration.md)。

底层 backbone Config 使用 `build(input_dim)`，encoder Config 使用
`build(input_dim, layout)`；policy/critic/mixer Config 使用 `build(spec)` 并负责装配。
QMIX mixer 使用超网络字段与 value_backbone，不提供普通 backbone 字段。
可以直接运行的历史编码例子及各组件共享关系见[网络教程](networks.md)。

### 自定义 critic：最短可运行路径

[examples/custom_component.py](../examples/custom_component.py) 的 TanhValueConfig 是完整模板：
`build(spec)` 返回 `nn.Sequential(Linear(S,H), Tanh(), Linear(H,N))`，输入 `[...,S]`，
输出 `[...,N]`。Python 可直接：

```python
from examples.custom_component import TanhValueConfig
from marl.algorithms import MAPPOConfig
config = MAPPOConfig(critic=TanhValueConfig(hidden_dim=32))
```

示例把有参数的模块挂在 nn.Module 内，parameters/state_dict/to(device) 都能管理它。
不要用普通 Python list 存放应被训练的子网络，使用 ModuleList/ModuleDict。
仅为 Python 组合不需要 registry、factory 或 ExtensionCatalog。
教学脚本中的等价完整训练：

```powershell
.\.venv\Scripts\python.exe doc\examples\train_walkthrough.py --custom-critic
```

### 每类模块必须提供什么

| 被替换部分 | 算法需要的能力 | 输入输出和注意点 |
|---|---|---|
| MAPPO/MAAC policy | DiscretePolicy：act/evaluate/logits/parameters | act 返回 MARLModelOutput；evaluate 评价给定动作；mask 贯穿两者 |
| 循环离散 policy | 上述接口 + is_recurrent，接收 hidden_state | 正确返回 h/c，状态 [K,B,N,H]；无状态时不伪造状态 |
| MADDPG/MASAC policy | PolicyTopology.act/parameters | 连续 actions [...,N,A]；MASAC 需要可重参数化动作与 log_prob |
| QMIX policy | LocalQPolicy.act/q_values | Q [...,N,A]，合法动作 argmax |
| MAPPO critic | ValueNetwork：call/parameters | state [...,S] → value [...,N] |
| 循环 MAPPO critic | RecurrentValueNetwork | hidden_state、return_state；critic 状态 [K,B,H] |
| MAAC critic | AttentionQNetwork | obs/联合离散动作 → Q 表 [...,N,A] |
| MADDPG critic | QEnsemble，暴露 critics | 每个独立 critic 读取联合观测动作，输出 [...,1] |
| MASAC critic | TwinQEnsemble，暴露 first/second.critics | 双 Q [...,N]；逐玩家参数隔离 |
| QMIX mixer | MixingNetwork | agent Q [...,N] + state [...,S] → 团队 Q [...,1] |

完整协议见 [policy](api/marl-modules-policy.md)、[critic](api/marl-modules-critic.md)、
[mixer](api/marl-modules-mixer.md)。满足方法名称还不够：必须满足数学含义、形状和梯度。
外部 mixer 若违反单调性就不再是标准 QMIX，不能因为接口相同就省略机制审计。

### 从 YAML 使用库外组件

显式创建 catalog，在加载配置时传入；完整示例可直接运行：

```powershell
.\.venv\Scripts\python.exe examples\custom_component.py
```

```python
from examples.custom_component import TanhValueConfig, read_settings
from marl.config import algorithm_config_from_dict
from marl.envs import ActionKind
from marl.extensions import ExtensionCatalog

catalog = ExtensionCatalog()
catalog.register(
    "critic", "tanh_value",
    parse=read_settings,
    build=TanhValueConfig.build,
    action_kinds=frozenset({ActionKind.DISCRETE}),
)
config = algorithm_config_from_dict({
    "schema_version": 2, "algorithm": "mappo",
    "critic": {"kind": "tanh_value", "hidden_dim": 32},
}, catalog=catalog)
```

文件路径版本用 `load_algorithm_config(path, catalog=catalog)`。parse 负责范围、类型和
未知字段检查；configure 在加载时绑定配置；build 在已知 spec 时实例化网络一次。
不在 import 时全局注册，不从 YAML import 类路径。恢复 checkpoint 时先创建同目录、
同解析器和组件实现；checkpoint 不携带 Python 源码。

### 增加一个数学目标

先用固定张量明确输入、标量公式和梯度流向。无参数目标采用冻结 dataclass，入口验证形状，
返回 ObjectiveResult(loss, metrics)；loss 保留图，日志在导出时 detach。
将 Config 字段放在目标或算法相邻文件，在算法构造函数创建目标，在对应 typed loss 方法
调用。新指标名必须唯一，LossBundle.combine 会拒绝重复名称。

例如增加 policy 正则，应接入 policy loss，不放进 collector，不在目标内部 optimizer.step。
有可训练参数的温度/辅助模型必须使用 nn.Module 并明确其 optimizer 归属；MASAC 是参考。
新 return estimator 放 returns.py，先确认输入是 TD transition 还是时间优先 rollout，
二者能力不同。YAML 选择新增估计器还需显式更新 config.py 对应分支，不能只改 kind 字符串。

<a id="algorithms"></a>
## 自定义完整算法的步骤

首先形成论文要求表：问题类型、智能体、观测、动作、奖励、fresh/replay、return、loss、
optimizer 顺序、target 时机、终止和评估。列出已满足、可复用需配置、缺失/偏差；没有
论文依据的默认值标注假设。若只需换组件，不增加新算法类。

确需新增时，按下面的可独立验收顺序实施：

1. 新文件 `marl/algorithms/<name>.py` 中定义冻结 LossConfig 与 Config；Config.build(spec)
   直接调用具体类；具体类直接继承 BaseMARLAlgorithm。
2. 构造函数校验 spec，构建 policy/critic/mixer、目标和 target。无环境常量或奖励公式。
3. act 只定义执行语义；compute_*_loss_bundle 分开各优化器的计算图。
4. update 接收 MARLBatch 或 PreparedRollout 和 OptimizerRuntime，按论文顺序调用 optimize，
   每个真实 step 调 record_optimizer_step，一轮结束调 finish，最后 export_metrics。
5. 声明 target_pairs 的 `(target, online)` 顺序；没有 target 就复用基类空结果。
6. 在 experiment.py 中分配命名 optimizer，并明确不重叠的参数集合。现有离策略生命周期
   复用 OffPolicyTrainer；新数据生命周期才增加通用 trainer 能力。
7. 更新 algorithms/__init__.py、必要的根导出、config.py 的 AlgorithmConfig union/解析/
   序列化，以及 experiment.py 的类型与构造分支。只增加类不会自动获得 YAML 支持。
8. 添加完整独立算法 YAML，与 Python 默认配置等价；写数学、形状、梯度、恢复、最小训练
   测试，再补本手册和 API 索引。

下面是编排示意，**不是可直接运行的新算法**，网络、损失与角色必须按论文提供：

```text
update(experience,runtime):
    critic_bundle = compute_critic_loss_bundle(experience)
    norm = optimize(critic_optimizer, critic_bundle, critic_clip, parameters=critic_parameters)
    runtime.record_optimizer_step("critic",norm)
    actor_bundle = compute_actor_loss_bundle(experience)  # 使用更新后的 critic
    norm = optimize(actor_optimizer, actor_bundle, actor_clip, parameters=actor_parameters)
    runtime.record_optimizer_step("actor",norm)
    runtime.finish(target_pairs())
    return runtime.export_metrics(detached_diagnostics)
```

参考实际 [MADDPG.update](api/marl-algorithms-maddpg.md#maddpg-update) 或
[MAPPO.update](api/marl-algorithms-mappo.md#mappo-update)。不能用示意代替新论文公式。
不要新增 BasePPOAlgorithm/UpdatePlan，不复制采集、GAE、replay 或第二套 optimize。

## 扩展完成标准

固定 Tensor 验证公式，显式检查目标参数变化与非目标参数不变，测试 mask/终止/截断，
Python/YAML 一致，checkpoint 可恢复下一次更新，最小环境可训练。最后运行
[完整门禁](training.md#verification)，更新文档与公共导出；长曲线不能替代这些检查。
