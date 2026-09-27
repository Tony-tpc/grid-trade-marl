# MARL 仓库开发指南

本文件是本仓库后续开发、论文复现和自动化 Agent 修改代码时的最高优先级工程约定。
目标不是让每篇论文都拥有一套互不兼容的代码，而是把论文差异拆成可复用的环境、
策略拓扑、目标函数、return 估计、经验来源和更新计划。

## 1. 核心原则

1. **先审计，后实现。** 新论文到来时，先列出论文公式、训练数据、网络结构、动作空间、
   奖励语义和终止语义，再与当前实现逐项对照。不要仅凭算法名称判断“已经支持”。
2. **优先组合，不复制算法。** 超参数差异写配置；已有组件的不同组合写新配置；
   新数学机制写新组件；只有组件组合不足以表达完整生命周期时才增加具体算法类。
3. **具体算法直接继承基类。** 所有具体算法直接继承 `BaseMARLAlgorithm`，不要增加
   `BasePPOAlgorithm`、`BaseActorCriticAlgorithm` 等第二层算法基类。
4. **环境差异留在环境层。** 数据集、物理约束、市场规则、动作编解码和奖励公式应放在
   环境或 adapter 中，不能进入通用算法源码。
5. **保持逐智能体语义。** 个体奖励、value、advantage 和 return 保持 `[..., N]`；
   除非任务被明确声明为共享奖励，否则禁止跨智能体求均值后当作训练目标。
6. **on-policy 与 off-policy 严格分离。** PPO/MAPPO 只能使用当前策略采集的新鲜 rollout；
   replay buffer 只用于允许重放旧经验的离策略算法。
7. **公共配置是静态数据。** YAML 只能选择内置组件或显式传入 ExtensionCatalog 的外部组件，禁止从 YAML 动态导入
   Python 类。智能体数、观测维度、动作维度和 state 维度只从 `EnvironmentSpec` 注入。
8. **有实质进展再提交、完整验证。** 开发步骤与 Git 提交边界分开；形成可独立验证的完整
   成果后再提交。同一目标所需的代码、配置、测试和文档一起提交，不按函数或小步骤拆分。

## 2. 修改前的论文审计

开始编码前建立一张“论文要求—仓库能力”对照表，至少回答以下问题：

| 类别 | 必须确认的内容 | 常见落点 |
|---|---|---|
| 问题定义 | 合作、一般和博弈、零和或层级博弈 | environment、reward structure |
| 智能体 | 同构/异构、固定/可变数量、是否参数共享 | policy topology、model |
| 观测 | 局部观测、全局 state、历史窗口、通信信息 | adapter、backbone |
| 动作 | 离散、连续、混合、约束或 action mask | adapter、action head、policy |
| 奖励 | 个体奖励还是共享团队奖励 | adapter、batch、objective |
| 数据 | fresh rollout、replay、序列、优先经验 | experience source、trainer |
| return | TD、n-step、GAE、lambda-return | return estimator |
| 目标 | policy/value/Q/entropy/auxiliary loss | objective |
| 更新 | 优化器数量、更新顺序、target update、频率 | update plan |
| 终止 | true termination 与 time-limit truncation | environment、return estimator |
| 评估 | 指标、baseline、随机种子、checkpoint | example、experiment code |

审计输出应明确分成三类：

- **已满足**：现有代码与论文数学及语义一致；
- **可复用但需配置**：只需配置或环境参数；
- **缺失或有偏差**：需要新组件、训练器能力或算法实现。

论文没有说明的细节不得伪装成论文事实。可以采用合理默认值，但必须在配置、注释和实验
记录中标明假设。

## 3. 如何选择正确的扩展层

按以下顺序判断，命中后不要继续向更重的层级扩展：

1. **仅超参数不同**

   新建 Python/YAML 配置，不增加类。
2. **现有机制重新组合**

   新建配置，复用现有 policy、critic、objective、return 和 update 组件。
3. **出现一个新的数学机制**

   在对应模块新增组件与其 Config，例如新 policy objective、新 return estimator 或
   target update；不要复制整个算法类。
4. **网络表示变化，但输入输出语义不变**

   在 `marl/models/` 增加 backbone，或在 `marl/modules/` 增加 Actor/Critic/Mixer；
   再由 policy 或配置组装。
5. **环境规则、数据或动作编码变化**

   增加 environment 和 `EnvironmentAdapter`，保持算法不变。
6. **经验生命周期或更新顺序根本不同**

   扩展经验缓冲/采样器；算法专用更新顺序放在具体算法的 `update()`，
   复用 `BaseMARLAlgorithm.optimize()`，只有数据生命周期不同才新增通用 trainer。
7. **以上仍无法表达一个独立算法**

   才新增直接继承 `BaseMARLAlgorithm` 的具体算法类。算法类负责组件装配和算法特有
   前向语义，不应重新实现已有 rollout、GAE、replay 或优化器循环。

### 典型判断示例

- PPO clip ratio 从 0.2 改为 0.1：配置。
- MAPPO 换成 GRU/LSTM policy：backbone/policy 配置 + 正确的状态传递与连续序列采样；
  不得只替换网络层就宣称支持循环训练。
- PPO 增加 value clipping：新 objective 组件 + 配置。
- GAE 改为 V-trace：新 return estimator。
- 新能源市场规则：新 environment/adapter。
- 离散动作改为论文定义的混合动作：action head、policy 和 adapter，不能只改动作维度。
- 新算法需要 actor、critic 使用不同优化器并交替更新：具体算法的 `update()` 接线；
  不新增 UpdatePlan 或在 policy 中实现优化器循环。
- 新算法的数据流与现有算法完全不同：新 trainer；仍应复用 batch、模型和目标组件。

## 4. 模块边界

| 路径 | 职责 | 不应放入 |
|---|---|---|
| `marl/core/` | 跨算法数据协议：`MARLBatch`、`MARLModelOutput` | 论文公式、环境规则 |
| `marl/envs/` | 环境、adapter、`EnvironmentSpec`、动作编解码 | optimizer、loss |
| `marl/models/` | MLP/GRU/LSTM/GNN/Transformer 等表示学习 backbone | 训练循环 |
| `marl/modules/` | Actor、Policy 拓扑、Critic、Mixer、ActionHead | 环境结算规则、rollout 生命周期 |
| `marl/objectives.py` | 可复用目标函数和 `LossBundle` | optimizer.step |
| `marl/returns.py` | GAE、未来可扩展的 return estimator | 环境 reset |
| `marl/config.py` | 算法专用 YAML 解析与配置序列化 | 网络实例化、环境尺寸默认值 |
| `marl/experiment.py` | 一次性构造算法、optimizer 和 trainer | 算法数学 |
| `marl/extensions.py` | 外部 policy/critic/mixer 可选目录与 Buildable 协议 | 内置组件注册、动态 import |
| `marl/training/` | rollout、mini-batch、更新计划和 trainer | 按算法名称分支 |
| `marl/runtime.py` | 设备、向量环境和离策略 replay 等执行设施 | 算法数学 |
| `marl/algorithms/` | 具体算法的薄装配与特有前向语义 | 可复用训练循环 |
| `examples/` | 可运行实验入口和配置示例 | 隐藏的核心实现 |
| `tests/` | 数学、形状、边界和端到端回归 | 依赖人工观察的断言 |

依赖方向应尽量保持：

```text
core <- models <- modules <- algorithms
  ^                   ^          ^
  +-- objectives/returns --------+
  +-- envs/runtime/training -----+
config 只解析数据；experiment 负责装配；内置组件配置直接 build。
```

若为解决循环依赖而需要延迟导入，先检查模块职责。trainer 不导入算法配置，
checkpoint 接收 experiment 生成的配置数据快照；算法可以读取同模块训练参数 dataclass。
网络接口定义在实际使用它们的模块旁，不创建无调用方的公共协议文件。

## 5. 公共张量与语义约定

统一使用以下符号：

- `B`：batch 或并行环境维；
- `T`：时间维；
- `N`：智能体维；
- `O`：单个智能体观测维；
- `A`：动作维或离散动作数量；
- `S`：全局 state 维。

关键形状：

- observations：`[*B, N, O]`
- discrete actions：`[*B, N]`
- continuous actions：`[*B, N, A]`
- rewards/value/advantage/return/log-prob：`[*B, N]`
- state：`[*B, S]`
- action mask：`[*B, N, A]`，`True` 表示合法

新增组件必须在入口尽早验证形状，并在 docstring 中写明输入输出。不能依赖错误的广播在
深层 loss 中“自然报错”。

### termination 与 truncation

- `terminated=True`：MDP 真正终止，下一状态不得 bootstrap；
- `truncated=True`：时间限制截断，当前 TD/GAE delta 仍可使用下一状态 value；
- episode 边界之后不能继续把下一段 episode 的 advantage 递推回来。

不得只用一个模糊的 `done` 同时处理上述两种语义。

### 梯度边界

- 环境采样和 bootstrap value 使用 `torch.inference_mode()` 或 `no_grad()`；
- old log-prob、old value、TD target 不应在更新时重新连接计算图；
- actor loss 可通过 critic 对 action 求导，但不应误更新 critic 参数；
- target network 更新必须在 `no_grad` 下进行；
- 每个 optimizer 应只持有它负责的参数，测试中验证非目标参数不变化。

## 6. 新增环境

1. 在 `marl/envs/` 实现具体环境或包装外部环境。
2. 实现 `EnvironmentAdapter`：
   - `spec` 返回静态 `EnvironmentSpec`；
   - `reset(seed)` 返回统一的 `EnvironmentStep`；
   - `step(actions)` 负责标准动作与环境动作之间的转换。
3. 每次返回前调用 `validate_step()`。
4. 明确设置 `RewardStructure.INDIVIDUAL` 或 `SHARED`。
5. 为离散动作始终提供 `[N,A]` action mask；连续动作必须返回 `None`。
6. 在 `marl/envs/__init__.py` 暴露稳定公共类型。
7. 增加测试：
   - reset/step 形状、dtype 和有限值；
   - 固定 seed 可复现；
   - action 边界和 mask；
   - 每个 reward 分量；
   - termination/truncation；
   - adapter 不泄漏算法专用张量。

不要为了接入新环境去修改算法中的 observation/action 维度；这些尺寸必须来自
`adapter.spec`。

## 7. 新增或修改组件

### 新增带参数组件

带可训练参数的组件使用 `nn.Module`。实现协议所需方法，并保证所有子模块挂在
`self` 上，以便 `parameters()`、`state_dict()` 和 `.to(device)` 正常工作。

### 新增无参数组件

无参数 objective、return estimator 或 target update 优先使用
`@dataclass(frozen=True, slots=True)`，并在 `__post_init__` 校验参数范围。

### 内置组件开发

1. 在组件实现旁定义冻结 Config，默认值、范围和 `build(spec)` 放在同一文件。
2. Python 直接传给算法 Config；内置路径不使用 factory 或注册表。
3. 需要 YAML 时在 `marl/config.py` 对应算法分支显式添加选择，并更新字段说明。
4. 算法依赖最小能力协议，具体网络必须是 `nn.Module` 并正确注册参数。
5. 只有 YAML/checkpoint/外部数据边界允许必要的 `Any` 或 `cast`；
   禁止在算法装配中先返回 `object` 再逐层强转。
6. 扩展数学目标时，直接在算法构造函数接线；不引入通用无类型 objective 列表。

### 外部组件开发

外部 policy/critic/mixer 使用显式创建的 `ExtensionCatalog`，注册配置解析器、
构造方法以及动作/奖励兼容性。catalog 传给 `load_algorithm_config`，加载时绑定配置，
构建时只实例化一次。内置组件不经过 catalog。禁止 import 时自动注册和 YAML 动态导入。
外部网络只需实现实际能力接口，不要求继承内置网络；提供 checkpoint 往返测试。
示例见 `examples/custom_component.py`。

## 8. 新增配置或算法变体

### 配置布局

每个算法使用完整独立的 `examples/configs/algorithms/<algorithm>.yaml`；
环境使用独立的 `examples/configs/environments/<environment>.yaml`。
seed、device、并行环境数和输出路径属于运行参数。

算法 YAML 必填 `schema_version: 2` 和 `algorithm`；其余节与字段使用 Config 默认值。
网络使用 `kind + 直接字段`。MAPPO 使用 advantage/rollout，
离策略算法使用 value_target/replay/target_update，QMIX 使用 mixer。
不增加全算法通用的万能配置对象；不引入无意义的 target_update: none。
示例完整列出可选字段、默认值、范围和列表含义，并与 Python 默认值做等价测试。

### 新增具体算法

1. 新类直接继承 `BaseMARLAlgorithm`，同文件定义算法 Config 和 loss 参数。
2. Config 的 `build(spec)` 直接构造该算法；算法构造函数直接构造网络及数学目标。
3. 尺寸只来自 EnvironmentSpec；输出和损失分别复用 MARLModelOutput、LossBundle。
4. 更新公共导出、AlgorithmConfig union、YAML 解析分支和必要的 experiment 类型声明。
5. 现有离策略生命周期可直接复用 OffPolicyTrainer；新生命周期才新增 trainer。
6. trainer 不按算法名称分支，不导入配置解析或目录；算法基类只复用单次优化原语，
   算法专用的 optimizer 顺序与 PPO epoch 在具体算法的 `update()` 中。
7. 测试配置解析、固定张量公式、梯度边界、最小训练与 checkpoint 续训。
8. 不复制已有 rollout/GAE/replay/update 实现。新机制优先新增组件和少量装配接线。

正常使用入口是 `build_experiment(env, config, device=..., seed=...)`。
底层可直接 `MAPPO(spec, MAPPOConfig())`；不存在旧 recipe 兼容接口。

## 9. On-policy 与 off-policy 修改要求

### On-policy

- rollout 保存 observation、state、action、mask、reward、old log-prob、old value、
  terminated 和 truncated；
- rollout 完成后计算 bootstrap value、advantage 和 return；
- PPO 可对同一 rollout 做多 epoch/minibatch 更新，但完成后必须废弃；
- old log-prob/value 不得被更新过程覆盖；
- 记录 policy loss、value loss、entropy、approx KL、clip fraction、explained variance
  和 gradient norm。

### Off-policy

- replay 中保存当前和下一状态、动作、奖励及终止信息；
- target 使用 true termination 控制 bootstrap；
- 明确 target network 的 hard/soft update 时机；
- 多智能体 critic 不得为了批处理而共享本应独立的参数或 reward target；
- 性能优化必须有数值等价测试，不能只比较速度。

## 10. 测试策略

每个功能至少覆盖以下层次：

1. **纯数学单测**：固定小张量验证公式、边界和逐智能体结果；
2. **形状/错误测试**：错误 action kind、state、mask、reward structure 应尽早失败；
3. **梯度测试**：目标参数有梯度，冻结或 target 参数无意外梯度；
4. **状态测试**：checkpoint 恢复模型、optimizer、配置和更新计数；
5. **端到端小环境**：固定 seed 完成一次采样和参数更新；
6. **迁移等价回归**：迁移前后固定输入的 action/log-prob/value/loss 等价。

不要用长时间训练代替数学单测。训练曲线可用于最终验证，但不能证明公式实现正确。

训练审计应区分高频训练诊断与独立策略评估，保留真实采样步数、原始指标及逐 seed 曲线。
长基准不能只记录少量终局或十个粗粒度区间；评估频率与诊断频率应可配置，并记录实际
采样覆盖率。提供回报、分离的 actor/critic 损失及滚动趋势图；不同量纲分轴展示。
旧结果缺失的历史不得用插值或最终 checkpoint 重评估来冒充。曲线趋平、数值有限与
策略收敛分别解释，不以此直接宣称达到博弈均衡。

## 11. 开发验证与提交质量门禁

开发中持续执行相关回归，测试阶段不强制对应 Git 提交。每个完整成果提交前使用项目
虚拟环境执行完整门禁：

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp=.pytest_cache/refactor-tmp
.\.venv\Scripts\python.exe -m ruff check marl examples tests
.\.venv\Scripts\python.exe -m mypy marl examples tests
.\.venv\Scripts\python.exe -m pip check
git diff --check
git status --short
```

提交前还需人工审查：

- 张量形状和广播是否符合注释；
- 个体/共享奖励是否被正确保留；
- termination/truncation 与 bootstrap 是否正确；
- old policy 数据、TD target 和 target network 是否正确 detach；
- optimizer 是否更新了正确参数；
- action mask 是否贯穿采样与 evaluate；
- checkpoint 是否足以继续训练；
- 配置解析和 ExtensionCatalog 是否拒绝未知配置和不兼容环境；
- 通用代码是否出现论文名、特定数据集或特定环境分支；
- 文档和示例是否与公共 API 同步。

## 12. Git 提交与成果划分

- 开始前先运行 `git status --short`，不要覆盖用户已有改动。
- 只有形成明显进展、可独立验证的完整成果，且通过测试和审查后才提交。
- 开发步骤不等于提交边界；不为单个函数、测试、配置接线或文档补充单独提交。
- 同一目标所需的源码、配置、测试和说明合并提交，不合并无关目标。
- 稳定性优化和运行速度优化分别形成完整成果；各部分内部可以分步骤开发和验证。
- 不提交红测或“下一阶段再修”的半成品，也不将每个测试阶段机械拆成一次提交。
- commit 标题与正文使用中文：标题明确主要改进，正文详细列出代码改动、解决的问题、
  实际验证结果和兼容性影响。禁止仅写“优化代码”“修复问题”等笼统说明。
- 提交说明只描述已完成并验证的改动，不把计划或未完成实验写成成果。
- 不提交缓存、训练产物、临时 PDF、可视化、checkpoint 或本地虚拟环境。
- 未经明确要求不要推送远端。

## 13. Definition of Done

一项新论文或算法扩展只有同时满足以下条件才算完成：

- 论文要求与实现的对应关系有记录；
- 公共机制被抽成组件，没有复制整套算法；
- 环境专用逻辑没有进入通用算法；
- 关键公式有固定张量测试；
- 个体奖励、终止和梯度语义经过审查；
- Python/YAML 配置（若适用）行为一致；
- 最小端到端训练与 checkpoint 恢复通过；
- pytest、Ruff、mypy、`git diff --check` 全部通过；
- README、示例和公共导出已更新；
- 工作树只包含本任务预期改动。

### 循环 MAPPO 的额外契约

- MLP/GRU/LSTM 共用 MAPPO，不新增算法别名、UpdatePlan 或 optimizer 循环。
- LSTM 的 h/c 都必须保存；actor 和 critic 状态独立，采样保存处理当前观测之前的状态。
- 循环 mini-batch 只打乱片段顺序，不打乱时间；padding 必须从 loss、统计与计数排除。
- 在最终观测上、reset 之前计算截断 bootstrap；新 rollout 重置环境与状态。
- 初态为采样策略的 detached 状态，片段内反传；第一版无 burn-in，必须披露此近似。
- 评估/cross-play 必须持续传状态，来源之间互不串用，结束恢复原始 train/eval 模式。
- checkpoint schema 4 只承诺完整 rollout 更新后的恢复；旧 schema 不静默部分恢复。
- 循环能力应通过专用记忆任务验证；有限 loss 或曲线趋平不等于博弈收敛。
