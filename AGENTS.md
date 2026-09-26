# MARL 仓库开发指南

本文件是本仓库后续开发、论文复现和自动化 Agent 修改代码时的最高优先级工程约定。
目标不是让每篇论文都拥有一套互不兼容的代码，而是把论文差异拆成可复用的环境、
策略拓扑、目标函数、return 估计、经验来源和更新计划。

## 1. 核心原则

1. **先审计，后实现。** 新论文到来时，先列出论文公式、训练数据、网络结构、动作空间、
   奖励语义和终止语义，再与当前实现逐项对照。不要仅凭算法名称判断“已经支持”。
2. **优先组合，不复制算法。** 超参数差异写 recipe；已有组件的不同组合写新 recipe；
   新数学机制写新组件；只有组件组合不足以表达完整生命周期时才增加具体算法类。
3. **具体算法直接继承基类。** 所有具体算法直接继承 `BaseMARLAlgorithm`，不要增加
   `BasePPOAlgorithm`、`BaseActorCriticAlgorithm` 等第二层算法基类。
4. **环境差异留在环境层。** 数据集、物理约束、市场规则、动作编解码和奖励公式应放在
   环境或 adapter 中，不能进入通用算法源码。
5. **保持逐智能体语义。** 个体奖励、value、advantage 和 return 保持 `[..., N]`；
   除非任务被明确声明为共享奖励，否则禁止跨智能体求均值后当作训练目标。
6. **on-policy 与 off-policy 严格分离。** PPO/MAPPO 只能使用当前策略采集的新鲜 rollout；
   replay buffer 只用于允许重放旧经验的离策略算法。
7. **公共配置是静态数据。** YAML 只能引用已显式注册的组件名称，禁止从 YAML 动态导入
   Python 类。智能体数、观测维度、动作维度和 state 维度只从 `EnvironmentSpec` 注入。
8. **小提交、全门禁。** 每个阶段必须独立可测试、可审查；不要把环境、算法、性能优化和
   文档重写混在同一个提交中。

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
- **可复用但需配置**：只需 recipe 或环境参数；
- **缺失或有偏差**：需要新组件、训练器能力或算法实现。

论文没有说明的细节不得伪装成论文事实。可以采用合理默认值，但必须在配置、注释和实验
记录中标明假设。

## 3. 如何选择正确的扩展层

按以下顺序判断，命中后不要继续向更重的层级扩展：

1. **仅超参数不同**

   新建 Python/YAML recipe，不增加类。
2. **现有机制重新组合**

   新建 recipe，复用已注册 policy、critic、objective、return 和 update 组件。
3. **出现一个新的数学机制**

   在对应平级模块新增组件并注册，例如新 policy objective、新 return estimator 或
   target update；不要复制整个算法类。
4. **网络表示变化，但输入输出语义不变**

   在 `marl/models/` 增加 backbone，或在 `marl/modules/` 增加 Actor/Critic/Mixer；
   再由 policy 或 recipe 组装。
5. **环境规则、数据或动作编码变化**

   增加 environment 和 `EnvironmentAdapter`，保持算法不变。
6. **经验生命周期或更新顺序根本不同**

   增加 `ExperienceSource` 或 `UpdatePlan`，必要时新增通用 trainer。
7. **以上仍无法表达一个独立算法**

   才新增直接继承 `BaseMARLAlgorithm` 的具体算法类。算法类负责组件装配和算法特有
   前向语义，不应重新实现已有 rollout、GAE、replay 或优化器循环。

### 典型判断示例

- PPO clip ratio 从 0.2 改为 0.1：recipe。
- MAPPO 换成 GRU policy：backbone/policy topology 组件 + recipe。
- PPO 增加 value clipping：新 objective 组件 + recipe。
- GAE 改为 V-trace：新 return estimator。
- 新能源市场规则：新 environment/adapter。
- 离散动作改为论文定义的混合动作：action head、policy 和 adapter，不能只改动作维度。
- 新算法需要 actor、critic 使用不同优化器并交替更新：新 UpdatePlan。
- 新算法的数据流与现有算法完全不同：新 trainer；仍应复用 batch、模型和目标组件。

## 4. 模块边界

| 路径 | 职责 | 不应放入 |
|---|---|---|
| `marl/core/` | 跨算法数据协议：`MARLBatch`、`MARLModelOutput` | 论文公式、环境规则 |
| `marl/envs/` | 环境、adapter、`EnvironmentSpec`、动作编解码 | optimizer、loss |
| `marl/models/` | MLP/GRU/GNN/Transformer 等表示学习 backbone | 训练循环 |
| `marl/modules/` | Actor、Critic、Mixer、ActionHead | 环境结算规则 |
| `marl/policies.py` | 观测到动作分布的策略拓扑 | rollout 生命周期 |
| `marl/objectives.py` | 可复用目标函数和 `LossBundle` | optimizer.step |
| `marl/returns.py` | GAE、未来可扩展的 return estimator | 环境 reset |
| `marl/components.py` | 平级组件协议 | 具体算法注册 |
| `marl/registry.py` | 分类显式注册和兼容性验证 | 动态 import |
| `marl/recipes.py` | 不可变 recipe、YAML 解析、环境绑定 | 环境尺寸默认值 |
| `marl/builtins.py` | 内置组件工厂及严格 options 校验 | 论文专用配置 |
| `marl/training/` | rollout、mini-batch、更新计划和 trainer | 按算法名称分支 |
| `marl/runtime.py` | 设备、向量环境和离策略 replay 等执行设施 | 算法数学 |
| `marl/algorithms/` | 具体算法的薄装配与特有前向语义 | 可复用训练循环 |
| `examples/` | 可运行实验入口和 recipe 示例 | 隐藏的核心实现 |
| `tests/` | 数学、形状、边界和端到端回归 | 依赖人工观察的断言 |

依赖方向应尽量保持：

```text
core <- models/modules <- policies/objectives/returns
  ^              ^                 ^
  |              +------ algorithms+
  +-- envs/runtime/training --------+
registry/recipes 只负责选择和装配，不反向依赖具体论文环境。
```

若为解决循环依赖而需要大量延迟导入，先重新检查模块职责是否放错。

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

### 注册流程

1. 实现组件及独立单元测试；
2. 在 `marl/builtins.py` 增加严格 factory，拒绝未知 options；
3. 使用对应类别注册接口，并声明支持的 `ActionKind`、`RewardStructure`；
4. 在 recipe 中引用注册名称；
5. 测试未知名称、重复注册、非法 options 和不兼容环境。

注册名称使用稳定的小写 snake_case。不要把 Python 文件路径或类路径写入 YAML。

## 8. 新增 recipe 或算法变体

### 仅新增 recipe

当 policy、critic、objectives、return、experience、update 和 target update 均已有时，
只新增 YAML/Python recipe 与行为测试。YAML 顶层必须保持 `schema_version: 1`。

检查 Python recipe 与 YAML recipe 在同一 `EnvironmentSpec` 下编译出等价组件。
环境尺寸不能出现在 component options 中。

每个算法使用一份完整的 `examples/configs/algorithms/<algorithm>.yaml`；`algorithm`
字段只用于类型校验，不能通过仅修改该字符串切换算法。环境动力学和动作编码使用独立的
`examples/configs/environments/<environment>.yaml`。seed、device、并行环境数量和输出
路径属于实验运行参数，不进入算法或环境 recipe。新增或修改 YAML 时同步更新
`examples/configs/README.md` 中的全部可选字段、默认值和约束。

### 新增具体算法

只有新算法具有不能由当前组件表达的特有前向/损失语义时才增加算法类：

1. 新类直接继承 `BaseMARLAlgorithm`；
2. 使用不可变 `AlgorithmRecipe`，环境尺寸只来自 `EnvironmentSpec`；
3. 模型输出优先使用 `MARLModelOutput`；
4. 组合损失优先使用 `LossBundle`；
5. 算法类不拥有通用 optimizer epoch 循环；
6. trainer 依赖协议能力，不使用 `if algorithm_name == ...`；
7. checkpoint 必须覆盖模型、优化器、recipe 和必要计数器；
8. 在固定张量下用论文公式或可信旧实现做数值等价测试。

如果为了一个论文变体准备复制现有 `mappo.py` 或覆写整个 `compute_loss`，先停止，
重新判断差异能否成为 objective、return estimator、policy 或 update plan。

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
4. **状态测试**：checkpoint 恢复模型、optimizer、recipe 和更新计数；
5. **端到端小环境**：固定 seed 完成一次采样和参数更新；
6. **迁移等价回归**：迁移前后固定输入的 action/log-prob/value/loss 等价。

不要用长时间训练代替数学单测。训练曲线可用于最终验证，但不能证明公式实现正确。

## 11. 每阶段质量门禁

使用项目虚拟环境，在每个独立阶段执行：

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check marl examples tests
.\.venv\Scripts\python.exe -m mypy marl examples tests
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
- registry 是否拒绝未知配置和不兼容环境；
- 通用代码是否出现论文名、特定数据集或特定环境分支；
- 文档和示例是否与公共 API 同步。

## 12. Git 与阶段拆分

- 开始前先运行 `git status --short`，不要覆盖用户已有改动。
- 一次提交只完成一个可描述的目标。
- 推荐顺序：协议/数据结构 → 数学组件 → 训练链 → 算法迁移 → 示例/文档。
- 每个阶段先测试、review，再提交；不要提交红测或“下一阶段再修”的半成品。
- 不提交缓存、训练产物、临时 PDF、可视化、checkpoint 或本地虚拟环境。
- 未经明确要求不要推送远端。

## 13. Definition of Done

一项新论文或算法扩展只有同时满足以下条件才算完成：

- 论文要求与实现的对应关系有记录；
- 公共机制被抽成组件，没有复制整套算法；
- 环境专用逻辑没有进入通用算法；
- 关键公式有固定张量测试；
- 个体奖励、终止和梯度语义经过审查；
- Python/YAML recipe（若适用）行为一致；
- 最小端到端训练与 checkpoint 恢复通过；
- pytest、Ruff、mypy、`git diff --check` 全部通过；
- README、示例和公共导出已更新；
- 工作树只包含本任务预期改动。
