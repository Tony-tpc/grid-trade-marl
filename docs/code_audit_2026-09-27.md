# 全库代码审计与阅读说明（2026-09-27）

## 范围与结论

以 `7b8be64` 为修改前基线，开始时工作树干净。检查了 `marl/` 43 个 Python 文件、
`benchmarks/` 7 个文件、`examples/` 4 个文件，以及已有测试、配置和工程说明。
采用全目录 AST 结构/重复函数体扫描，结合核心公式、数据流、梯度、接口调用方的人工
审查和完整回归测试。结构扫描不等于对任意外部插件或任意论文实现的正确性证明。

五个具体算法仍直接继承 `BaseMARLAlgorithm`；没有新增算法基类、UpdatePlan、内置
registry 或第二套 optimizer 循环。并非“没有发现问题”：本次删除了多余配置壳类，
修复生命周期和诊断问题，并把扩展限制写入对应源码。没有覆盖历史训练结果或启动长训。

## 已满足、可复用与修复项

### 已满足并保留

- `models` 只做特征，`modules` 组装网络，`algorithms` 持有数学与更新顺序。
- `BaseMARLAlgorithm.optimize` 是库内唯一 backward/optimizer.step 实现；trainer
  不按算法名分支。`OptimizerRuntime` 仅保存状态、AMP 上下文和指标。
- MAPPO 使用 fresh rollout；离策略算法使用 replay。个体目标维 `[... ,N]` 保留，
  QMIX 在环境和 batch 层检查共享奖励，不用平均非合作奖励绕过兼容性约束。
- 动作分布、attention、优化器使用 PyTorch 组件；没有自写 attention kernel。
- YAML 只解析数据，内置 Config 直接 build；外部组件通过显式 ExtensionCatalog
  接入，不从 YAML 动态导入 Python 类。
- 基准复用同一算法训练入口，按动作空间分组，分角色报告收益；进程并发不复制公式。

### 可复用，但需要明确配置或调用契约

- Adam 创建处有显式 actor/critic/temperature 参数分组，这是可审查的装配，不是
  多套优化算法。共享编码器若要被多个角色训练，必须先设计唯一参数所有者，不能把
  同一参数同时交给两个 optimizer；现在会尽早拒绝这种情况。
- `act`、`sample`、`evaluate` 分别服务执行、rollout 采样和给定动作评估；它们复用
  同一 policy，不是重复策略实现。
- `ObjectiveResult` 是一个目标，`LossBundle` 是多个目标的组合；`BackboneOutput`、
  `ActionHeadOutput`、`MARLModelOutput` 分属特征、单 actor、多智能体三个边界。
- 确定性和高斯 policy 的动作堆叠代码相似，但分布和默认采样语义不同。本轮保留
  直观的少量组装代码，不引入新的策略父类或通用工厂来隐藏这些差异。
- `update_batch` 允许外部经验来源绕过 replay 抽样，仍调用同一个算法 update；
  性能基准中的 `_LegacyPreparedRollout` 仅为旧数据路径对照，不是公共训练接口。

### 已修复

| 问题 | 改动 | 验证方式 |
|---|---|---|
| MAAC/MADDPG/MASAC update 配置空继承壳 | 直接使用已有共享配置类型，删除旧名称 | YAML 字段往返、专用学习率测试 |
| trainer 协议要求未使用的网络/公式能力 | 仅保留实际调用的方法；collector 独立依赖 act | mypy、结构审计测试 |
| MAPPO 调用 runtime 私有归约方法 | 原方法改为公开命名，不保留重复别名 | CPU/CUDA 归约回归 |
| 不等长 PPO mini-batch 指标等权平均 | 普通指标按样本数；梯度按 step；极值/计数不加权 | 手工参考循环对照 loss、参数、Adam 状态、RNG |
| rollout 迁移设备后旧对象仍可训练 | to 转交一次性消费权；迁移失败不消费 | 转移、重复消费、失败回退测试 |
| 重复 rollout 调用会先修改 target 统计再报错 | 在统计更新前检查 consumed | 全部模型/统计 state 精确不变 |
| Actor/Critic 直接调用 backbone.forward 绕过 hooks | 使用标准 nn.Module 调用；保留 typed 输出 | hooks 触发与固定前向输出测试 |
| runtime 未检查 optimizer 参数重叠 | 构造时检查空参数组和重复归属 | 错误构造立即失败 |
| batch 校验遗漏下一观测、逐体奖励、状态等形状 | 检查元数据，拒绝静默广播，不增加 GPU 数值同步 | 参数化形状错误测试 |
| QMixer 可混淆展平后大小相同的不同批次布局 | 展平前检查前置维相同 | `[2,3]` 与 `[3,2]` 布局反例 |
| soft target 忽略自定义模块的非参数 buffer | 参数 EMA，float/int buffer 完整复制 | BatchNorm 统计/整数计数器测试 |
| QMIX 未记录 value 梯度裁剪事件 | 同其他算法记录真实 value step 事件 | 梯度峰值、step 数和裁剪率测试 |
| 无调用方的 NoTargetUpdate 和 MAPPO 包装方法 | 删除空更新对象/EV 包装，复用 values 方法 | 全目录引用扫描、完整测试 |

## Python 接口迁移

本次主动删除没有独立语义的旧 Python 名称，不添加兼容别名。仓库内调用方和测试已
同步迁移；外部代码如果曾直接导入这些类型，需要修改导入和类型名：

| 旧名称 | 当前唯一类型 |
|---|---|
| `marl.algorithms.maac.MAACUpdateConfig` | `marl.training.ActorCriticUpdateConfig` |
| `marl.algorithms.maddpg.MADDPGUpdateConfig` | `marl.training.ActorCriticUpdateConfig` |
| `marl.algorithms.masac.MASACUpdateConfig` | `marl.training.TemperatureActorCriticUpdateConfig` |
| `NoTargetUpdate()` | `OptimizerRuntime(target_update=None, ...)` |

YAML 字段和模型参数布局未因这次配置合并而改变。不要把旧配置对象 pickle 当成公共
保存协议；使用现有纯数据配置快照。基准精确续训仍校验源码 hash，本轮修改后旧任务
会被拒绝静默拼接；旧模型可显式加载用于评估。旧 loss 历史不会被“补算”为新日志。

## 建议的源码阅读顺序

1. [数据协议](../marl/core/batch.py)：先理解 `[B,N,O]`、`[B,N]`、终止与截断。
2. [实验装配](../marl/experiment.py)：确认网络和 optimizer 各构造一次、参数归属清晰。
3. [MAPPO](../marl/algorithms/mappo.py)：从 sample、typed loss 到 update 顺序阅读；
   再看 [rollout](../marl/training/on_policy.py) 与 [GAE](../marl/returns.py)。
4. [MADDPG](../marl/algorithms/maddpg.py)、[MASAC](../marl/algorithms/masac.py)：
   对照冻结 critic 参数但保留 action 梯度，与 TD target 整段 no_grad 的区别。
5. [MAAC](../marl/algorithms/maac.py) 与 [attention critic](../marl/modules/critic.py)：
   理解逐体 head、固定其他玩家动作、反事实 baseline 和单 actor 梯度隔离。
6. [公共优化动作](../marl/algorithms/base.py) 与 [运行状态](../marl/training/optimization.py)：
   区分更新次数、真实 optimizer steps、均值/峰值/裁剪事件。
7. [runtime](../marl/runtime.py) 与 [checkpoint](../marl/training/checkpoint.py)：
   理解 CPU 环境、批量推理、pinned 传输、replay 和 RNG 恢复边界。

源码注释补充了上述输入输出、梯度边界、行为原因和扩展限制；不逐行翻译 Python 语法。

## 明确保留的限制

- 当前训练器不支持部分智能体/部分环境异步 reset。GAE 的 values[t+1] 必须对应实际
  后继状态；截断后自动 reset 的 packed rollout 需要额外 bootstrap 数据设计。
- GRU 组件存在不等于循环 PPO 已接通。Transformer 没有位置编码和因果 mask，不能
  直接当作因果时间策略；共享 encoder、异质维度或混合动作也不能只靠改 Config 接入。
- MLP Config.hidden_dim 控制输出特征层，中间层仍用 backbone 的默认 (128,128)；
  本轮不更改网络布局，以免破坏旧模型。需要完整层宽控制时显式构造 backbone。
- target 的 requires_grad=False 不等于 eval；外部 BatchNorm/dropout 网络须明确推理
  模式。软更新的 buffer 复制是状态同步约定，不是对所有外部网络语义的自动证明。
- checkpoint 加载使用完整 pickle 状态，只允许可信本地文件。通用 trainer checkpoint
  不保存 episode 中途环境状态；基准精确续训只承诺已记录的 rollout 边界。
- 产物 validator 的检查以报告内列出的任务为范围；核对外部要求的算法集合时还需
  对照启动清单。曲线完整、有限值、短期健康和博弈收敛是四个不同结论。
- 本轮没有做新一轮长训或端到端提速验收；不宣称之前未达到的性能目标现已达成。

## 验证

最终验证：152 项 pytest 全部通过（48.19 秒），Ruff 通过，mypy 检查 75 个源文件
通过，pip check 与 git diff --check 通过；基础组件和外部 critic 示例也运行成功。
新增 `tests/test_architecture_audit.py` 的 23 个回归用例，包含实际 CUDA 指标归约。
完整测试覆盖原有数学公式、梯度隔离、checkpoint 恢复、外部组件、训练、Windows
spawn 和绘图，再叠加本次接口/生命周期回归。最终门禁命令：

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp=.pytest_cache/audit-final
.\.venv\Scripts\python.exe -m ruff check marl examples benchmarks tests
.\.venv\Scripts\python.exe -m mypy marl examples benchmarks tests
.\.venv\Scripts\python.exe -m pip check
git diff --check
```

审计测试防止已知结构倒退，不能保证未来任意新增组件天然正确；新增数学或生命周期
仍须按 AGENTS.md 补公式、形状、梯度、状态和端到端测试。
