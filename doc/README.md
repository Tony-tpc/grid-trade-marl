# MARL 中文使用与开发手册

从环境输入到参数更新，按任务和源码两条路线阅读。`algorithm` 对应实际目录
`marl/algorithms`，`train` 对应 `marl/training`。本手册以当前源码为准。

## 按任务阅读

| 想解决的问题 | 入口 |
|---|---|
| 各文件夹放什么、从哪里读代码 | [架构与数据流](architecture.md) |
| 第一次安装、运行和保存训练 | [从零开始训练](training.md#start) |
| 换成自己的环境、奖励、数据集 | [环境接入教程](environments.md) |
| 换 actor、critic、backbone 或外部模块 | [模块扩展](extensions.md#modules) |
| 实现论文新目标或新算法 | [自定义算法](extensions.md#algorithms) |
| 调参、评估、排查不学习和恢复训练 | [实验与调参](training.md#tuning) |
| 每个算法怎么计算 loss、如何更新 | [五个算法逐步说明](algorithms.md) |
| rollout/replay、循环状态、优化器分工 | [训练模块说明](trainers.md) |
| 查某个函数的参数、返回值与源码 | [全库函数索引](api/README.md) |

建议先运行 [小环境示例](examples/train_walkthrough.py)，再按
`EnvironmentStep → MARLBatch → loss → optimize` 阅读。

## 函数级跳转

[MAPPO.sample](api/marl-algorithms-mappo.md#mappo-sample) →
[OnPolicyTrainer.collect](api/marl-training-on_policy.md#onpolicytrainer-collect) →
[RolloutBuffer.finish](api/marl-training-on_policy.md#rolloutbuffer-finish) →
[MAPPO.update](api/marl-algorithms-mappo.md#mappo-update) →
[BaseMARLAlgorithm.optimize](api/marl-algorithms-base.md#basemarlalgorithm-optimize)。

API 页面提供目录、稳定的显式锚点、实际签名和源码行号链接。算法和训练模块的全部
函数（包含私有辅助函数、协议、属性、嵌套函数）展开用途、输入、输出及副作用；其他模块
按文件列出类、函数说明和签名。dataclass 自动生成的构造参数见类字段，继承自 PyTorch
但未在仓库重写的方法不重复列出。

GitHub 支持源码 `#L行号`；若本地阅读器不能定位行号，可打开文件后搜索完整函数名。
手册内部锚点不依赖源码行数，重新生成后仍可跳转。

## 示例与维护

- [新环境完整实现](examples/cue_environment.py)：固定长度、逐智能体奖励教学环境。
- [训练与续训](examples/train_walkthrough.py)：MAPPO/MAAC、循环策略、自定义 critic。
- [API 生成工具](tools/build_api.py)：只解析源码，不导入或运行算法。
- [文档检查](tools/check_docs.py)：检查链接、锚点、源码行号和核心函数注释覆盖。

```powershell
.\.venv\Scripts\python.exe doc\tools\build_api.py
.\.venv\Scripts\python.exe doc\tools\check_docs.py
```

已有专题保留：[循环 MAPPO](../docs/recurrent_mappo.md)、[代码审计](../docs/code_audit_2026-09-27.md)、
[基准与图表](../benchmarks/README.md)、[配置全集](../examples/configs/README.md)、
[开发规范](../AGENTS.md)。专题中的历史实验数字属于各自记录日期，不表示本次重新训练。
