# 训练模块、数据生命周期与状态

[首页](README.md) · [算法](algorithms.md) · [训练教程](training.md) · [函数索引](api/README.md)

## 文件分工

| 文件 | 对象 | 职责 |
|---|---|---|
| [on_policy.py](api/marl-training-on_policy.md) | RolloutBuffer、PreparedRollout、OnPolicyTrainer | 当前策略采集、GAE、连续切片、消费保护 |
| [off_policy.py](api/marl-training-off_policy.md) | OffPolicyCollector、OffPolicyTrainer、ReplayConfig | 采集、记录 replay、抽样并委派更新 |
| [optimization.py](api/marl-training-optimization.md) | OptimizerRuntime | 参数归属、AMP、计数、指标与恢复 |
| [gradients.py](api/marl-training-gradients.md) | isolate_agent_action、frozen_parameters | 自己动作梯度、critic 参数冻结 |
| [checkpoint.py](api/marl-training-checkpoint.md) | 文件/校验/RNG 工具 | 两种 trainer 共用状态协议 |

Protocol 是调用方的能力契约，没有第二套实现。trainer 调 algorithm.update；算法编排
更新顺序；基类复用单次优化。

## On-policy 函数链

```text
train_rollout(seeds)
  collect(seeds)
    reset 环境；policy_state/value_state = None
    T 次：sample → environment.step → buffer.add
    values(最终观测，最后 critic 状态) → buffer.finish(next_value,estimator)
  PreparedRollout.to(device)
  algorithm.update(rollout,optimization)
```

[collect](api/marl-training-on_policy.md#onpolicytrainer-collect) 不更新参数。
采样在 inference_mode 中进行，普通缓存在其外创建；add 保存 detached 数据。
CPU 观测/state 不先上卡再拷回，CUDA 隐藏历史留原设备。

`add` 输入一个时刻的 `[B,N,O]` 观测、`[B,S]` state、动作、`[B,N]` 奖励、log-prob、
value、终止标记，返回 None 并推进位置。`finish` 必须在写满后调用一次，输入最终
`[B,N]` bootstrap value，输出带 advantage/return 的 PreparedRollout。

`_allocate` 分配 `[T,B,...]`，`_copy` 校验并复制 detached 数据，`_record_state`
记录处理当前观测之前的 h/c。这些函数不计算训练目标。

### 展平与循环 mini-batch

没有跨步循环 backbone：`[T,B,N,O] → [T*B,N,O]`，打乱环境时间位置。
这也适用于带历史 GRU/LSTM/Transformer encoder、后接 MLP 的网络：窗口 W 已包含在
每条观测中，窗口内顺序不被打乱，`update.sequence_length` 必须为 null。

MAPPO actor 或 critic 使用跨步 GRU/LSTM backbone：转为 `[B,T,N,O]`，按 L 步切片。
B=2、T=10、L=4 时共 6 段，每环境有效长度
4、4、2，尾段补齐 2 步。mini_batch_size=8 每批最多放 2 条长 4 片段，不是 8 个玩家
或 8 条轨迹。sequence_mask 排除 padding，补齐位置的动作 mask 仍须有合法动作。

`size` 是 B*T；开始推进 minibatches 后即 consumed，只允许这一次迭代内部的多 epoch。
`to` 成功后把消费权转给新对象，旧对象失效。手工 PreparedRollout 不自动 detach 外部数据。

### 终止与截断

terminated 禁止下一 value bootstrap。truncated 的当前 TD delta 仍使用最终观测 value，
但不跨 episode 递推 GAE。当前采集器不能在 rollout 中途 reset 后继续填。
环境提前结束会报错；可缩短采集长度，或先实现部分 reset 和逐 transition bootstrap。
不能删报错或使用 reset 后观测。每份新 rollout 都 reset 环境/h/c，短 horizon 不会续接。

## Off-policy 函数链

```text
OffPolicyCollector.rollout(seeds) → 每时刻 yield tuple[Transition,...]
trainer.record_many(transitions) → replay
trainer.ready → replay.sample(batch_size,rng) → algorithm.update(batch,runtime)
```

collector 不自动写 replay 或训练。action_transform 可加入探索，须保持动作形状/范围，
replay 记录实际执行动作。collector 不支持一部分环境先结束、其余环境继续的 episode 批次。

record/record_many 返回 None；ready 只判断样本达到 batch_size。update 抽样训练一次；
update_batch 直接接收并迁移外部批次，不自动写入 replay。warm-up、每多少新 transition
训练一次、每次多少梯度步属于实验调度，不能因此把 replay 接入 PPO。

历史网络仍保存原始观测而非缓存的编码特征。每次更新从抽中的观测重建完整窗口，
窗口内可反传，每条样本的 GRU/LSTM 初态为零；不跨 replay 条目携带 hidden state。
因此现有 transition replay 可用于窗口编码，但不支持跨环境步循环 backbone。
两种时间语义的形状与配置见[网络教程](networks.md#time).

## 优化器和日志

optimizer(name)、parameters(name)、max_grad_norm(name) 返回现有实例、缓存参数、阈值。
任何参数只归一个 optimizer，构造后参数集合不能变。record_optimizer_step 记录真实 step；
finish 记录一次算法 update 并同步目标网络。
PPO 4 epoch、每 epoch 2 个 mini-batch、每批 actor/critic 各 step 一次，则 update_count
增加 1，optimizer_step_count 增加 16。hard target interval 不是环境步数。

accumulate_metrics 普通项加权求和，*_count 累加，*_abs_max/*_nonfinite 取极值；
mean_metrics 除以权重，以真实事件计算裁剪率。MAPPO loss 按有效样本数加权，梯度按 step 平均。

默认 export_metrics 返回 CPU float；设置 optimization.sync_metrics=False 后 update
返回 {}，调用 flush_metrics 才同步并清空累计指标。空字典不代表没有训练。

## Checkpoint 边界

共有状态是模型、配置、`input_spec`、runtime、Torch CPU/CUDA RNG。trainer schema=5，
内部 OptimizerRuntime schema=3，不是同一个版本号。off-policy 另存 replay 和 NumPy RNG，
on-policy 另存 mini-batch generator 状态。`input_spec` 包含影响前向的历史/节点索引及邻接。
加载先检查版本、配置、完整 spec、模型键名/shape/dtype 和静态布局 buffer 值；布局变更即使
输入总维度相同也会拒绝加载。schema 4 及更早文件不静默部分恢复。
save_checkpoint 自动创建父目录并覆盖同名文件；load_checkpoint 必须在同配置实验上调用。
外部组件需预先提供相同 catalog 和实现。完整状态使用 pickle，只读可信本地文件。

MAPPO 只承诺完整 rollout 更新后恢复，不保存半段 rollout、活跃环境、外部 seed 调度。
on-policy 对损坏 optimizer/RNG 有回滚；off-policy/runtime 独立恢复不承诺完整事务回滚。
仅恢复模型权重属于评估或重新初始化实验，不是完整续训。
