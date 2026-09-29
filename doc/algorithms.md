# 五个算法：函数职责与数学含义

[首页](README.md) · [架构](architecture.md) · [训练器](trainers.md) · [函数索引](api/README.md)

## 能力与公共入口

| 算法 | 当前动作 | 奖励 | 经验及更新順序 | 全部函数 |
|---|---|---|---|---|
| MAPPO | 离散；MLP/GRU/LSTM | 个体或共享 | fresh rollout；actor → critic，多 epoch | [MAPPO](api/marl-algorithms-mappo.md) |
| MAAC | 离散；拒绝循环 policy | 个体或共享 | replay；critic → 各 agent actor → target | [MAAC](api/marl-algorithms-maac.md) |
| MADDPG | 连续 | 个体或共享 | replay；critic → actor → target | [MADDPG](api/marl-algorithms-maddpg.md) |
| MASAC | 连续 | 个体或共享 | replay；twin critic → actor → temperature → target | [MASAC](api/marl-algorithms-masac.md) |
| QMIX | 离散 | 仅共享团队奖励 | replay；value → target | [QMIX](api/marl-algorithms-qmix.md) |

当前没有混合动作、离策略循环 replay、变长存活智能体支持；不以算法名称推断论文变体。
各文件包含 LossConfig、Config 和具体类。`__post_init__` 检查范围，`validate(spec)`
检查兼容性，`build(spec)` 直接返回实例，`__init__` 注册网络/目标/target/统计模块。

`act` 返回动作；`compute_*_loss_bundle` 构图；`update` 执行优化顺序。总 loss 是诊断入口，
不要用一个含所有参数的 optimizer 同时训练 actor/critic/temperature。
[optimize](api/marl-algorithms-base.md#basemarlalgorithm-optimize) 统一清梯度、反传、裁剪和 step。

## MAPPO：一次数据，多轮有限更新

1. [sample](api/marl-algorithms-mappo.md#mappo-sample)：观测 `[...,N,O]`、state `[...,S]`
   → 动作、log-prob、entropy、value 和下一循环状态。
2. [values](api/marl-algorithms-mappo.md#mappo-values)：仅算 value；bootstrap 调用方关闭梯度。
3. [_validate_training_batch](api/marl-algorithms-mappo.md#mappo-_validate_training_batch)：
   检查 actions、old_log_prob、advantages、returns；循环训练须为连续 `[B,T,N,O]`。
4. [policy loss](api/marl-algorithms-mappo.md#mappo-compute_policy_loss_bundle)：评价已保存
   动作，计算 PPO 和熵。不能替换成新采样动作。
5. [value loss](api/marl-algorithms-mappo.md#mappo-compute_value_loss_bundle)：拟合 returns，
   可选围绕 old_values 做 value clipping。
6. [update](api/marl-algorithms-mappo.md#mappo-update)：检查消费权，固定本轮 target 统计，
   多 epoch/minibatch 更新 actor/critic，按有效位置汇总日志。

```text
rho = exp(new_log_prob - old_log_prob)
policy_loss = -mean(min(rho*advantage, clip(rho,1-eps,1+eps)*advantage))
entropy contribution = -entropy_coefficient * mean(entropy)
value contribution = value_coefficient * mean((value-return)^2)
```

上式的 mean 是各个体目标形成后的损失归约，不是把个体奖励平均成团队奖励。
value clipping 在原始 value 单位下进行。GAE return 在 advantage 标准化之前计算。
循环片段初态来自采样策略并 detach，片段内部反传，当前无 burn-in；更新后不重算片段
初态。在线循环策略使用 `sample` 传状态，`act` 会拒绝循环 actor。

## MAAC：注意力 Q 与反事实动作比较

[critic loss](api/marl-algorithms-maac.md#maac-compute_critic_loss_bundle) 用 replay 真实
联合动作计算 Q 表 `[...,N,A]`，gather 实际动作值，并拟合各自的 soft TD target：

```text
target_i = reward_i + gamma*(1-terminated_i)
           * (target_Q_i(next_joint_action) - entropy_coefficient*next_log_prob_i)
```

[actor loss](api/marl-algorithms-maac.md#maac-compute_actor_loss_bundle) 对单个 agent_index
读取 `[...,A]` logits，固定 Q 与其他玩家动作：

```text
baseline_i = sum_a pi_i(a)*Q_i(a)
actor_loss_i = -mean(sum_a detach(pi_i(a))*detach(Q_i(a)-baseline_i)*log pi_i(a))
```

这是 score-function 梯度；Q、概率权重和 baseline 不反传，另加熵正则。
[update](api/marl-algorithms-maac.md#maac-update) 先更新 critic，再无梯度计算一份 Q 表，
逐玩家更新 actor。内置单玩家前向隔离其他参数；外部拓扑清零梯度兜底防止 Adam 动量误动。

## MADDPG：确定性策略与动作梯度

actor_i 只看自己的观测。critic_i 看联合观测和动作，拼接输入 `[...,N*(O+A)]`，
输出自己的 Q_i。不同玩家保留独立 critic 参数与奖励标签。

```text
target_i = reward_i + gamma*(1-terminated_i)*target_Q_i(next_obs,target_actions)
actor contribution_i = -Q_i(obs, detached_other_actions, own_action)
```

[actor loss](api/marl-algorithms-maddpg.md#maddpg-compute_actor_loss_bundle) 通过
`isolate_agent_action` 保留自己动作梯度，通过 `frozen_parameters` 固定 critic 参数。
不能用 no_grad 包住该 critic 前向，否则 actor 也失去梯度。`act` 始终确定性，探索噪声
由采集器加入并裁剪到动作范围。

## MASAC：随机策略、双 Q、温度

每个智能体使用 tanh-Gaussian、两个独立 Q 和可学习 alpha_i。target 只有 critic，
下一动作来自当前策略，没有 target actor。

```text
soft_next_i = min(target_Q1_i,target_Q2_i) - alpha_i*next_log_prob_i
target_i = reward_i + gamma*(1-terminated_i)*soft_next_i
actor_loss = mean(detach(alpha_i)*log_prob_i - min(Q1_i,Q2_i))
alpha_loss = -mean(log_alpha_i*(mean_batch(detach(log_prob_i))+target_entropy))
```

[temperature loss](api/marl-algorithms-masac.md#masac-compute_temperature_loss_bundle)
在 actor 更新后重新无梯度采样，仅对 log_alpha 构图，不前向 critic。
默认目标熵 `-A` 是连续密度约定，不是离散最大熵。三个 optimizer 的参数不得重叠。

## QMIX：团队回报的单调分解

局部 Q 输出 `[...,N,A]`。mixer 读取选中动作 Q `[...,N]` 和 state `[...,S]`，输出
团队 Q_tot `[...,1]`；非负权重保证对各局部 Q 单调。
[compute_loss_bundle](api/marl-algorithms-qmix.md#qmix-compute_loss_bundle) 先验证所有
奖励相同，才能取第一列作为团队奖励。在线 Q 按合法动作选 argmax，target Q 评价并混合。
任一成员真正终止即停止团队 bootstrap，截断仍允许 bootstrap。
不能平均不同玩家收益来伪造共享奖励。`act` 始终贪心，epsilon 探索放在采集端。
`update` 用一个 value optimizer 联合训练在线 Q 与 mixer，然后同步 target 对。
