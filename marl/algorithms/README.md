# MARL 算法源码阅读导引

这份说明用于辅助阅读同目录下的算法实现。建议先阅读 `base.py`，再按
`MAPPO -> MADDPG -> MASAC -> QMIX -> MAAC` 的顺序阅读。

## 1. 常用符号和张量形状

| 符号 | 含义 | 典型张量 |
|---|---|---|
| B | batch size，一次训练使用的样本数 | 第一维 |
| N | 智能体数量 | agent 维 |
| O | 单个智能体的局部观测维度 | observations 最后一维 |
| A | 动作维度或离散动作数量 | actions/logits 最后一维 |
| S | 全局 state 维度，仅训练时可用 | state 最后一维 |

主要数据：

- `observations [B,N,O]`：每个智能体只能看到的局部信息。
- `actions [B,N]`：离散动作编号。
- `actions [B,N,A]`：连续动作向量。
- `rewards [B,N]`：每个智能体的奖励。个体收益博弈保留每个 `r_i`；
  QMIX 等纯合作算法应由环境明确提供相同团队奖励。
- `dones [B,N]`：值为 1 表示 episode 已终止，不能继续 bootstrap。
- `state [B,S]`：训练 Critic/Mixer 时可使用的全局信息。

`...` 或 `*B` 表示前面可以有多个维度。例如 rollout 可能使用
`[time,batch,agent,observation]`。

## 2. Actor、Critic 和 Q 值

- Actor 是策略：输入观测，输出动作或动作概率。
- Critic 是评分器：估计当前状态/动作能带来多少未来累计奖励。
- `V(s)` 评价状态，不指定当前动作。
- `Q(s,a)` 评价状态与动作的组合。
- Advantage `A(s,a)=Q(s,a)-V(s)` 表示某动作比平均水平好多少。

Actor-Critic 的基本循环是：Critic 学习更准确地评分，Actor 再根据评分提高好动作的
概率。它们需要同时学习，因此目标不断变化；target network、PPO clipping 和 twin-Q
都是为了让这个过程更稳定。

## 3. CTDE

CTDE 是 centralized training, decentralized execution：

- 训练时 Critic 可以看所有智能体的观测、动作或全局 state；
- 执行时每个 Actor 只能使用自己的局部观测。

这既让 Critic 更容易判断联合行为，也保证部署时智能体不必共享全部信息。MADDPG、
MAPPO、MASAC 和 MAAC 都采用这种思路；QMIX 则在训练阶段用 Mixer 和全局 state。

## 4. PyTorch 梯度概念

### `backward()`

前向计算会创建计算图。`loss.backward()` 从 loss 反向应用链式法则，把结果累加到每个
参数的 `.grad`。之后 `optimizer.step()` 才真正修改参数。

### `torch.no_grad()`

在代码块中完全不记录计算图。适合 target network 和环境推理，因为这些结果只作为
标签或动作使用，不需要被训练。

### `detach()`

返回共享数据但切断历史计算图的 Tensor。比如 Actor loss 使用 Critic 的评分，却不希望
该 loss 更新 Critic 时，可以对评分或某些系数 `detach()`。

### 冻结参数

`BaseMARLAlgorithm.frozen(critic)` 暂时把 Critic 参数设为不求梯度，但保留输出对输入
action 的梯度。这样梯度能穿过 Critic 回到 Actor，却不会错误地由 Actor loss 更新
Critic 参数。

### `gather`

Q 网络常输出 `[B,N,A]`，包含所有动作的 Q。实际执行动作是 `[B,N]` 的索引：

```python
chosen_q = all_q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
```

其作用是从 A 个候选值中选出 replay 中实际执行动作对应的 Q。

## 5. Target network 与 TD target

TD 学习的常见一步目标是：

```text
target = reward + gamma * (1 - done) * next_value
```

如果 next_value 由同一个快速变化的在线网络产生，标签本身也会剧烈变化。目标网络是
在线网络的慢速副本，使用 Polyak 平均更新：

```text
target_param = (1 - tau) * target_param + tau * online_param
```

MADDPG、MASAC、QMIX 和 MAAC 使用目标网络；on-policy 的 MAPPO 不需要。

## 6. 五个实现之间的差异

| 算法 | 动作 | 数据 | 价值函数 | 稳定机制 |
|---|---|---|---|---|
| MAPPO | 离散随机 | on-policy rollout | 逐智能体集中式 V_i | 概率比裁剪、熵奖励 |
| MADDPG | 连续确定性 | replay buffer | 每个玩家的集中式 Q_i | target network |
| MASAC | 连续随机 | replay buffer | 每个玩家的集中式 twin-Q_i | 熵目标、双 Q、target critic |
| QMIX | 离散贪心 | replay buffer | 个体 Q + 团队 Mixer | 只适用于共享奖励合作任务 |
| MAAC | 离散随机 | replay buffer | 逐智能体 Attention Q_i | 反事实 baseline、target network |

## 7. 一次训练更新

MAPPO 使用完整的 fresh on-policy 链路：`OnPolicyTrainer` 采集固定 horizon 的 rollout，
保存 old log-prob/value，按 termination/truncation 语义计算 GAE，再由 `PPOUpdatePlan`
随机生成 mini-batch 并执行多个 epoch。一次 rollout 更新后即失效，不能进入长期 replay。

MAAC、MADDPG、MASAC、QMIX 使用 `OffPolicyTrainer`：trainer 持有 replay buffer，
`OffPolicyUpdatePlan` 统一执行 `LossBundle.total.backward()`、梯度裁剪、optimizer step
和 recipe 选择的 soft/hard target update。具体算法只实现 `compute_loss_bundle()` 和
`target_pairs()`，不直接更新参数。

## 8. 当前基础版本的边界

这些实现用于提供清晰、可修改的算法核心。正式复现实验还需要根据论文和环境决定：

- replay buffer 和采样比例；
- observation/reward normalization；
- recurrent policy 的 hidden state 和 episode mask；
- 异质智能体是否分别使用 Actor；
- termination 与 time-limit truncation 的区别；
- 多个优化器、学习率调度、评估和 checkpoint 频率。
