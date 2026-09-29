# marl/modules/actor.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/modules/actor.py)

模块职责见类与函数说明。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [Actor](#actor)
- [Actor.__init__](#actor-__init__)
- [Actor._attach_hidden_state](#actor-_attach_hidden_state)
- [Actor._encode](#actor-_encode)
- [Actor.forward](#actor-forward)
- [Actor.evaluate_actions](#actor-evaluate_actions)
- [Actor.discrete_logits](#actor-discrete_logits)

<a id="actor"></a>

## Actor

[源码位置](../../marl/modules/actor.py#L10) · [页内目录](#符号目录)

`class Actor(nn.Module)`

单智能体策略网络：``observation -> backbone -> action head``。

Actor 不知道智能体数量和参数共享关系；这些多智能体拓扑由
:mod:`marl.modules.policy` 负责。输入 ``[..., O]``，输出动作、log-prob、
entropy 和分布参数；hidden_state 单独返回，不是分布参数。

<a id="actor-__init__"></a>

## Actor.__init__

[源码位置](../../marl/modules/actor.py#L18) · [页内目录](#符号目录)

```python
def __init__(self, backbone: BaseBackbone, action_head: BaseActionHead) -> None
```

构造 Actor，按下方参数初始化网络子模块或运行状态；返回 None。

<a id="actor-_attach_hidden_state"></a>

## Actor._attach_hidden_state

[源码位置](../../marl/modules/actor.py#L23) · [页内目录](#符号目录)

```python
def _attach_hidden_state(result: ActionHeadOutput, hidden_state: RecurrentState) -> ActionHeadOutput
```

将 backbone 输出的新状态写入动作输出的 hidden_state，返回该动作输出。

<a id="actor-_encode"></a>

## Actor._encode

[源码位置](../../marl/modules/actor.py#L29) · [页内目录](#符号目录)

```python
def _encode(self, observations: Tensor, hidden_state: RecurrentState, **kwargs: Tensor) -> BackboneOutput
```

校验输入布局并调用 backbone，返回带 features 和隐藏状态的 BackboneOutput。

<a id="actor-forward"></a>

## Actor.forward

[源码位置](../../marl/modules/actor.py#L42) · [页内目录](#符号目录)

```python
def forward(self, observations: Tensor, *, deterministic: bool=False, action_mask: Tensor | None=None, hidden_state: RecurrentState=None, **backbone_kwargs: Tensor) -> ActionHeadOutput
```

采样或确定性选择一个动作。

<a id="actor-evaluate_actions"></a>

## Actor.evaluate_actions

[源码位置](../../marl/modules/actor.py#L58) · [页内目录](#符号目录)

```python
def evaluate_actions(self, observations: Tensor, actions: Tensor, *, action_mask: Tensor | None=None, hidden_state: RecurrentState=None, **backbone_kwargs: Tensor) -> ActionHeadOutput
```

在同一 Actor 中评估给定动作，供 PPO 等 on-policy 目标使用。

<a id="actor-discrete_logits"></a>

## Actor.discrete_logits

[源码位置](../../marl/modules/actor.py#L75) · [页内目录](#符号目录)

```python
def discrete_logits(self, observations: Tensor, *, action_mask: Tensor | None=None, hidden_state: RecurrentState=None, **backbone_kwargs: Tensor) -> Tensor
```

只计算离散策略参数，不采样动作或构造分布。
