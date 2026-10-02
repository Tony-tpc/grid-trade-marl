# marl/training/implicit.py

[手册首页](../README.md) · [全部 API](README.md)

[打开源码](../../marl/training/implicit.py)

可复用的随机目标高阶估计和隐式响应梯度，不包含环境/博弈角色规则。

本页由 `doc/tools/build_api.py` 生成；签名来自源码，说明优先引用 docstring。

## 符号目录

- [magic_box](#magic_box)
- [leave_one_out_baseline](#leave_one_out_baseline)
- [dice_objective](#dice_objective)
- [flat_gradient](#flat_gradient)
- [ImplicitResponseConfig](#implicitresponseconfig)
- [ImplicitResponseConfig.__post_init__](#implicitresponseconfig-__post_init__)
- [LinearSolveAttempt](#linearsolveattempt)
- [ImplicitGradient](#implicitgradient)
- [implicit_response_gradient](#implicit_response_gradient)
- [implicit_response_gradient.product](#implicit_response_gradient-product)
- [implicit_response_gradient.callback](#implicit_response_gradient-callback)
- [gradient_surrogate](#gradient_surrogate)

<a id="magic_box"></a>

## magic_box

[源码位置](../../marl/training/implicit.py#L12) · [页内目录](#符号目录)

```python
def magic_box(log_prob: Tensor) -> Tensor
```

```text
DiCE 算子：前向恒为 1，任意阶导数保留随机节点 score。

Args:
    log_prob: 可微随机节点 log-prob；DiCE 为 [B,T,K]，MagicBox 保留任意输入形状。

Returns:
    与输入同形的 Tensor；前向为一，梯度为 score，不 detach 高阶依赖。
```

<a id="leave_one_out_baseline"></a>

## leave_one_out_baseline

[源码位置](../../marl/training/implicit.py#L24) · [页内目录](#符号目录)

```python
def leave_one_out_baseline(rewards: Tensor) -> Tensor
```

```text
用其他独立轨迹的即时奖励均值构造 detached baseline [B,T]。

Args:
    rewards: detached 有限浮点奖励 [B,T]，B>=2；各轨迹可共享外部场景，
        但不能共享动作样本。baseline 不依赖当前轨迹的任一动作。

Returns:
    [B,T]，第 b 行只使用其他 B-1 行，不跨时间取均值。
```

<a id="dice_objective"></a>

## dice_objective

[源码位置](../../marl/training/implicit.py#L41) · [页内目录](#符号目录)

```python
def dice_objective(log_prob: Tensor, rewards: Tensor, gamma: float=0.99, *, baseline: Tensor | None=None) -> Tensor
```

```text
联合随机轨迹的 DiCE 回报 [B,T,K] logp、[B,T] reward → 标量。

K 必须包含影响对应时刻奖励的全部随机策略节点。沿 T 的累积依赖表示因果前缀。
actions/rewards 必须已 detach；不把普通 PPO 的旧 advantage 误当跨策略计算图。
baseline 必须独立于该行全部随机动作；允许由其他独立轨迹构造。
使用 (MagicBox(prefix)-1)*(-baseline)，保持前向回报和高阶估计的期望。
不能直接把依赖本轨迹过去动作的 value/GAE 代入此全前缀 baseline。

Args:
    log_prob: 可微随机节点 log-prob；DiCE 为 [B,T,K]，MagicBox 保留任意输入形状。
    rewards: detached 环境回报 [B,T]；所有随机父节点须包含在 log_prob 中。
    gamma: 每物理步折扣，范围 [0,1]。
    baseline: 可选 detached [B,T] 控制变量；不依赖该行任一随机动作。

Returns:
    标量折扣回报估计；对 batch 取均值，对时间求和。
```

<a id="flat_gradient"></a>

## flat_gradient

[源码位置](../../marl/training/implicit.py#L81) · [页内目录](#符号目录)

```python
def flat_gradient(loss: Tensor, parameters: Sequence[nn.Parameter], *, create_graph: bool=False) -> Tensor
```

```text
标量 loss 对指定参数的展平梯度；无依赖项返回同形零，支持二阶零块。

Args:
    loss: 可微标量目标；不依赖参数时返回带零导数的同形向量。
    parameters: 有序的目标参数列表；展平顺序决定梯度坐标。
    create_graph: 是否保留梯度图，以便继续计算二阶导数。

Returns:
    一维 Tensor，长度等于参数总数；未依赖参数填零，不累积 .grad。
```

<a id="implicitresponseconfig"></a>

## ImplicitResponseConfig

[源码位置](../../marl/training/implicit.py#L104) · [页内目录](#符号目录)

`class ImplicitResponseConfig()`

数据字段、默认值与方法如下；构造参数由类字段或显式 __init__ 定义。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
damping: float = 0.001
residual_tolerance: float = 0.0001
max_iterations: int = 100
attempts: int = 3
damping_multiplier: float = 10.0
krylov_dimension: int = 100
```

<a id="implicitresponseconfig-__post_init__"></a>

## ImplicitResponseConfig.__post_init__

[源码位置](../../marl/training/implicit.py#L112) · [页内目录](#符号目录)

```python
def __post_init__(self) -> None
```

```text
绑定并验证本实例所负责的组件与状态。

Inputs:
    无显式参数；读取当前实例的配置或训练状态。

Returns:
    None；非法配置在构造时抛 ValueError，不修改训练状态。
```

<a id="linearsolveattempt"></a>

## LinearSolveAttempt

[源码位置](../../marl/training/implicit.py#L133) · [页内目录](#符号目录)

`class LinearSolveAttempt()`

单个阻尼系统的真实残差、Arnoldi 迭代数与 SciPy 返回状态。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
damping: float
residual: float
iterations: int
status: int
```

<a id="implicitgradient"></a>

## ImplicitGradient

[源码位置](../../marl/training/implicit.py#L142) · [页内目录](#符号目录)

`class ImplicitGradient()`

失败时 gradient=None，调用方必须明确跳过/报错，不得当零修正成功。

声明字段（dataclass 的 init 字段同时也是构造输入）：

```python
gradient: Tensor | None
direct: Tensor
correction: Tensor | None
residual: float
damping: float
iterations: int
converged: bool
attempts: tuple[LinearSolveAttempt, ...]
undamped_residual: float
```

<a id="implicit_response_gradient"></a>

## implicit_response_gradient

[源码位置](../../marl/training/implicit.py#L158) · [页内目录](#符号目录)

```python
def implicit_response_gradient(leader_loss: Tensor, follower_loss: Tensor, leader_parameters: Sequence[nn.Parameter], follower_parameters: Sequence[nn.Parameter], config: ImplicitResponseConfig=_DEFAULT_IMPLICIT_CONFIG) -> ImplicitGradient
```

```text
求 gu = Lu,u - Lc,cuᵀ (Lc,cc+mu I)^(-T) Lu,c，返回 detached 梯度。

复用 autograd HVP 和 SciPy GMRES；实测 residual 决定成功，不依赖 solver status。
Hessian 应来自可二次微分的标量，不要求正定。损失与参数须在同设备同 dtype。
mu I 定义局部正则化响应，不代表原始 best response；同时返回未加阻尼的残差。
在当前点 (u0,c0)，令 g0=stopgrad(grad_c Lc(u0,c0))，本方向对应局部方程
grad_c Lc(u,c)+mu*(c-c0)-g0=0 的导数。无需伪设 g0=0；H+mu I 仍须可逆。
solver CPU 向量转换产生同步开销；这是正确性优先的小规模实现。

Args:
    leader_loss: leader 最小化的可微标量目标，需保留对 follower 的 score 依赖。
    follower_loss: follower 最小化的可微标量目标，Hessian 和交叉块均来自此目标。
    leader_parameters: 外层参数序列；只返回这些参数的修订梯度。
    follower_parameters: 被求隐式响应的参数序列，不能与 leader 参数重叠。
    config: 冻结算法/求解配置，控制网络或 damping、残差阈值及迭代上限。

Returns:
    ImplicitGradient；成功含 detached 修订梯度，失败 gradient=None。
```

<a id="implicit_response_gradient-product"></a>

## implicit_response_gradient.product

[源码位置](../../marl/training/implicit.py#L203) · [页内目录](#符号目录)

```python
def product(vector: np.ndarray, damping: float=damping) -> np.ndarray
```

```text
一维 float64 NumPy 向量 (H+mu I)v。

Args:
    vector: GMRES 当前 CPU NumPy 向量，长度为 follower 参数总数。
    damping: 本次重试绑定的非负对角阻尼，不读取后续重试值。

Returns:
    一维 float64 NumPy 向量 (H+mu I)v；HVP 在模型设备和 dtype 下计算。
```

<a id="implicit_response_gradient-callback"></a>

## implicit_response_gradient.callback

[源码位置](../../marl/training/implicit.py#L219) · [页内目录](#符号目录)

```python
def callback(relative_residual: float) -> None
```

```text
计数每次 Arnoldi 内迭代；不修改模型参数。

Args:
    relative_residual: SciPy 的内迭代残差，仅用于回调计数；验收另行重算。

Returns:
    None；累计实际内迭代次数。
```

<a id="gradient_surrogate"></a>

## gradient_surrogate

[源码位置](../../marl/training/implicit.py#L259) · [页内目录](#符号目录)

```python
def gradient_surrogate(parameters: Sequence[nn.Parameter], gradient: Tensor) -> Tensor
```

```text
将已计算的梯度接入现有 optimize；不手写 parameter.data 更新。

Args:
    parameters: 有序的目标参数列表；展平顺序决定梯度坐标。
    gradient: 与 parameters 展平结果同形的有限梯度，将在代理目标中 detach。

Returns:
    标量代理目标，其反向梯度等于指定 gradient；不执行 optimizer.step。
```
