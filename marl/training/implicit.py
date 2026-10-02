"""可复用的随机目标高阶估计和隐式响应梯度，不包含环境/博弈角色规则。"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn


def magic_box(log_prob: Tensor) -> Tensor:
    """DiCE 算子：前向恒为 1，任意阶导数保留随机节点 score。

    Args:
        log_prob: 可微随机节点 log-prob；DiCE 为 [B,T,K]，MagicBox 保留任意输入形状。

    Returns:
        与输入同形的 Tensor；前向为一，梯度为 score，不 detach 高阶依赖。
    """
    return torch.exp(log_prob - log_prob.detach())


def leave_one_out_baseline(rewards: Tensor) -> Tensor:
    """用其他独立轨迹的即时奖励均值构造 detached baseline [B,T]。

    Args:
        rewards: detached 有限浮点奖励 [B,T]，B>=2；各轨迹可共享外部场景，
            但不能共享动作样本。baseline 不依赖当前轨迹的任一动作。

    Returns:
        [B,T]，第 b 行只使用其他 B-1 行，不跨时间取均值。
    """
    if (rewards.ndim != 2 or rewards.shape[0] < 2 or rewards.shape[1] < 1
            or rewards.requires_grad or not rewards.is_floating_point()
            or not torch.isfinite(rewards).all()):
        raise ValueError('baseline 需要 detached 有限浮点奖励 [B>=2,T>=1]')
    return (rewards.sum(0, keepdim=True)-rewards)/(rewards.shape[0]-1)


def dice_objective(log_prob: Tensor, rewards: Tensor, gamma: float = .99, *,
                   baseline: Tensor | None = None) -> Tensor:
    """联合随机轨迹的 DiCE 回报 [B,T,K] logp、[B,T] reward → 标量。

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
    """
    if log_prob.ndim != 3 or rewards.shape != log_prob.shape[:2]:
        raise ValueError('DiCE 需要 log_prob [B,T,K] 和 rewards [B,T]')
    if not 0 <= gamma <= 1 or rewards.requires_grad:
        raise ValueError('gamma 必须在 [0,1]；环境 rewards 必须 detached')
    if log_prob.shape[0] == 0 or log_prob.shape[1] == 0:
        raise ValueError('DiCE 不接受空轨迹')
    if (log_prob.device != rewards.device or log_prob.dtype != rewards.dtype
            or not log_prob.is_floating_point() or not torch.isfinite(log_prob).all()
            or not torch.isfinite(rewards).all()):
        raise ValueError('DiCE log_prob/rewards 必须是同设备同 dtype 的有限浮点张量')
    if baseline is not None and (baseline.shape != rewards.shape or baseline.requires_grad
            or baseline.device != rewards.device or baseline.dtype != rewards.dtype
            or not torch.isfinite(baseline).all()):
        raise ValueError('baseline 必须为同设备同 dtype 的 detached 有限 [B,T]')
    prefix = log_prob.sum(-1).cumsum(-1)
    discount = gamma ** torch.arange(rewards.shape[1], device=rewards.device, dtype=rewards.dtype)
    box = magic_box(prefix)
    terms = box*rewards if baseline is None else box*(rewards-baseline)+baseline
    return (terms * discount).sum(-1).mean()


def flat_gradient(loss: Tensor, parameters: Sequence[nn.Parameter], *,
                  create_graph: bool = False) -> Tensor:
    """标量 loss 对指定参数的展平梯度；无依赖项返回同形零，支持二阶零块。

    Args:
        loss: 可微标量目标；不依赖参数时返回带零导数的同形向量。
        parameters: 有序的目标参数列表；展平顺序决定梯度坐标。
        create_graph: 是否保留梯度图，以便继续计算二阶导数。

    Returns:
        一维 Tensor，长度等于参数总数；未依赖参数填零，不累积 .grad。
    """
    if not parameters or loss.ndim:
        raise ValueError('需要非空参数列表和标量 loss')
    if not loss.requires_grad:
        return torch.cat([p.reshape(-1)*0 for p in parameters])
    grads = torch.autograd.grad(loss, parameters, create_graph=create_graph,
                                retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else p*0).reshape(-1)
                      for p, g in zip(parameters, grads, strict=True)])


@dataclass(frozen=True, slots=True)
class ImplicitResponseConfig:
    damping: float = 1e-3
    residual_tolerance: float = 1e-4
    max_iterations: int = 100
    attempts: int = 3
    damping_multiplier: float = 10.0
    krylov_dimension: int = 100

    def __post_init__(self) -> None:
        """绑定并验证本实例所负责的组件与状态。

        Inputs:
            无显式参数；读取当前实例的配置或训练状态。

        Returns:
            None；非法配置在构造时抛 ValueError，不修改训练状态。
        """
        if not np.isfinite([self.damping, self.residual_tolerance,
                            self.damping_multiplier]).all():
            raise ValueError('求解配置必须有限')
        if self.damping < 0 or not 0 < self.residual_tolerance < 1:
            raise ValueError('damping >= 0，residual_tolerance 在 (0,1)')
        if (any(type(v) is not int or v < 1 for v in
                (self.max_iterations, self.attempts, self.krylov_dimension))
                or self.damping_multiplier <= 1):
            raise ValueError('iterations/attempts 为正，damping_multiplier > 1')


@dataclass(frozen=True, slots=True)
class LinearSolveAttempt:
    """单个阻尼系统的真实残差、Arnoldi 迭代数与 SciPy 返回状态。"""
    damping: float
    residual: float
    iterations: int
    status: int


@dataclass(frozen=True, slots=True)
class ImplicitGradient:
    """失败时 gradient=None，调用方必须明确跳过/报错，不得当零修正成功。"""
    gradient: Tensor | None
    direct: Tensor
    correction: Tensor | None
    residual: float
    damping: float
    iterations: int
    converged: bool
    attempts: tuple[LinearSolveAttempt, ...]
    undamped_residual: float


_DEFAULT_IMPLICIT_CONFIG = ImplicitResponseConfig()


def implicit_response_gradient(
    leader_loss: Tensor,
    follower_loss: Tensor,
    leader_parameters: Sequence[nn.Parameter],
    follower_parameters: Sequence[nn.Parameter],
    config: ImplicitResponseConfig = _DEFAULT_IMPLICIT_CONFIG,
) -> ImplicitGradient:
    """求 gu = Lu,u - Lc,cuᵀ (Lc,cc+mu I)^(-T) Lu,c，返回 detached 梯度。

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
    """
    from scipy.sparse.linalg import LinearOperator, gmres  # type: ignore[import-untyped]

    if {id(p) for p in leader_parameters} & {id(p) for p in follower_parameters}:
        raise ValueError('上下层参数不能重叠')

    direct = flat_gradient(leader_loss, leader_parameters).detach()
    rhs = flat_gradient(leader_loss, follower_parameters).detach()
    follower_gradient = flat_gradient(follower_loss, follower_parameters, create_graph=True)
    if (not torch.isfinite(rhs).all() or not torch.isfinite(direct).all()
            or not torch.isfinite(follower_gradient).all()):
        raise ValueError('目标梯度非有限')
    rhs_array = rhs.cpu().double().numpy()
    damping, iterations, residual = config.damping, 0, float('inf')
    records: list[LinearSolveAttempt] = []
    undamped_residual = float('inf')
    for attempt in range(config.attempts):
        if attempt:
            damping = max(damping * config.damping_multiplier, 1e-8)

        def product(vector: np.ndarray, damping: float = damping) -> np.ndarray:
            """一维 float64 NumPy 向量 (H+mu I)v。

            Args:
                vector: GMRES 当前 CPU NumPy 向量，长度为 follower 参数总数。
                damping: 本次重试绑定的非负对角阻尼，不读取后续重试值。

            Returns:
                一维 float64 NumPy 向量 (H+mu I)v；HVP 在模型设备和 dtype 下计算。
            """
            v = torch.as_tensor(vector, dtype=rhs.dtype, device=rhs.device)
            hvp = flat_gradient((follower_gradient*v).sum(), follower_parameters).detach()
            if not torch.isfinite(hvp).all():
                raise ValueError('Hessian-vector product 非有限')
            return np.asarray(hvp.cpu().double().numpy())+damping*vector

        def callback(relative_residual: float) -> None:
            """计数每次 Arnoldi 内迭代；不修改模型参数。

            Args:
                relative_residual: SciPy 的内迭代残差，仅用于回调计数；验收另行重算。

            Returns:
                None；累计实际内迭代次数。
            """
            nonlocal iterations
            iterations += 1

        operator = LinearOperator((rhs.numel(), rhs.numel()), matvec=product, dtype=np.float64)
        before = iterations
        # legacy 只用于 SciPy 的计数定义：maxiter 是内迭代数，不是 restart 周期数。
        # 显式 Arnoldi 正交化改善短递推 MINRES 在有误差的 HVP 上的收敛表现。
        solution, status = gmres(
            operator, rhs_array, rtol=config.residual_tolerance*.5, atol=0.,
            restart=min(config.krylov_dimension, config.max_iterations),
            maxiter=config.max_iterations, callback=callback, callback_type='legacy',
        )
        residual = float(np.linalg.norm(product(solution)-rhs_array)
                         / max(np.linalg.norm(rhs_array), np.finfo(float).tiny))
        undamped_residual = float(np.linalg.norm(product(solution)-damping*solution-rhs_array)
                                  / max(np.linalg.norm(rhs_array), np.finfo(float).tiny))
        records.append(LinearSolveAttempt(damping, residual, iterations-before, int(status)))
        if np.isfinite(solution).all() and residual <= config.residual_tolerance:
            response = torch.as_tensor(solution, dtype=rhs.dtype, device=rhs.device).detach()
            correction = flat_gradient(
                (follower_gradient*response).sum(), leader_parameters
            ).detach()
            gradient = direct-correction
            if torch.isfinite(gradient).all():
                return ImplicitGradient(gradient, direct, correction, residual,
                                        damping, iterations, True, tuple(records),
                                        undamped_residual)
    return ImplicitGradient(None, direct, None, residual, damping, iterations, False,
                            tuple(records), undamped_residual)


def gradient_surrogate(parameters: Sequence[nn.Parameter], gradient: Tensor) -> Tensor:
    """将已计算的梯度接入现有 optimize；不手写 parameter.data 更新。

    Args:
        parameters: 有序的目标参数列表；展平顺序决定梯度坐标。
        gradient: 与 parameters 展平结果同形的有限梯度，将在代理目标中 detach。

    Returns:
        标量代理目标，其反向梯度等于指定 gradient；不执行 optimizer.step。
    """
    flat = torch.cat([p.reshape(-1) for p in parameters])
    if gradient.shape != flat.shape or not torch.isfinite(gradient).all():
        raise ValueError('gradient 维度不匹配或非有限')
    return (flat * gradient.detach()).sum()
