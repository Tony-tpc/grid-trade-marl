"""高阶修订的解析与枚举验收；训练曲线不能代替这些数学测试。"""
import pytest
import torch
from torch import nn

from marl.training.implicit import (
    ImplicitResponseConfig,
    dice_objective,
    flat_gradient,
    gradient_surrogate,
    implicit_response_gradient,
    leave_one_out_baseline,
    magic_box,
)


def test_correct_cross_block_and_finite_difference():
    u = nn.Parameter(torch.tensor(.3, dtype=torch.float64))
    c = nn.Parameter(torch.tensor(.6, dtype=torch.float64))
    lower = .5 * (c-2*u).square()
    upper = u.square()+.8*u+.5*c.square()+.4*c
    result = implicit_response_gradient(upper, lower, [u], [c], ImplicitResponseConfig(damping=0))
    assert result.converged and result.residual < 1e-10
    assert result.gradient.item() == pytest.approx(3.4)
    assert result.direct.item() == pytest.approx(1.4)  # 错用 upper 的交叉块会漏掉响应。
    h = 1e-5
    def f(x):
        return x*x + .8*x + .5*(2*x)**2 + .4*(2*x)
    assert result.gradient.item() == pytest.approx((f(.3+h)-f(.3-h))/(2*h))
    surrogate = gradient_surrogate([u], result.gradient)
    surrogate.backward()
    assert u.grad.item() == pytest.approx(3.4)


def test_diagonal_indefinite_hessian_and_zero_coupling():
    u = nn.Parameter(torch.tensor([.1, .2], dtype=torch.float64))
    c = nn.Parameter(torch.tensor([.2, .4], dtype=torch.float64))
    lower = .5*c[0]**2-c[1]**2
    upper = (u**2).sum()+(u*c).sum()
    result = implicit_response_gradient(upper, lower, [u], [c], ImplicitResponseConfig(damping=0))
    assert result.converged
    torch.testing.assert_close(result.gradient, 2*u+c)
    assert result.correction.abs().sum() == 0


def test_regularized_local_response_has_defined_derivative_away_from_stationarity():
    u = nn.Parameter(torch.tensor(.3, dtype=torch.float64))
    c = nn.Parameter(torch.tensor(.1, dtype=torch.float64))  # c != 2*u，明确不在 best response。
    mu = .05
    lower = .5*(c-2*u).square()
    upper = u.square()+.8*u+.5*c.square()+.4*c
    result = implicit_response_gradient(upper, lower, [u], [c],
                                        ImplicitResponseConfig(damping=mu))
    g0, c0 = float((c-2*u).detach()), float(c.detach())

    def local_upper(x):
        response = (2*x+g0+mu*c0)/(1+mu)
        return x*x+.8*x+.5*response**2+.4*response
    h = 1e-5
    assert result.converged
    assert result.gradient.item() == pytest.approx((local_upper(.3+h)-local_upper(.3-h))/(2*h))
    assert result.undamped_residual == pytest.approx(mu/(1+mu))


def test_failed_solver_is_explicit_and_leaves_parameters_untouched():
    u = nn.Parameter(torch.tensor(1., dtype=torch.float64))
    c = nn.Parameter(torch.arange(1., 5., dtype=torch.float64))
    before = c.clone()
    result = implicit_response_gradient(u*c.sum(), (.5*c*c*torch.arange(1., 5.)).sum(),
                                        [u], [c], ImplicitResponseConfig(
                                            damping=0, max_iterations=1, attempts=1,
                                            residual_tolerance=1e-12))
    assert not result.converged and result.gradient is None
    torch.testing.assert_close(c, before)


def test_dice_joint_expectation_matches_exact_first_and_second_derivatives():
    u = nn.Parameter(torch.tensor(.3, dtype=torch.float64))
    c = nn.Parameter(torch.tensor(-.2, dtype=torch.float64))
    logp, rewards, weights = [], [], []
    for a in (0., 1.):
        for b in (0., 1.):
            first = torch.distributions.Bernoulli(logits=u)
            second = torch.distributions.Bernoulli(logits=c)
            lp = torch.stack([first.log_prob(torch.tensor(a, dtype=u.dtype)),
                              second.log_prob(torch.tensor(b, dtype=c.dtype))])
            logp.append(lp)
            rewards.append(3*a*b+a-2*b)
            weights.append(lp.sum().exp().detach())
    joint = torch.stack(logp)
    reward = torch.tensor(rewards, dtype=torch.float64)
    # 穷举的采样权重只用于消除 Monte Carlo 方差，不参与 score 微分。
    estimate = (torch.stack(weights)*magic_box(joint.sum(-1))*reward).sum()
    exact = 3*u.sigmoid()*c.sigmoid()+u.sigmoid()-2*c.sigmoid()
    got = flat_gradient(estimate, [u, c], create_graph=True)
    expected = flat_gradient(exact, [u, c], create_graph=True)
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(flat_gradient(got.sum(), [u, c]),
                               flat_gradient(expected.sum(), [u, c]))
    assert dice_objective(joint[:, None, :], reward[:, None]).item() == pytest.approx(reward.mean())


def test_leave_one_out_causal_estimator_exact_gradient_and_full_hessian():
    """两条独立两步轨迹穷举；第二动作条件于第一动作，检查全部交叉项。"""
    from itertools import product
    u = nn.Parameter(torch.tensor(.3, dtype=torch.float64))
    c = nn.Parameter(torch.tensor(-.2, dtype=torch.float64))
    estimated = u*0
    gamma = .9
    for outcomes in product((0., 1.), repeat=4):
        logps, rewards = [], []
        for a, b in (outcomes[:2], outcomes[2:]):
            pa = torch.distributions.Bernoulli(logits=u)
            pb = torch.distributions.Bernoulli(logits=c+.3*a)
            logps.append(torch.stack([pa.log_prob(u.new_tensor(a)),
                                      pb.log_prob(c.new_tensor(b))]))
            rewards.append([.7*a, 3*a*b+a-2*b])
        joint = torch.stack(logps).unsqueeze(-1)
        reward = u.new_tensor(rewards)
        baseline = leave_one_out_baseline(reward)
        estimate = dice_objective(joint, reward, gamma, baseline=baseline)
        torch.testing.assert_close(estimate, (reward*u.new_tensor([1, gamma])).sum(1).mean())
        estimated = estimated+joint.sum().exp().detach()*estimate
    p, q0, q1 = u.sigmoid(), c.sigmoid(), (c+.3).sigmoid()
    exact = .7*p+gamma*(p*(1+q1)-2*(1-p)*q0)
    got, expected = [flat_gradient(x, [u, c], create_graph=True) for x in (estimated, exact)]
    torch.testing.assert_close(got, expected)
    for i in range(2):
        torch.testing.assert_close(flat_gradient(got[i], [u, c]),
                                   flat_gradient(expected[i], [u, c]))


def test_baseline_removes_constant_reward_noise_and_rejects_bad_inputs():
    theta = nn.Parameter(torch.tensor(.3, dtype=torch.float64))
    actions = theta.new_tensor([[1., 1., 0.], [0., 0., 0.]])
    joint = torch.distributions.Bernoulli(logits=theta).log_prob(actions).unsqueeze(-1)
    reward = theta.new_tensor([[3., 7., 11.], [3., 7., 11.]])
    corrected = dice_objective(joint, reward, baseline=leave_one_out_baseline(reward))
    grad = flat_gradient(corrected, [theta], create_graph=True)
    torch.testing.assert_close(grad, torch.zeros_like(grad))
    torch.testing.assert_close(flat_gradient(grad.sum(), [theta]), torch.zeros_like(grad))
    assert flat_gradient(dice_objective(joint, reward), [theta]).abs().item() > 1
    with pytest.raises(ValueError, match='baseline'):
        leave_one_out_baseline(reward[:1])
    with pytest.raises(ValueError, match='baseline'):
        dice_objective(joint, reward, baseline=reward.clone().requires_grad_())


def test_gmres_resolves_large_scale_range_and_actual_residual_controls_acceptance(monkeypatch):
    import numpy as np
    import scipy.sparse.linalg  # type: ignore[import-untyped]
    u = nn.Parameter(torch.tensor(1., dtype=torch.float64))
    c = nn.Parameter(torch.ones(3, dtype=torch.float64))
    diagonal = c.new_tensor([1e-6, 1., 1e6])
    lower = .5*(diagonal*c.square()).sum()-u*c.sum()
    upper = u.square()+c.sum()
    config = ImplicitResponseConfig(damping=0, residual_tolerance=1e-8, attempts=1)
    result = implicit_response_gradient(upper, lower, [u], [c], config)
    assert result.converged and result.residual < 1e-8
    torch.testing.assert_close(result.gradient, (2*u+diagonal.reciprocal().sum()).reshape(1))

    def false_success(operator, rhs, **kwargs):
        return np.zeros_like(rhs), 0
    monkeypatch.setattr(scipy.sparse.linalg, 'gmres', false_success)
    result = implicit_response_gradient(upper, lower, [u], [c], config)
    assert not result.converged and result.gradient is None
    assert result.attempts[0].status == 0 and result.residual == 1
