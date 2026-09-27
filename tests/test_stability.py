"""稳定计算、逐角色统计及真实裁剪事件的独立回归。"""
from dataclasses import replace

import pytest
import torch

from benchmarks.benchmark_noncooperative import BenchmarkSettings, _algorithm_config
from marl.algorithms import MAAC, MADDPG, MAPPO, MASAC
from marl.config import algorithm_config_from_dict, config_to_dict
from marl.core import MARLBatch
from marl.envs import ActionKind, EnvironmentSpec, RewardStructure
from marl.modules.action_head import GaussianActionHead
from marl.objectives import ClippedValueObjective
from marl.training import PreparedRollout
from marl.training.optimization import OptimizerRuntime
from marl.value_scaling import TargetScale, value_diagnostics


def test_saturated_tanh_density_and_gradient() -> None:
    head = GaussianActionHead(1, 1).double()
    with torch.no_grad():
        head.mean_layer.weight.fill_(1)
        head.mean_layer.bias.zero_()
    x = torch.tensor([0., -5., 5., -10., 10., -20., 20.], dtype=torch.float64)[:, None]
    x.requires_grad_()
    output = head(x, deterministic=True)
    # log(1-tanh(x)^2) = -2 log(cosh(x))；float64 中 cosh(20) 有限。
    expected = -0.5 * torch.log(torch.tensor(2 * torch.pi, dtype=x.dtype)) + 2 * x.cosh().log()
    torch.testing.assert_close(output.log_prob, expected.squeeze(-1), rtol=1e-12, atol=1e-12)
    gradient, = torch.autograd.grad(output.log_prob.sum(), x)
    torch.testing.assert_close(gradient, 2 * x.tanh(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(output.entropy, -output.log_prob)
    assert "gaussian_entropy" in output.distribution_params


def test_target_statistics_chunking_units_and_checkpoint() -> None:
    data = torch.tensor([[1., 1000.], [3., 2000.], [5., 3000.], [7., 4000.]])
    scale = TargetScale(2, True)
    scale.update(data[:1])
    scale.update(data[1:])
    torch.testing.assert_close(scale.mean, data.double().mean(0))
    torch.testing.assert_close(scale.scale(), data.double().std(0, unbiased=False))
    prediction = data.clone().requires_grad_()
    loss = (scale(prediction) - scale(data + 1)).square().mean()
    loss.backward()
    torch.testing.assert_close(prediction.grad, -2 / (data.numel() * data.var(0, unbiased=False))
                               * torch.ones_like(data))
    restored = TargetScale(2, True)
    restored.load_state_dict(scale.state_dict())
    torch.testing.assert_close(restored(data), scale(data))
    disabled = TargetScale(2)
    assert disabled.state_dict() == {}
    assert disabled(data) is data


def test_value_clipping_uses_raw_units_when_scaling() -> None:
    values = torch.tensor([[5., 50.]], requires_grad=True)
    old, target = torch.tensor([[4., 40.]]), torch.tensor([[6., 60.]])
    scale = torch.tensor([2., 10.])
    result = ClippedValueObjective(1., 0.2)(values, old, target, scale=scale)
    clipped = torch.tensor([[4.2, 40.2]])
    expected = torch.maximum((values-target).square(), (clipped-target).square()) / scale.square()
    torch.testing.assert_close(result.loss, expected.mean())


def test_role_variance_does_not_hide_prediction_bias() -> None:
    targets = torch.tensor([[1000., -1000.], [1002., -1002.]])
    predictions = torch.tensor([[1010., -1010.], [1010., -1010.]])
    metrics = value_diagnostics(predictions, targets)
    assert metrics["explained_variance"].item() == 0
    assert metrics["value_bias_agent_0"].item() == 9
    assert metrics["value_rmse_agent_0"].item() == pytest.approx(82 ** 0.5)


def test_clip_events_peak_nonfinite_and_restore() -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.))
    runtime = OptimizerRuntime({"actor": torch.optim.Adam([parameter])},
                               {"actor": 10.}, sync_metrics=False)
    for value in [1., 11., 100.]:
        runtime.record_optimizer_step("actor", torch.tensor(value))
        runtime.export_metrics({"loss": torch.tensor(value)})
    restored = OptimizerRuntime({"actor": torch.optim.Adam([parameter])}, {"actor": 10.})
    restored.load_state_dict(runtime.state_dict())
    metrics = restored.flush_metrics()
    assert metrics["actor_gradient_clip_rate"] == pytest.approx(2/3)
    assert metrics["actor_step_count"] == 3
    assert metrics["actor_gradient_abs_max"] == 100
    runtime.record_optimizer_step("actor", torch.tensor(float("inf")))
    runtime.export_metrics({"loss": torch.tensor(1.)})
    assert runtime.flush_metrics()["actor_gradient_nonfinite"] == 1


@pytest.mark.parametrize("name", ["maac", "mappo", "maddpg", "masac"])
def test_stability_config_roundtrip(name: str) -> None:
    settings = replace(BenchmarkSettings(), layer_norm=True, normalize_targets=True)
    config = _algorithm_config(name, settings)
    assert algorithm_config_from_dict(config_to_dict(config)) == config


@pytest.mark.parametrize("name", ["maac", "mappo", "maddpg", "masac"])
def test_statistics_update_once_and_diagnostic_loss_is_read_only(name: str) -> None:
    discrete = name in ("maac", "mappo")
    kind = ActionKind.DISCRETE if discrete else ActionKind.CONTINUOUS
    spec = EnvironmentSpec(2, 3, 4, 6, kind, 8, RewardStructure.INDIVIDUAL)
    config = _algorithm_config(name, BenchmarkSettings(normalize_targets=True))
    algorithm = config.build(spec)
    assert isinstance(algorithm, (MAAC, MADDPG, MAPPO, MASAC))
    critic = algorithm.get_submodule("critic" if isinstance(algorithm, (MAAC, MAPPO))
                                    else "critics")
    optimizers = {"actor": torch.optim.Adam(algorithm.policy.parameters()),
                  "critic": torch.optim.Adam(critic.parameters())}
    if isinstance(algorithm, MASAC):
        optimizers["temperature"] = torch.optim.Adam(algorithm.entropy_objective.parameters())
    runtime = OptimizerRuntime(optimizers, {name: 10. for name in optimizers})
    batch = MARLBatch(
        observations=torch.randn(6, 2, 3), state=torch.randn(6, 6),
        next_observations=torch.randn(6, 2, 3),
        actions=torch.zeros(6, 2, dtype=torch.long) if discrete else torch.zeros(6, 2, 4),
        rewards=torch.randn(6, 2),
        terminated=torch.zeros(6, 2, dtype=torch.bool),
        truncated=torch.ones(6, 2, dtype=torch.bool),
        extras={"old_log_prob": torch.zeros(6, 2), "old_values": torch.zeros(6, 2),
                "advantages": torch.ones(6, 2), "returns": torch.randn(6, 2)},
    )
    assert isinstance(algorithm, (MAAC, MADDPG, MAPPO, MASAC))
    algorithm.compute_loss_bundle(batch)
    assert algorithm.target_scale.count == 0
    if isinstance(algorithm, MAPPO):
        algorithm.update(PreparedRollout(batch), runtime)
    else:
        algorithm.update(batch, runtime)
    assert algorithm.target_scale.count == 6  # 不是 twin critic 数或 PPO epoch 倍数。
    before = {key: value.clone() for key, value in algorithm.target_scale.state_dict().items()}
    algorithm.compute_loss_bundle(batch)
    for key, value in before.items():
        torch.testing.assert_close(algorithm.target_scale.state_dict()[key], value)
