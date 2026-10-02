"""接口边界及全库审计发现的生命周期、日志和 PyTorch 扩展回归。"""
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import Tensor, nn

from marl.algorithms import MAAC, MADDPG, MAPPO, MASAC, QMIX, MAPPOConfig, QMIXConfig
from marl.algorithms.base import BaseMARLAlgorithm
from marl.algorithms.sn_mappo import SNMAPPOConfig
from marl.config import algorithm_config_from_dict, config_to_dict
from marl.core import MARLBatch
from marl.envs import ActionKind, EnvironmentSpec, RewardStructure
from marl.models import MLPBackbone
from marl.modules.action_head import DiscreteActionHead
from marl.modules.actor import Actor
from marl.modules.critic import CentralizedCritic, ValueCritic
from marl.modules.mixer import QMixer
from marl.target_updates import SoftTargetUpdate, frozen_target
from marl.training import (
    ActorCriticUpdateConfig,
    PPOUpdateConfig,
    PreparedRollout,
    TemperatureActorCriticUpdateConfig,
)
from marl.training.optimization import OptimizerRuntime, metrics_to_float


def _spec(shared: bool = False) -> EnvironmentSpec:
    return EnvironmentSpec(2, 3, 4, 6, ActionKind.DISCRETE, 5,
                           RewardStructure.SHARED if shared else RewardStructure.INDIVIDUAL)


def _runtime(algorithm: MAPPO) -> OptimizerRuntime:
    return OptimizerRuntime(
        {"actor": torch.optim.Adam(algorithm.policy.parameters(), lr=.001),
         "critic": torch.optim.Adam(algorithm.critic.parameters(), lr=.001)},
        {"actor": .5, "critic": .5}, generator=torch.Generator().manual_seed(27),
    )


def test_single_algorithm_base_and_optimizer_lifecycle() -> None:
    for algorithm in (MAAC, MADDPG, MAPPO, MASAC, QMIX):
        assert algorithm.__bases__ == (BaseMARLAlgorithm,)
    root = Path(__file__).resolve().parents[1] / "marl"
    backward_sites = []
    optimizer_sites = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                assert not node.name.endswith("UpdatePlan"), path
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "backward":
                    backward_sites.append(path.relative_to(root).as_posix())
                if (node.func.attr == "step" and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "optimizer"):
                    optimizer_sites.append(path.relative_to(root).as_posix())
            # 通用层只依赖能力协议，不认识具体算法类。
            if path.parts[-2] in ("models", "modules", "training") and isinstance(
                node, ast.ImportFrom
            ):
                assert not (node.module or "").startswith("marl.algorithms"), path
    assert backward_sites == ["algorithms/base.py"]
    assert optimizer_sites == ["algorithms/base.py"]


@pytest.mark.parametrize("name,kind", [
    ("maac", ActorCriticUpdateConfig), ("maddpg", ActorCriticUpdateConfig),
    ("masac", TemperatureActorCriticUpdateConfig),
])
def test_shared_update_config_preserves_yaml_fields(name: str, kind: type) -> None:
    config = algorithm_config_from_dict({"schema_version": 2, "algorithm": name})
    assert not isinstance(config, SNMAPPOConfig)
    assert type(config.update) is kind
    data = config_to_dict(config)
    assert config_to_dict(algorithm_config_from_dict(data)) == data


def test_rollout_device_transfer_moves_single_use_ownership() -> None:
    original = PreparedRollout(MARLBatch(torch.zeros(5, 2, 3)))
    moved = original.to("cpu")
    assert original.consumed and not moved.consumed
    with pytest.raises(RuntimeError, match="消费"):
        original.to("cpu")
    with pytest.raises(RuntimeError, match="消费"):
        list(original.minibatches(epochs=1, mini_batch_size=3))
    assert len(list(moved.minibatches(epochs=2, mini_batch_size=3))) == 4
    with pytest.raises(RuntimeError, match="消费"):
        list(moved.minibatches(epochs=1, mini_batch_size=3))


def test_failed_rollout_transfer_retains_ownership() -> None:
    original = PreparedRollout(MARLBatch(torch.zeros(5, 2, 3)))
    with patch.object(MARLBatch, "to", side_effect=RuntimeError("传输失败")):
        with pytest.raises(RuntimeError, match="传输失败"):
            original.to("cpu")
    assert not original.consumed


def test_reused_rollout_does_not_mutate_target_statistics() -> None:
    config = MAPPOConfig()
    algorithm = replace(config, loss=replace(config.loss, normalize_targets=True)).build(_spec())
    rollout = PreparedRollout(MARLBatch(
        torch.zeros(5, 2, 3), extras={"returns": torch.ones(5, 2)}
    ))
    list(rollout.minibatches(epochs=1, mini_batch_size=5))
    before = deepcopy(algorithm.state_dict())
    with pytest.raises(RuntimeError, match="消费"):
        algorithm.update(rollout, _runtime(algorithm))
    for name, value in algorithm.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_mappo_sample_weighted_logs_preserve_updates_and_optimizer_state() -> None:
    torch.manual_seed(12)
    config = replace(MAPPOConfig(), update=PPOUpdateConfig(epochs=2, mini_batch_size=3))
    actual = config.build(_spec())
    reference = deepcopy(actual)
    obs, state = torch.randn(5, 2, 3), torch.randn(5, 6)
    with torch.no_grad():
        sample = actual.sample(obs, state)
    assert sample.log_prob is not None and sample.values is not None
    batch = MARLBatch(obs, actions=sample.actions, state=state, extras={
        "old_log_prob": sample.log_prob, "old_values": sample.values,
        "advantages": torch.randn(5, 2), "returns": torch.randn(5, 2),
    })
    runtime, reference_runtime = _runtime(actual), _runtime(reference)
    metrics = actual.update(PreparedRollout(batch), runtime)
    after_rng = torch.get_rng_state().clone()
    weighted_loss, equal_loss, actor_norm, critic_norm = 0., 0., 0., 0.
    # 手写参考仅在测试中保留：训练顺序和旧实现相同，独立核对日志归约权重。
    for mini in PreparedRollout(batch).minibatches(
        epochs=2, mini_batch_size=3, generator=reference_runtime.generator
    ):
        policy = reference.compute_policy_loss_bundle(mini)
        actor_norm += float(reference.optimize(reference_runtime.optimizer("actor"), policy, .5))
        value = reference.compute_value_loss_bundle(mini)
        critic_norm += float(reference.optimize(reference_runtime.optimizer("critic"), value, .5))
        loss = float(policy.total.detach() + value.total.detach())
        weighted_loss += loss * mini.observations.shape[0]
        equal_loss += loss
    assert metrics["loss"] == pytest.approx(weighted_loss / 10, abs=1e-6)
    assert abs(weighted_loss / 10 - equal_loss / 4) > 1e-5
    assert metrics["actor_gradient_norm"] == pytest.approx(actor_norm / 4)
    assert metrics["critic_gradient_norm"] == pytest.approx(critic_norm / 4)
    assert metrics["actor_step_count"] == metrics["critic_step_count"] == 4
    assert runtime.update_count == 1 and runtime.optimizer_step_count == 8
    assert torch.equal(torch.get_rng_state(), after_rng)
    for name, tensor in actual.state_dict().items():
        torch.testing.assert_close(tensor, reference.state_dict()[name], rtol=0, atol=0)
    for name in ("actor", "critic"):
        expected = reference_runtime.optimizer(name).state_dict()["state"]
        for parameter, fields in runtime.optimizer(name).state_dict()["state"].items():
            for field, tensor in fields.items():
                torch.testing.assert_close(tensor, expected[parameter][field], rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_metric_weights_do_not_scale_counts_peaks_or_retain_graph(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("需要 CUDA")
    sums: dict[str, Tensor] = {}
    for size, value in ((3, 2.), (1, 6.)):
        values = {f"loss_{i}": torch.tensor(value, device=device, requires_grad=True)
                  for i in range(10)}
        values.update({"actor_clip_count": torch.tensor(1., device=device),
                       "actor_step_count": torch.tensor(1., device=device),
                       "q_abs_max": torch.tensor(value, device=device)})
        OptimizerRuntime.accumulate_metrics(sums, values, weight=size)
    means = metrics_to_float(OptimizerRuntime.mean_metrics(sums, 4))
    assert means["loss_0"] == 3.
    assert means["actor_clip_count"] == 2.
    assert means["actor_gradient_clip_rate"] == 1.
    assert means["q_abs_max"] == 6.
    assert not any(value.requires_grad for value in sums.values())
    assert metrics_to_float({}) == {}


def test_optimizer_runtime_rejects_overlapping_and_empty_parameter_groups() -> None:
    parameter = nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="重复归属"):
        OptimizerRuntime({"actor": torch.optim.Adam([parameter]),
                          "critic": torch.optim.Adam([parameter])}, {"actor": 1., "critic": 1.})
    with pytest.raises(ValueError, match="没有参数"):
        OptimizerRuntime({"value": torch.optim.Adam([{"params": []}])}, {"value": 1.})


@pytest.mark.parametrize("field,shape", [
    ("rewards", (5, 1)), ("terminated", (5,)), ("truncated", (5, 1)),
    ("actions", (5, 3)), ("next_observations", (5, 2, 4)),
    ("state", (1, 6)), ("next_state", (1, 6)), ("next_action_mask", (5, 1, 4)),
])
def test_batch_rejects_silent_broadcasts(field: str, shape: tuple[int, ...]) -> None:
    batch = MARLBatch(torch.zeros(5, 2, 3))
    setattr(batch, field, torch.zeros(shape))
    with pytest.raises(ValueError, match=field):
        batch.validate(2, 3)


def test_qmixer_rejects_different_leading_shapes_with_same_flat_size() -> None:
    with pytest.raises(ValueError, match="批次维"):
        QMixer(2, 6)(torch.zeros(2, 3, 2), torch.zeros(3, 2, 6))


def test_actor_and_critic_preserve_standard_module_hooks() -> None:
    actor = Actor(MLPBackbone(3, (8,), 8), DiscreteActionHead(8, 4))
    obs = torch.randn(5, 3)
    expected = actor(obs, deterministic=True)
    calls: list[nn.Module] = []
    handles = [module.register_forward_pre_hook(lambda module, args: calls.append(module))
               for module in (actor.backbone, actor.action_head)]
    try:
        actual = actor(obs, deterministic=True)
        actor.evaluate_actions(obs, actual.actions)
        actor.discrete_logits(obs)
    finally:
        for handle in handles:
            handle.remove()
    assert calls.count(actor.backbone) == 3 and calls.count(actor.action_head) == 1
    torch.testing.assert_close(actual.actions, expected.actions, rtol=0, atol=0)
    for critic in (ValueCritic(actor.backbone), CentralizedCritic(actor.backbone)):
        calls.clear()
        handle = critic.backbone.register_forward_pre_hook(
            lambda module, args: calls.append(module)
        )
        try:
            critic(obs)
        finally:
            handle.remove()
        assert calls == [critic.backbone]


def test_soft_target_updates_parameters_and_copies_float_and_integer_buffers() -> None:
    online = nn.BatchNorm1d(3)
    target = frozen_target(online)
    assert online.running_mean is not None and online.num_batches_tracked is not None
    assert target.num_batches_tracked is not None
    with torch.no_grad():
        online.weight.fill_(3.)
        online.running_mean.fill_(2.)
        online.num_batches_tracked.fill_(7)
    SoftTargetUpdate(.25).step([(target, online)])
    torch.testing.assert_close(target.weight, torch.full((3,), 1.5))
    torch.testing.assert_close(target.running_mean, online.running_mean)
    assert target.num_batches_tracked.item() == 7
    assert all(parameter.grad is None for parameter in target.parameters())


def test_qmix_records_value_optimizer_events() -> None:
    algorithm = QMIXConfig().build(_spec(shared=True))
    runtime = OptimizerRuntime(
        {"value": torch.optim.Adam(p for p in algorithm.parameters() if p.requires_grad)},
        {"value": .01},
    )
    batch = MARLBatch(
        torch.randn(5, 2, 3), actions=torch.zeros(5, 2, dtype=torch.long),
        rewards=torch.ones(5, 2), next_observations=torch.randn(5, 2, 3),
        state=torch.randn(5, 6), next_state=torch.randn(5, 6),
    )
    metrics = algorithm.update(batch, runtime)
    assert metrics["value_step_count"] == 1
    assert metrics["value_gradient_clip_rate"] in (0., 1.)
    assert metrics["value_gradient_abs_max"] == metrics["gradient_norm"]
