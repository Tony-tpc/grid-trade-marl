"""循环状态、序列 PPO 与恢复的固定输入回归，不以长训练代替公式测试。"""

from copy import deepcopy
from dataclasses import dataclass, replace

import pytest
import torch

from marl.algorithms import MAACConfig, MAPPOConfig
from marl.config import algorithm_config_from_dict, config_to_dict
from marl.core.recurrent import map_state
from marl.envs import ActionKind, EnvironmentSpec, RewardStructure
from marl.experiment import build_experiment
from marl.models import GRUBackboneConfig, LSTMBackboneConfig, MLPBackboneConfig
from marl.modules.critic import CentralizedValueConfig
from marl.modules.policy import IndependentDiscreteConfig
from marl.training import PPOUpdateConfig, PreparedRollout
from marl.training.optimization import OptimizerRuntime

KINDS = ("mlp", "gru", "lstm")


def backbone(kind):
    if kind == "mlp":
        return MLPBackboneConfig(hidden_dims=(8, 8), output_dim=8)
    return GRUBackboneConfig(hidden_dim=8) if kind == "gru" else LSTMBackboneConfig(hidden_dim=8)


def config(actor, critic):
    return MAPPOConfig(
        policy=IndependentDiscreteConfig(backbone=backbone(actor)),
        critic=CentralizedValueConfig(backbone=backbone(critic)),
        update=PPOUpdateConfig(
            epochs=2, mini_batch_size=6, sequence_length=None if actor == critic == "mlp" else 3
        ),
    )


def spec():
    return EnvironmentSpec(2, 3, 2, 6, ActionKind.DISCRETE, 5, RewardStructure.INDIVIDUAL)


def runtime(algorithm, device="cpu"):
    return OptimizerRuntime(
        {
            "actor": torch.optim.Adam(algorithm.policy.parameters(), lr=3e-4),
            "critic": torch.optim.Adam(algorithm.critic.parameters(), lr=3e-4),
        },
        {"actor": 10.0, "critic": 10.0},
        generator=torch.Generator(device=device).manual_seed(13),
    )


def rollout(algorithm, device="cpu"):
    from marl.returns import GAEConfig
    from marl.training import RolloutBuffer

    buffer = RolloutBuffer(spec(), horizon=5, num_envs=2)
    policy_state, value_state = None, None
    for time in range(5):
        observations = torch.randn(2, 2, 3, device=device)
        state = observations.flatten(-2)
        mask = torch.ones(2, 2, 2, dtype=torch.bool, device=device)
        with torch.inference_mode():
            output = algorithm.sample(
                observations,
                state,
                action_mask=mask,
                policy_state=policy_state,
                value_state=value_state,
            )
        buffer.add(
            observations=observations.cpu(),
            states=state.cpu(),
            actions=output.actions,
            rewards=torch.randn(2, 2),
            old_log_prob=output.log_prob,
            old_values=output.values,
            terminated=torch.zeros(2, 2, dtype=torch.bool),
            truncated=torch.full((2, 2), time == 4, dtype=torch.bool),
            action_masks=mask.cpu(),
            policy_state=policy_state
            if policy_state is not None
            else map_state(output.policy_state, torch.zeros_like),
            value_state=value_state
            if value_state is not None
            else map_state(output.value_state, torch.zeros_like),
        )
        policy_state, value_state = output.policy_state, output.value_state
    with torch.inference_mode():
        next_value = algorithm.values(torch.zeros(2, 2, 3, device=device), value_state=value_state)
    return buffer.finish(next_value, GAEConfig().build()).to(device)


@pytest.mark.parametrize("kind", ("gru", "lstm"))
@pytest.mark.parametrize("layers", (1, 2))
def test_native_and_streaming_equivalence(kind, layers):
    torch.manual_seed(2)
    model = replace(backbone(kind), num_layers=layers).build(3).double()
    x = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
    initial = model.initial_state(2)
    assert all(t.dtype == x.dtype for t in (initial if isinstance(initial, tuple) else (initial,)))
    actual = model(x, hidden_state=initial)
    native = model.gru if kind == "gru" else model.lstm
    expected, expected_state = native(x, initial)
    torch.testing.assert_close(actual.features, expected)
    streamed, state = [], initial
    for time in range(5):
        step = model(x[:, time : time + 1], hidden_state=state)
        streamed.append(step.features)
        state = step.hidden_state
    torch.testing.assert_close(torch.cat(streamed, 1), expected)
    left = state if isinstance(state, tuple) else (state,)
    right = expected_state if isinstance(expected_state, tuple) else (expected_state,)
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b)
    # 最后输出依赖第一步，证明循环图未在序列内部被 detach。
    actual.features[:, -1].sum().backward()
    assert x.grad[:, 0].abs().sum() > 0
    with pytest.raises(ValueError, match="形状"):
        model(x, hidden_state=map_state(initial, lambda t: t[:, :1]))
    with pytest.raises(ValueError, match="dtype"):
        model(x, hidden_state=map_state(initial, lambda t: t.float()))


@pytest.mark.parametrize("actor", KINDS)
@pytest.mark.parametrize("critic", KINDS)
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_nine_combinations_old_prob_padding_update_and_checkpoint(actor, critic, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(19)
    algorithm = config(actor, critic).build(spec()).to(device)
    prepared = rollout(algorithm, device)
    if algorithm.is_recurrent:
        check = PreparedRollout(
            prepared.batch,
            policy_history=prepared.policy_history,
            value_history=prepared.value_history,
        )
        batches = list(check.minibatches(epochs=1, mini_batch_size=6, sequence_length=3))
        assert sum(b.valid_sample_count for b in batches) == 10
        assert sum(b.sequence_mask.numel() for b in batches) == 12
        for batch in batches:
            output = algorithm.policy.evaluate(
                batch.observations,
                batch.actions,
                action_mask=batch.action_mask,
                hidden_state=batch.policy_state,
            )
            torch.testing.assert_close(
                batch.valid(output.log_prob),
                batch.valid(batch.extras["old_log_prob"]),
                atol=1e-6,
                rtol=1e-5,
            )
            values = algorithm.values(
                batch.observations, batch.state, value_state=batch.value_state
            )
            torch.testing.assert_close(
                batch.valid(values), batch.valid(batch.extras["old_values"]), atol=1e-6, rtol=1e-5
            )
            before = algorithm.compute_loss_bundle(batch)
            changed = deepcopy(batch)
            invalid = ~changed.sequence_mask
            changed.observations[invalid] = 71
            changed.state[invalid] = -42
            for key in changed.extras:
                changed.extras[key][invalid] = 100
            after = algorithm.compute_loss_bundle(changed)
            torch.testing.assert_close(before.total, after.total, atol=1e-6, rtol=1e-5)
            gradients = torch.autograd.grad(
                before.total, tuple(algorithm.parameters()), allow_unused=True
            )
            other = torch.autograd.grad(
                after.total, tuple(algorithm.parameters()), allow_unused=True
            )
            for a, b in zip(gradients, other, strict=True):
                if a is not None:
                    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    optimization = runtime(algorithm, device)
    initial = deepcopy(algorithm.state_dict())
    metrics = algorithm.update(prepared, optimization)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert any(not torch.equal(v, initial[k]) for k, v in algorithm.state_dict().items())
    assert optimization.optimizer_step_count > 0
    snapshot = deepcopy(algorithm.state_dict())
    with pytest.raises(RuntimeError, match="消费"):
        algorithm.update(prepared, optimization)
    for key, value in snapshot.items():
        torch.testing.assert_close(value, algorithm.state_dict()[key], rtol=0, atol=0)
    restored = config(actor, critic).build(spec()).to(device)
    restored.load_state_dict(snapshot)
    other_runtime = runtime(restored, device)
    other_runtime.load_state_dict(optimization.state_dict())
    next_rollout = rollout(algorithm, device)
    copied = deepcopy(next_rollout)
    algorithm.update(next_rollout, optimization)
    restored.update(copied, other_runtime)
    for key, value in algorithm.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=1e-6, rtol=1e-5)


def test_config_roundtrip_and_invalid_combinations():
    for actor in KINDS:
        for critic in KINDS:
            original = config(actor, critic)
            assert algorithm_config_from_dict(config_to_dict(original)) == original
    with pytest.raises(ValueError, match="schema_version"):
        algorithm_config_from_dict({"schema_version": 1, "algorithm": "mappo"})
    with pytest.raises(ValueError, match="backbone"):
        algorithm_config_from_dict(
            {"schema_version": 2, "algorithm": "mappo", "policy": {"hidden_dim": 64}}
        )
    with pytest.raises(ValueError, match="sequence"):
        replace(config("mlp", "mlp"), update=PPOUpdateConfig(sequence_length=3)).build(spec())
    with pytest.raises(ValueError, match="sequence"):
        replace(config("lstm", "mlp"), update=PPOUpdateConfig(sequence_length=6)).build(spec())
    with pytest.raises(ValueError, match="replay"):
        MAACConfig(policy=IndependentDiscreteConfig(backbone=backbone("lstm"))).build(spec())


def test_recurrent_act_cannot_silently_discard_memory():
    algorithm = config("lstm", "mlp").build(spec())
    with pytest.raises(RuntimeError, match="policy_state"):
        algorithm.act(torch.randn(2, 2, 3))


def test_policy_and_critic_gradients_are_isolated():
    algorithm = config("lstm", "gru").build(spec())
    data = rollout(algorithm)
    batch = next(data.minibatches(epochs=1, mini_batch_size=6, sequence_length=3))
    algorithm.compute_policy_loss_bundle(batch).total.backward()
    assert all(p.grad is None for p in algorithm.critic.parameters())
    assert any(p.grad is not None for p in algorithm.policy.parameters())
    algorithm.zero_grad(set_to_none=True)
    algorithm.compute_value_loss_bundle(batch).total.backward()
    assert all(p.grad is None for p in algorithm.policy.parameters())


def test_stateless_value_extension_remains_supported():
    from marl.envs import MemoryCueAdapter

    env = MemoryCueAdapter(horizon=5)

    @dataclass(frozen=True)
    class StatelessValueConfig:
        kind: str = "stateless"

        def build(self, spec):
            return torch.nn.Sequential(torch.nn.Linear(spec.state_dim, spec.num_agents))

    original = replace(config("lstm", "mlp"), critic=StatelessValueConfig())
    experiment = build_experiment(env, original)
    metrics = experiment.trainer.train_rollout([7])
    assert metrics["loss"] == pytest.approx(metrics["loss"])
    experiment.trainer.load_state_dict(experiment.trainer.state_dict())


@pytest.mark.parametrize("actor", KINDS)
@pytest.mark.parametrize("critic", KINDS)
def test_collector_checkpoint_restores_next_rollout_and_reset(actor, critic):
    from marl.envs import MemoryCueAdapter

    experiment = build_experiment(MemoryCueAdapter(horizon=5), config(actor, critic), seed=7)
    experiment.trainer.train_rollout([13])
    snapshot = experiment.trainer.state_dict()
    reference = experiment.trainer.train_rollout([17])
    weights = deepcopy(experiment.algorithm.state_dict())
    restored = build_experiment(MemoryCueAdapter(horizon=5), config(actor, critic), seed=99)
    restored.trainer.load_state_dict(snapshot)
    actual = restored.trainer.train_rollout([17])
    assert actual == pytest.approx(reference, rel=1e-5, abs=1e-6)
    for key, value in weights.items():
        torch.testing.assert_close(value, restored.algorithm.state_dict()[key], rtol=0, atol=0)
    prior = deepcopy(restored.algorithm.state_dict())
    with pytest.raises(ValueError, match="不兼容"):
        restored.trainer.load_state_dict({**snapshot, "schema_version": 3})
    for key, value in prior.items():
        torch.testing.assert_close(value, restored.algorithm.state_dict()[key], rtol=0, atol=0)


def test_memory_task_does_not_leak_cue_after_first_frame():
    import numpy as np

    from marl.envs import MemoryCueAdapter

    envs = [MemoryCueAdapter() for _ in range(4)]
    first = [env.reset(seed=i) for i, env in enumerate(envs)]
    assert len({tuple(s.observations[:, 0]) for s in first}) == 4
    for _ in range(9):
        states = [env.step(np.zeros(2, dtype=np.int64)) for env in envs]
        for row in states[1:]:
            np.testing.assert_array_equal(row.observations, states[0].observations)
            np.testing.assert_array_equal(row.state, states[0].state)
    results = [env.step(np.zeros(2, dtype=np.int64)) for env in envs]
    assert all(row.terminated and not row.truncated for row in results)
    assert np.mean([row.rewards for row in results]) == 0


@pytest.mark.parametrize("end", ("terminated", "truncated"))
@pytest.mark.parametrize("critic", ("gru", "lstm"))
def test_collector_bootstraps_final_observation_before_reset(end, critic, monkeypatch):
    from marl.envs import MemoryCueAdapter

    class EndingMemory(MemoryCueAdapter):
        def step(self, actions):
            following = super().step(actions)
            if following.done:
                following.terminated = end == "terminated"
                following.truncated = end == "truncated"
            return following

    experiment = build_experiment(EndingMemory(horizon=5), config("mlp", critic))
    captured = []

    def bootstrap(observations, state=None, *, value_state=None):
        assert value_state is not None
        values = value_state if isinstance(value_state, tuple) else (value_state,)
        assert all(t.abs().sum() > 0 for t in values)
        captured.append(observations.clone())
        return torch.full((1, 2), 100.0)

    monkeypatch.setattr(experiment.algorithm, "values", bootstrap)
    prepared = experiment.trainer.collect([7])
    assert len(captured) == 1 and torch.equal(captured[0], torch.zeros_like(captured[0]))
    last = prepared.batch.extras["returns"][:, -1]
    reward = prepared.batch.rewards[:, -1]
    torch.testing.assert_close(last, reward + (99.0 if end == "truncated" else 0.0),
                               atol=1e-5, rtol=1e-5)


def test_bad_network_checkpoint_rejected_before_parameters_or_rng_change():
    from marl.envs import MemoryCueAdapter
    experiment = build_experiment(MemoryCueAdapter(horizon=5), config("lstm", "gru"))
    snapshot = experiment.trainer.state_dict()
    before = deepcopy(experiment.algorithm.state_dict())
    key = next(iter(snapshot["algorithm"]))
    snapshot["algorithm"][key] = torch.zeros(100)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="结构"):
        experiment.trainer.load_state_dict(snapshot)
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in before.items():
        assert torch.equal(value, experiment.algorithm.state_dict()[key])


def test_new_benchmark_validator_rejects_missing_runs_and_bad_histories(tmp_path):
    from benchmarks.benchmark_recurrent import run, validate_report
    from benchmarks.plot_recurrent import plot
    report = run(tmp_path / "run", seeds=(7,), tasks=("memory",),
                 memory_steps=200, mpe_steps=100)
    assert report["acceptance"]["outputs_complete"]
    paths = plot(report, tmp_path / "plots")
    assert len(paths) == 7 and all(p.stat().st_size > 1000 for p in paths)
    broken = deepcopy(report)
    broken["runs"].pop()
    with pytest.raises(ValueError, match="矩阵"):
        validate_report(broken)
    broken = deepcopy(report)
    broken["runs"][0]["evaluation_history"].pop(0)
    with pytest.raises(ValueError, match="评估"):
        validate_report(broken)
    broken = deepcopy(report)
    broken["runs"][0]["training_history"][0]["metrics"].pop("actor_gradient_clip_rate")
    with pytest.raises(ValueError, match="缺少训练诊断"):
        validate_report(broken)
    broken = deepcopy(report)
    broken["runs"][0]["training_history"][0]["metrics"]["actor_gradient_nonfinite"] = 1.0
    with pytest.raises(ValueError, match="非有限"):
        validate_report(broken)


def test_corrupt_optimizer_checkpoint_rolls_back_model_runtime_and_rng():
    from marl.envs import MemoryCueAdapter
    experiment = build_experiment(MemoryCueAdapter(horizon=5), config("lstm", "gru"), seed=7)
    experiment.trainer.train_rollout([7])
    old = experiment.trainer.state_dict()
    experiment.trainer.train_rollout([17])
    previous = experiment.trainer.state_dict()
    old["optimization"]["optimizers"].pop("critic")
    with pytest.raises(ValueError, match="optimizer"):
        experiment.trainer.load_state_dict(old)
    current = experiment.trainer.state_dict()
    for key, tensor in previous["algorithm"].items():
        assert torch.equal(tensor, current["algorithm"][key])
    assert torch.equal(previous["torch_rng"], current["torch_rng"])
    assert current["optimization"]["update_count"] == previous["optimization"]["update_count"]
    # 用合法的回滚快照继续下一次更新，而不只是比较计数器。
    reference = experiment.trainer.train_rollout([29])
    restored = build_experiment(MemoryCueAdapter(horizon=5), config("lstm", "gru"), seed=9)
    restored.trainer.load_state_dict(previous)
    actual = restored.trainer.train_rollout([29])
    assert actual == pytest.approx(reference, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("actor", KINDS)
@pytest.mark.parametrize("critic", KINDS)
def test_cuda_collector_and_trainer_checkpoint_roundtrip(actor, critic):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from marl.envs import MemoryCueAdapter
    experiment = build_experiment(MemoryCueAdapter(horizon=5), config(actor, critic),
                                  seed=7, device="cuda")
    experiment.trainer.train_rollout([7])
    snapshot = experiment.trainer.state_dict()
    reference = experiment.trainer.train_rollout([17])
    restored = build_experiment(MemoryCueAdapter(horizon=5), config(actor, critic),
                               seed=9, device="cuda")
    restored.trainer.load_state_dict(snapshot)
    actual = restored.trainer.train_rollout([17])
    assert actual == pytest.approx(reference, rel=1e-5, abs=1e-6)
    for key, value in experiment.algorithm.state_dict().items():
        torch.testing.assert_close(value, restored.algorithm.state_dict()[key],
                                   atol=1e-6, rtol=1e-5)


def test_bad_recurrent_layout_rejected_before_target_statistics_change():
    algorithm = config("lstm", "gru").build(spec())
    prepared = rollout(algorithm)
    with pytest.raises(ValueError, match="历史"):
        PreparedRollout(prepared.batch,
                        policy_history=map_state(prepared.policy_history, lambda t: t[:-1]),
                        value_history=prepared.value_history)
    before = deepcopy(algorithm.state_dict())
    prepared.batch.extras.pop("old_log_prob")
    with pytest.raises(ValueError, match="extras"):
        algorithm.update(prepared, runtime(algorithm))
    assert not prepared.consumed
    for name, tensor in before.items():
        assert torch.equal(tensor, algorithm.state_dict()[name])


@pytest.mark.parametrize("left_kind", KINDS)
@pytest.mark.parametrize("right_kind", KINDS)
def test_cross_play_preserves_both_sources_states(left_kind, right_kind):
    import numpy as np

    from benchmarks.benchmark_noncooperative import _cross_play_actions, _policy_actions

    left = config(left_kind, "mlp").build(spec())
    right = config(right_kind, "mlp").build(spec())
    actual_left = actual_right = expected_left = expected_right = None
    for _ in range(4):
        observations = torch.randn(2, 3).numpy()
        mask = np.ones((2, 2), dtype=bool)
        left_actions, expected_left = _policy_actions(
            left, observations, mask, device=torch.device("cpu"),
            deterministic=True, policy_state=expected_left,
        )
        right_actions, expected_right = _policy_actions(
            right, observations, mask, device=torch.device("cpu"),
            deterministic=True, policy_state=expected_right,
        )
        joint, actual_left, actual_right = _cross_play_actions(
            left, right, observations, mask, device=torch.device("cpu"),
            adversary_state=actual_left, good_state=actual_right,
        )
        np.testing.assert_array_equal(joint, np.r_[left_actions[:1], right_actions[1:]])
        for actual, expected in ((actual_left, expected_left), (actual_right, expected_right)):
            if expected is None:
                assert actual is None
            else:
                a = actual if isinstance(actual, tuple) else (actual,)
                e = expected if isinstance(expected, tuple) else (expected,)
                for x, y in zip(a, e, strict=True):
                    torch.testing.assert_close(x, y)


def test_evaluation_resets_episode_states_and_restores_module_modes(monkeypatch):
    pytest.importorskip("mpe2")
    from benchmarks import benchmark_noncooperative as benchmark
    from marl.envs import MPE2SimpleAdversaryConfig, build_mpe2_simple_adversary

    settings = benchmark.BenchmarkSettings(
        horizon=5, eval_episodes=2, actor_backbone="lstm", sequence_length=3,
    )
    adapter = build_mpe2_simple_adversary(
        MPE2SimpleAdversaryConfig(horizon=5), action_kind=ActionKind.DISCRETE,
    )
    try:
        algorithm = benchmark._algorithm_config("mappo", settings).build(adapter.spec)
    finally:
        adapter.close()
    algorithm.train()
    algorithm.policy.actors[0].eval()
    modes = [m.training for m in algorithm.modules()]
    seen = []
    original = benchmark._policy_actions

    def checked(*args, **kwargs):
        seen.append(kwargs.get("policy_state"))
        assert not any(module.training for module in algorithm.modules())
        return original(*args, **kwargs)

    monkeypatch.setattr(benchmark, "_policy_actions", checked)
    rng = torch.get_rng_state().clone()
    benchmark._evaluate(
        algorithm, ActionKind.DISCRETE, settings, device=torch.device("cpu"), seed=42,
    )
    assert len(seen) == 10 and seen[0] is None and seen[5] is None
    assert all(isinstance(seen[i], tuple) for i in (1, 2, 3, 4, 6, 7, 8, 9))
    assert [m.training for m in algorithm.modules()] == modes
    assert torch.equal(rng, torch.get_rng_state())


def test_actor_and_critic_can_have_different_layers_and_state_widths():
    cfg = MAPPOConfig(
        policy=IndependentDiscreteConfig(backbone=LSTMBackboneConfig(hidden_dim=7, num_layers=2)),
        critic=CentralizedValueConfig(backbone=GRUBackboneConfig(hidden_dim=5, num_layers=3)),
        update=PPOUpdateConfig(epochs=2, mini_batch_size=6, sequence_length=3),
    )
    algorithm = cfg.build(spec())
    experience = rollout(algorithm)
    assert isinstance(experience.policy_history, tuple)
    assert all(t.shape == (5, 2, 2, 2, 7) for t in experience.policy_history)
    assert experience.value_history.shape == (5, 3, 2, 5)
    metrics = algorithm.update(experience, runtime(algorithm))
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_cuda_pinning_preserves_state_history_and_consumption():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from marl.envs import MemoryCueAdapter
    experiment = build_experiment(
        MemoryCueAdapter(horizon=5), config("lstm", "gru"), device="cuda",
    )
    prepared = experiment.trainer.collect([7])
    assert prepared.batch.observations.is_pinned()
    assert all(value.is_pinned() for value in prepared.batch.extras.values())
    assert isinstance(prepared.policy_history, tuple)
    assert all(value.is_cuda for value in prepared.policy_history)
    assert prepared.value_history.is_cuda
    for length in (0, -1, True):
        with pytest.raises(ValueError, match="sequence_length"):
            prepared.validate_minibatches(6, length)
    moved = prepared.to("cuda", non_blocking=True)
    assert prepared.consumed and not moved.consumed
    assert moved.policy_history[0].data_ptr() == prepared.policy_history[0].data_ptr()
    assert moved.value_history.data_ptr() == prepared.value_history.data_ptr()
    with pytest.raises(RuntimeError, match="消费"):
        prepared.to("cuda")
    experiment.algorithm.update(moved, experiment.trainer.optimization)
    assert moved.consumed
