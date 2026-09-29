"""统一网络的时序、图、梯度、真实采样和恢复契约。"""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from marl.algorithms import MAACConfig, MADDPGConfig, MAPPOConfig, MASACConfig, QMIXConfig
from marl.config import (
    AlgorithmConfig,
    algorithm_config_from_dict,
    config_to_dict,
    load_algorithm_config,
)
from marl.core import HistoryLayout, StateLayout
from marl.envs import (
    ActionKind,
    EnvironmentAdapter,
    EnvironmentSpec,
    EnvironmentStep,
    RewardStructure,
    Transition,
    build_energy_trading_adapter,
    load_environment_config,
    transitions_to_batch,
)
from marl.experiment import build_experiment
from marl.models import (
    GNNBackbone,
    GNNBackboneConfig,
    GRUBackboneConfig,
    HistoryEncoder,
    HistoryEncoderConfig,
    LSTMBackboneConfig,
    MLPBackboneConfig,
    TransformerBackbone,
    TransformerBackboneConfig,
)
from marl.models.encoder import TemporalConfig
from marl.modules.critic import (
    AttentionQConfig,
    CentralizedValueConfig,
    IndependentQConfig,
    TwinQConfig,
)
from marl.modules.mixer import QMixerConfig
from marl.modules.policy import (
    IndependentDeterministicConfig,
    IndependentDiscreteConfig,
    IndependentGaussianConfig,
    SharedDiscreteQConfig,
)
from marl.training import OffPolicyTrainer, OnPolicyTrainer
from marl.training.off_policy import ReplayConfig
from marl.training.on_policy import PPOUpdateConfig

TEMPORAL = (
    GRUBackboneConfig(hidden_dim=8),
    LSTMBackboneConfig(hidden_dim=8),
    TransformerBackboneConfig(model_dim=8, num_heads=2, num_layers=1),
)
LAYOUT = HistoryLayout(((0, 3), (1, 4), (2, 5)), (6,))
SMALL = MLPBackboneConfig(hidden_dims=(), output_dim=8)


class WindowAdapter(EnvironmentAdapter):
    """动作影响个体收益的小环境；显式声明三时刻、两通道和当前特征。"""

    def __init__(self, name: str) -> None:
        continuous = name in ("maddpg", "masac")
        self._spec = EnvironmentSpec(
            2,
            7,
            2 if continuous else 3,
            14,
            ActionKind.CONTINUOUS if continuous else ActionKind.DISCRETE,
            4,
            RewardStructure.SHARED if name == "qmix" else RewardStructure.INDIVIDUAL,
            observation_history=LAYOUT,
            state_layout=StateLayout((tuple(range(7)), tuple(range(7, 14))), history=LAYOUT),
            adjacency=((0.0, 2.0), (1.0, 0.0)),
        )
        self.time = 0

    @property
    def spec(self) -> EnvironmentSpec:
        return self._spec

    def _step(self, rewards: np.ndarray) -> EnvironmentStep:
        observations = np.arange(14, dtype=np.float32).reshape(2, 7) / 10 + self.time / 10
        return self.validate_step(
            EnvironmentStep(
                observations,
                observations.flatten(),
                rewards,
                False,
                self.time == 4,
                np.ones((2, 3), dtype=bool)
                if self.spec.action_kind == ActionKind.DISCRETE
                else None,
            )
        )

    def reset(self, seed: int | None = None) -> EnvironmentStep:
        self.time = 0
        return self._step(np.zeros(2, dtype=np.float32))

    def step(self, actions: np.ndarray) -> EnvironmentStep:
        self.time += 1
        rewards = -np.square(actions.astype(np.float32) - 0.7)
        if rewards.ndim == 2:
            rewards = rewards.sum(-1)
        if self.spec.reward_structure == RewardStructure.SHARED:
            rewards[:] = rewards.sum()
        return self._step(rewards)


def network_config(name: str, temporal: TemporalConfig, graph: bool) -> AlgorithmConfig:
    encoder = HistoryEncoderConfig(temporal=temporal)
    gnn = GNNBackboneConfig(hidden_dim=8, num_layers=1) if graph else None
    discrete = IndependentDiscreteConfig(backbone=SMALL, encoder=encoder)
    replay = ReplayConfig(capacity=8, batch_size=2)
    if name == "mappo":
        return MAPPOConfig(
            policy=discrete,
            critic=CentralizedValueConfig(backbone=SMALL, encoder=encoder, graph=gnn),
            update=PPOUpdateConfig(epochs=1, mini_batch_size=4),
        )
    if name == "maac":
        return MAACConfig(
            policy=discrete,
            critic=AttentionQConfig(
                encoder=encoder, embedding=SMALL, backbone=SMALL, attention_heads=2, graph=gnn
            ),
            replay=replay,
        )
    if name == "maddpg":
        return MADDPGConfig(
            policy=IndependentDeterministicConfig(backbone=SMALL, encoder=encoder),
            critic=IndependentQConfig(backbone=SMALL, encoder=encoder, graph=gnn),
            replay=replay,
        )
    if name == "masac":
        return MASACConfig(
            policy=IndependentGaussianConfig(backbone=SMALL, encoder=encoder),
            critic=TwinQConfig(backbone=SMALL, encoder=encoder, graph=gnn),
            replay=replay,
        )
    return QMIXConfig(
        policy=SharedDiscreteQConfig(backbone=SMALL, encoder=encoder),
        mixer=QMixerConfig(
            encoder=encoder, graph=gnn, mixing_dim=8, hyper_hidden_dims=(8, 4), value_backbone=SMALL
        ),
        replay=replay,
    )


def collect(run, env):
    current = env.reset(3)
    transitions = []
    for _ in range(4):
        with torch.inference_mode():
            actions = run.algorithm.act(
                torch.from_numpy(current.observations).unsqueeze(0),
                action_mask=None
                if current.action_mask is None
                else torch.from_numpy(current.action_mask).unsqueeze(0),
            )[0].numpy()
        following = env.step(actions)
        transitions.append(Transition(current, actions, following))
        current = following
    run.trainer.record_many(transitions)
    return transitions


@pytest.mark.parametrize("temporal", TEMPORAL)
def test_history_layout_last_output_batch_and_replay_independence(temporal):
    model = HistoryEncoderConfig(temporal=temporal).build(7, LAYOUT)
    inputs = torch.arange(42, dtype=torch.float32).reshape(2, 3, 7) / 20
    history = torch.stack((inputs[..., :3], inputs[..., 3:6]), -1)
    expected = torch.cat((model.temporal(history).features[..., -1, :], inputs[..., 6:]), -1)
    actual = model(inputs)
    torch.testing.assert_close(actual, expected)
    individual = torch.stack([model(row) for row in inputs.reshape(-1, 7)]).reshape_as(actual)
    torch.testing.assert_close(actual, individual, atol=1e-6, rtol=1e-5)
    permutation = torch.tensor([3, 0, 5, 2, 1, 4])
    torch.testing.assert_close(
        model(inputs.reshape(-1, 7)[permutation]), actual.reshape(-1, actual.shape[-1])[permutation]
    )
    model(torch.randn_like(inputs))  # 不得污染下次调用的初态。
    torch.testing.assert_close(actual, model(inputs))
    # LayerNorm 各特征求和可恒为零；选择单个隐藏特征检验历史梯度。
    model(inputs.requires_grad_())[..., 0].sum().backward()
    assert inputs.grad is not None and inputs.grad[..., :6].abs().sum() > 0


def test_lstm_last_hidden_is_computed_with_zero_initial_state():
    encoder = HistoryEncoderConfig(temporal=LSTMBackboneConfig(hidden_dim=1)).build(
        3, HistoryLayout(((0,), (1,)), (2,))
    )
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.zero_()
        encoder.temporal.lstm.weight_ih_l0[2, 0] = 1
    x = torch.tensor([0.2, 0.8, 7.0])
    cell = 0.5 * (0.5 * x[0].tanh()) + 0.5 * x[1].tanh()
    expected = torch.stack((0.5 * cell.tanh(), x[2]))
    torch.testing.assert_close(encoder(x), expected)


def test_transformer_positions_causality_and_determinism():
    torch.manual_seed(5)
    model = TransformerBackbone(2, model_dim=8, num_heads=2, num_layers=1)
    x = torch.ones(2, 4, 2)
    original = model(x).features
    assert not torch.allclose(original[:, 0], original[:, 1])
    changed = x.clone()
    changed[:, 2:] += 100
    torch.testing.assert_close(original[:, :2], model(changed).features[:, :2])
    model.eval()
    torch.testing.assert_close(original, model(x).features)
    with pytest.raises(ValueError, match="dropout"):
        TransformerBackbone(2, dropout=0.1)


def test_graph_manual_aggregation_self_loops_and_single_node():
    model = GNNBackbone(1, hidden_dim=1, num_layers=1)
    with torch.no_grad():
        model.input_projection.weight.fill_(1)
        model.input_projection.bias.zero_()
        model.layers[0][0].weight.copy_(torch.tensor([[0.0, 1.0]]))
        model.layers[0][0].bias.zero_()
    x = torch.tensor([[1.0], [4.0]])
    torch.testing.assert_close(
        model(x, adjacency=torch.tensor([[0.0, 2.0], [0.0, 0.0]])).features,
        torch.tensor([[3.0], [4.0]]),
    )
    torch.testing.assert_close(model(x).features, torch.tensor([[2.5], [2.5]]))
    torch.testing.assert_close(
        model(torch.tensor([[5.0]]), adjacency=torch.zeros(1, 1)).features, torch.tensor([[5.0]])
    )
    for adjacency in (
        torch.zeros(2, 2, 2),
        torch.full((2, 2), -1.0),
        torch.full((2, 2), float("nan")),
    ):
        with pytest.raises(ValueError, match="adjacency"):
            model(x, adjacency=adjacency)


@pytest.mark.parametrize("name", ["mappo", "maac", "maddpg", "masac", "qmix"])
@pytest.mark.parametrize("temporal", TEMPORAL)
@pytest.mark.parametrize("graph", [False, True])
def test_all_networks_train_and_restore_next_update(name, temporal, graph, tmp_path):
    config = network_config(name, temporal, graph)
    assert algorithm_config_from_dict(config_to_dict(config)) == config
    env = WindowAdapter(name)
    run = build_experiment(env, config, seed=14)
    initial = {key: p.clone() for key, p in run.algorithm.named_parameters()}
    if isinstance(run.trainer, OnPolicyTrainer):
        metrics = run.trainer.train_rollout([2])
    else:
        collect(run, env)
        metrics = run.trainer.update()
    assert all(np.isfinite(value) for value in metrics.values())
    assert run.trainer.optimization.update_count == 1
    online_history = [
        (name, module)
        for name, module in run.algorithm.named_modules()
        if isinstance(module, HistoryEncoder) and not name.startswith("target")
    ]
    assert len(online_history) >= 2
    for name, module in online_history:
        assert any(
            not torch.equal(p, initial[name + "." + key]) for key, p in module.named_parameters()
        )
    assert all(
        p.grad is None for name, p in run.algorithm.named_parameters() if name.startswith("target")
    )

    checkpoint = tmp_path / "network.pt"
    run.trainer.save_checkpoint(checkpoint)
    if isinstance(run.trainer, OnPolicyTrainer):
        expected_metrics = run.trainer.train_rollout([9])
    else:
        expected_metrics = run.trainer.update()
    expected = deepcopy(run.algorithm.state_dict())
    restored = build_experiment(WindowAdapter(config.algorithm), config, seed=99)
    restored.trainer.load_checkpoint(checkpoint)
    if isinstance(restored.trainer, OnPolicyTrainer):
        actual_metrics = restored.trainer.train_rollout([9])
    else:
        actual_metrics = restored.trainer.update()
    assert actual_metrics == pytest.approx(expected_metrics, abs=1e-6)
    for key, value in restored.algorithm.state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=0, rtol=0)


@pytest.mark.parametrize("name", ["maac", "maddpg", "masac", "qmix", "mappo"])
def test_actors_are_local_and_parameters_are_independent(name):
    run = build_experiment(WindowAdapter(name), network_config(name, TEMPORAL[1], True), seed=6)
    policy = run.algorithm.get_submodule("policy")
    obs = torch.rand(3, 2, 7)
    first_output = policy.act(obs, deterministic=True)
    first = first_output.logits if first_output.logits is not None else first_output.actions
    changed = obs.clone()
    changed[:, 1] += 10
    changed_output = policy.act(changed, deterministic=True)
    following = (
        changed_output.logits if changed_output.logits is not None else changed_output.actions
    )
    torch.testing.assert_close(first[:, 0], following[:, 0])
    if name == "mappo":
        assert not torch.allclose(
            run.algorithm.critic(obs.flatten(-2)), run.algorithm.critic(changed.flatten(-2))
        )
    elif name == "maac":
        actions = torch.zeros(3, 2, dtype=torch.long)
        assert not torch.allclose(
            run.algorithm.critic(obs, actions)[:, 0], run.algorithm.critic(changed, actions)[:, 0]
        )
    elif name in ("maddpg", "masac"):
        critics = run.algorithm.critics
        critic = critics.critics[0] if name == "maddpg" else critics.first.critics[0]
        actions = first_output.actions.flatten(-2)
        assert not torch.allclose(
            critic(torch.cat((obs.flatten(-2), actions), -1)),
            critic(torch.cat((changed.flatten(-2), actions), -1)),
        )
    else:
        q = torch.ones(3, 2)
        assert not torch.allclose(
            run.algorithm.mixer(q, obs.flatten(-2)), run.algorithm.mixer(q, changed.flatten(-2))
        )
    # named_parameters 默认会去重，因此用 remove_duplicate=False 明确检查别名。
    parameters = list(run.algorithm.named_parameters(remove_duplicate=False))
    assert len({id(p) for _, p in parameters}) == len(parameters)


def test_qmix_graph_and_history_preserve_monotonicity():
    config = network_config("qmix", TEMPORAL[2], True)
    mixer = config.mixer.build(WindowAdapter("qmix").spec)
    q = torch.randn(4, 2, requires_grad=True)
    state = torch.randn(4, 14)
    derivative = torch.autograd.grad(mixer(q, state).sum(), q)[0]
    assert (derivative >= 0).all()


@pytest.mark.parametrize("name", ["maddpg", "masac", "maac"])
def test_actor_optimization_respects_parameter_ownership(name):
    config = network_config(name, TEMPORAL[1], True)
    env = WindowAdapter(name)
    run = build_experiment(env, config, seed=7)
    batch = transitions_to_batch(collect(run, env))
    algorithm = run.algorithm
    before = {key: p.clone() for key, p in algorithm.named_parameters()}
    bundle = (
        algorithm.compute_actor_loss_bundle(batch, 0)
        if name == "maac"
        else algorithm.compute_actor_loss_bundle(batch)
    )
    algorithm.optimize(run.trainer.optimization.optimizer("actor"), bundle, None)
    changed = [key for key, p in algorithm.named_parameters() if not torch.equal(p, before[key])]
    prefix = "policy.actors.0." if name == "maac" else "policy."
    assert changed and all(key.startswith(prefix) for key in changed)


def test_mappo_history_can_combine_with_streaming_backbones():
    config = network_config("mappo", TEMPORAL[1], True)
    config = replace(
        config,
        policy=replace(config.policy, backbone=GRUBackboneConfig(hidden_dim=8)),
        critic=replace(config.critic, backbone=LSTMBackboneConfig(hidden_dim=8)),
    )
    run = build_experiment(WindowAdapter("mappo"), config)
    assert all(np.isfinite(x) for x in run.trainer.train_rollout([4]).values())


def test_layout_and_checkpoint_fail_before_mutation():
    env = WindowAdapter("maac")
    config = network_config("maac", TEMPORAL[1], False)
    trainer = build_experiment(env, config).trainer
    saved = trainer.state_dict()
    assert saved["schema_version"] == 5 and saved["input_spec"]["observation_history"]
    before = deepcopy(trainer.algorithm.state_dict())
    for corruption in ("schema", "layout", "parameter", "buffer"):
        bad = deepcopy(saved)
        if corruption == "schema":
            bad["schema_version"] = 4
        elif corruption == "layout":
            bad["input_spec"]["observation_history"]["history_indices"] = ((3, 0), (4, 1), (5, 2))
        elif corruption == "parameter":
            key = next(iter(dict(trainer.algorithm.named_parameters())))
            bad["algorithm"][key] = torch.zeros(1)
        else:
            key = next(k for k in bad["algorithm"] if k.endswith("history_indices"))
            bad["algorithm"][key] = bad["algorithm"][key].flip(-1)
        rng = torch.get_rng_state().clone()
        with pytest.raises(ValueError):
            trainer.load_state_dict(bad)
        assert torch.equal(rng, torch.get_rng_state())
        assert all(torch.equal(v, before[k]) for k, v in trainer.algorithm.state_dict().items())
    changed_spec = replace(
        env.spec, observation_history=HistoryLayout(((3, 0), (4, 1), (5, 2)), (6,))
    )
    other = build_experiment(changed_spec, config).trainer
    with pytest.raises(ValueError, match="input_spec"):
        other.load_state_dict(saved)


def test_missing_invalid_layouts_and_network_positions():
    spec = WindowAdapter("mappo").spec
    with pytest.raises(ValueError, match="历史"):
        IndependentDiscreteConfig(encoder=HistoryEncoderConfig()).build(
            replace(spec, observation_history=None)
        )
    with pytest.raises(ValueError, match="state_layout"):
        CentralizedValueConfig(graph=GNNBackboneConfig()).build(replace(spec, state_layout=None))
    with pytest.raises(ValueError):
        HistoryLayout(((0, 1), (1, 2)), (3,))
    with pytest.raises(ValueError):
        replace(spec, adjacency=((0.0, -1.0), (0.0, 0.0)))
    for section in ("policy", "critic"):
        with pytest.raises(ValueError, match="旧/未知"):
            algorithm_config_from_dict(
                {"schema_version": 2, "algorithm": "maddpg", section: {"hidden_dim": 8}}
            )
    for network in ({"graph": {"kind": "gnn"}}, {"backbone": {"kind": "transformer"}}):
        with pytest.raises(ValueError):
            algorithm_config_from_dict(
                {"schema_version": 2, "algorithm": "maac", "policy": network}
            )
    for name in ("maddpg", "masac", "qmix"):
        with pytest.raises(ValueError):
            algorithm_config_from_dict(
                {"schema_version": 2, "algorithm": name, "policy": {"backbone": {"kind": "lstm"}}}
            )


def test_energy_maac_consumes_48_steps_and_updates_both_encoders():
    config = load_algorithm_config("examples/configs/algorithms/maac_energy_history.yaml")
    env = build_energy_trading_adapter(
        load_environment_config("examples/configs/environments/energy_trading.yaml")
    )
    run = build_experiment(env, config, seed=10)
    assert isinstance(run.trainer, OffPolicyTrainer)
    seen = []
    handles = []
    initial = {}
    for name, module in run.algorithm.named_modules():
        if isinstance(module, HistoryEncoder) and not name.startswith("target"):
            assert module.history_indices.shape == (48, 4)
            handles.append(
                module.temporal.register_forward_pre_hook(
                    lambda _module, args: seen.append(tuple(args[0].shape[-2:]))
                )
            )
            initial[name] = [p.detach().clone() for p in module.parameters()]
    current = env.reset(10)
    # 两条真实环境 transition 足够触发一次示例配置的离策略更新。
    for _ in range(config.replay.batch_size):
        with torch.inference_mode():
            actions = run.algorithm.act(
                torch.from_numpy(current.observations).unsqueeze(0),
                action_mask=torch.from_numpy(current.action_mask).unsqueeze(0),
            )[0].numpy()
        following = env.step(actions)
        run.trainer.record(Transition(current, actions, following))
        current = following
    metrics = run.trainer.update()
    assert metrics and all(np.isfinite(value) for value in metrics.values())
    assert seen and all(shape == (48, 4) for shape in seen)
    for name, before in initial.items():
        module = run.algorithm.get_submodule(name)
        assert any(not torch.equal(a, b) for a, b in zip(before, module.parameters(), strict=True))
    for handle in handles:
        handle.remove()
