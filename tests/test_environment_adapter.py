"""验证论文环境到算法协议的边界，避免更换环境时改动算法。"""

from dataclasses import replace

import numpy as np
import pytest
import torch

from marl.algorithms import (
    MAAC,
    MASAC,
    QMIX,
    MAACConfig,
    MASACConfig,
    QMIXConfig,
)
from marl.envs import (
    ActionKind,
    EnergyProfiles,
    EnergyTradingAdapter,
    EnergyTradingConfig,
    EnergyTradingEnv,
    EnvironmentAdapter,
    EnvironmentSpec,
    EnvironmentStep,
    RewardStructure,
    Transition,
    transitions_to_batch,
)
from marl.experiment import build_experiment
from marl.models import MLPBackboneConfig
from marl.modules.critic import AttentionQConfig
from marl.modules.policy import IndependentDiscreteConfig
from marl.training.off_policy import ReplayConfig


def test_energy_adapter_builds_config_algorithm_and_trainable_batch() -> None:
    config = EnergyTradingConfig(num_agents=2, has_sa=(True, False))
    adapter = EnergyTradingAdapter(EnergyTradingEnv(config), ActionKind.DISCRETE)
    spec = adapter.spec
    assert spec.num_agents == 2
    assert spec.action_dim == 54
    assert spec.state_dim == spec.num_agents * spec.observation_dim

    current = adapter.reset(seed=7)
    assert current.observations.shape == (2, spec.observation_dim)
    assert current.action_mask is not None
    assert current.action_mask.shape == (2, 54)
    transitions = []
    for _ in range(3):
        assert current.action_mask is not None
        actions = current.action_mask.argmax(axis=-1).astype(np.int64)
        following = adapter.step(actions)
        transitions.append(Transition(current, actions, following))
        current = following

    batch = transitions_to_batch(transitions)
    assert batch.observations.shape == (3, 2, spec.observation_dim)
    assert batch.actions is not None and batch.actions.dtype == torch.long
    assert batch.rewards is not None and batch.rewards.shape == (3, 2)
    assert batch.state is not None and batch.state.shape == (3, spec.state_dim)

    algorithm_config = replace(
        MAACConfig(),
        policy=IndependentDiscreteConfig(backbone=MLPBackboneConfig(output_dim=16)),
        critic=AttentionQConfig(
            embedding=MLPBackboneConfig(hidden_dims=(), output_dim=16),
            backbone=MLPBackboneConfig(hidden_dims=(), output_dim=16),
            attention_heads=4,
        ),
        replay=ReplayConfig(capacity=8, batch_size=3),
    )
    trainer = build_experiment(spec, algorithm_config, seed=8).trainer
    metrics = trainer.update_batch(batch)
    assert np.isfinite(metrics["loss"])

    with pytest.raises(ValueError, match="动作"):
        MASAC(spec, MASACConfig())
    with pytest.raises(ValueError, match="共享"):
        QMIX(spec, QMIXConfig())


def test_market_settlement_conserves_external_cash_flow() -> None:
    config = EnergyTradingConfig(
        num_agents=2,
        grid_limit_kw=100,
        has_ev=(False, False),
        has_storage=(False, False),
        has_hvac=(False, False),
        has_sa=(False, False),
    )
    horizon = config.horizon
    profiles = EnergyProfiles(
        demand_kw=np.tile([2.0, 0.0], (horizon, 1)),
        pv_kw=np.tile([0.0, 1.0], (horizon, 1)),
        buy_price=np.ones(horizon),
        sell_price=np.full(horizon, 0.2),
        outdoor_c=np.full(horizon, 23.0),
    )
    adapter = EnergyTradingAdapter(EnergyTradingEnv(config, profiles), ActionKind.CONTINUOUS)
    adapter.reset()
    step = adapter.step(np.zeros((2, 4), dtype=np.float32))

    assert step.info["local_buy_price"] == pytest.approx(0.8)
    assert step.info["local_sell_price"] == pytest.approx(0.6)
    # 社区净购电 1 kW × 半小时 × 供电商价格 1 = 0.5。
    assert step.info["community_energy_cost"] == pytest.approx(0.5)
    assert step.rewards.sum() == pytest.approx(-0.5)


def test_full_day_keeps_storage_within_bounds_and_enforces_sa_once() -> None:
    config = EnergyTradingConfig(num_agents=2)
    adapter = EnergyTradingAdapter(EnergyTradingEnv(config), ActionKind.CONTINUOUS)
    step = adapter.reset(seed=11)
    for _ in range(config.horizon):
        # EV 与储能持续充电；智能电器在第一段可用时间自动启动。
        step = adapter.step(np.ones((2, 4), dtype=np.float32))
        assert (adapter.environment.ev_energy >= 0).all()
        assert (adapter.environment.ev_energy <= config.ev_capacity_kwh).all()
        assert (adapter.environment.storage_energy >= 0).all()
        assert (adapter.environment.storage_energy <= config.storage_capacity_kwh).all()
    assert step.terminated and not step.truncated
    assert adapter.environment.sa_started.all()
    assert (adapter.environment.sa_progress == len(config.sa_cycle_kw)).all()
    with pytest.raises(RuntimeError, match="reset"):
        adapter.step(np.zeros((2, 4)))


def test_new_environment_dimensions_only_change_adapter_spec() -> None:
    small = EnergyTradingAdapter(
        EnergyTradingEnv(EnergyTradingConfig(num_agents=2, history_steps=4)), ActionKind.DISCRETE
    )
    large = EnergyTradingAdapter(
        EnergyTradingEnv(EnergyTradingConfig(num_agents=4, history_steps=8)), ActionKind.DISCRETE
    )
    small_algorithm = MAAC(small.spec, MAACConfig())
    large_algorithm = MAAC(large.spec, MAACConfig())
    assert small_algorithm.spec != large_algorithm.spec
    assert len(list(small_algorithm.get_submodule("policy.actors").children())) == 2
    assert len(list(large_algorithm.get_submodule("policy.actors").children())) == 4


def test_different_paper_adapter_uses_same_algorithm_and_batch_contract() -> None:
    class OtherPaperAdapter(EnvironmentAdapter):
        """模拟另一篇论文的环境：维度、回合长度与交易环境完全不同。"""

        @property
        def spec(self) -> EnvironmentSpec:
            return EnvironmentSpec(3, 4, 2, 12, ActionKind.DISCRETE, 2, RewardStructure.SHARED)

        def reset(self, seed: int | None = None) -> EnvironmentStep:
            return self.validate_step(
                EnvironmentStep(
                    observations=np.zeros((3, 4), dtype=np.float32),
                    state=np.zeros(12, dtype=np.float32),
                    rewards=np.zeros(3, dtype=np.float32),
                    terminated=False,
                    truncated=False,
                    action_mask=np.ones((3, 2), dtype=bool),
                )
            )

        def step(self, actions: np.ndarray) -> EnvironmentStep:
            assert actions.shape == (3,)
            return self.validate_step(
                EnvironmentStep(
                    observations=np.ones((3, 4), dtype=np.float32),
                    state=np.ones(12, dtype=np.float32),
                    rewards=np.ones(3, dtype=np.float32),
                    terminated=True,
                    truncated=False,
                    action_mask=np.ones((3, 2), dtype=bool),
                )
            )

    adapter = OtherPaperAdapter()
    algorithm = QMIX(adapter.spec, QMIXConfig())
    current = adapter.reset()
    actions = algorithm.act(torch.as_tensor(current.observations).unsqueeze(0))[0].numpy()
    following = adapter.step(actions)
    batch = transitions_to_batch([Transition(current, actions, following)])
    assert algorithm.compute_loss_bundle(batch).total.ndim == 0

    with pytest.raises(ValueError, match="observations 形状"):
        adapter.validate_step(
            EnvironmentStep(
                observations=np.zeros((2, 4)),
                state=np.zeros(12),
                rewards=np.zeros(3),
                terminated=False,
                truncated=False,
                action_mask=np.ones((3, 2), dtype=bool),
            )
        )
