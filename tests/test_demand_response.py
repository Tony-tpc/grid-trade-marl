"""物理、单位、经济奇点和可行清算；合成数组仅用于数学测试。"""
from dataclasses import replace

import numpy as np
import pytest

from marl.envs.demand_response import (
    DemandResponseConfig,
    DemandResponseEnv,
    FixedLeaderDemandAdapter,
    quadratic_ratio,
    retail_tariff,
)
from marl.envs.demand_response_data import OPSD_COLUMNS, OPSD_FILE, DemandProfiles, prepare_opsd


def profiles(load=1.0, der=1.0, n=3):
    return DemandProfiles(
        np.full((96, n), load), np.full(96, der), np.arange(96),
        np.datetime64('2016-06-01T00:00:00') + np.arange(96) * np.timedelta64(15, 'm'),
        np.zeros((96, n+1), dtype=bool), np.full(n, max(load, 1)), max(der, 1), 'test',
    )


def test_paper_m01_conservation_example_and_storage_signs():
    config = DemandResponseConfig(storage_kwh=100, storage_kw=20, enable_dr=False)
    env = DemandResponseEnv(profiles(100, 100, 1), config)
    env.reset()
    env.commit_leader(np.array([1, 1, .75, 0]))
    step = env.settle(np.array([[.4, -.4, -1, 0]]))
    expected = {'pg': 100, 'pb': 30, 'pa': 10, 'pc': 20, 'pd': 20,
                'pe': 0, 'po': 0, 'ps': 30}
    for key, value in expected.items():
        assert step.info['flows'][key] == pytest.approx(value, abs=1e-7)
    assert step.info['system_residual'] == pytest.approx(0, abs=1e-7)
    assert step.info['energy_kwh'] == pytest.approx(50 + .8*20*.25)
    before = env.energy
    env.commit_leader(np.array([-1, 0, -1, 0]))
    step = env.settle(np.array([[-1, 1, -1, 0]]))
    assert step.info['energy_kwh'] == pytest.approx(before-20*.25/.9)
    assert step.info['flows']['pc'] == 0


@pytest.mark.parametrize('seed', range(4))
def test_random_joint_actions_remain_feasible_and_settle_debt(seed):
    env = DemandResponseEnv(profiles())
    env.reset(seed)
    rng = np.random.default_rng(seed)
    shift_energy = np.zeros(3)
    for t in range(96):
        before = env.time
        env.commit_leader(rng.uniform(-1, 1, 4))
        assert env.time == before
        with pytest.raises(RuntimeError, match='阶段'):
            env.commit_leader(np.zeros(4))
        step = env.settle(rng.uniform(-1, 1, (3, 4)))
        flow = step.info['flows']
        executed = np.asarray(step.info['consumer_executed'])
        np.testing.assert_allclose(executed.sum(1), np.ones(3), atol=1e-7)
        assert flow['pg'] == pytest.approx(flow['pa']+flow['pd']+executed[:, 0].sum())
        assert flow['pb']+flow['po'] == pytest.approx(flow['pe']+executed[:, 1].sum())
        assert flow['pc']*flow['po'] == 0
        assert 0 <= env.energy <= env.config.storage_kwh
        assert abs(step.info['system_residual']) < 1e-7
        shift_energy += executed[:, 3]*.25
        costs = np.asarray(step.info['consumer_components']).sum(1)
        np.testing.assert_allclose(step.rewards[1:], -costs, rtol=1e-6)
        assert step.rewards[0] == pytest.approx(sum(step.info['uc_components']), rel=1e-6)
        assert step.terminated == (t == 95) and not step.truncated
    np.testing.assert_allclose(shift_energy, 0, atol=1e-7)
    with pytest.raises(RuntimeError):
        env.step(np.zeros((4, 4)))


def test_zero_denominators_and_signed_dr():
    np.testing.assert_array_equal(quadratic_ratio(np.array([0., -2., 2.]),
                                                 np.array([0., 1., 1.])), [0, 4, 4])
    with pytest.raises(ValueError, match='严格正'):
        quadratic_ratio(np.array([-1.]), np.array([0.]))
    env = DemandResponseEnv(profiles(load=0, der=0))
    env.reset()
    step = env.step(np.zeros((4, 4)))
    np.testing.assert_array_equal(step.rewards, 0)
    env = DemandResponseEnv(profiles(), replace(DemandResponseConfig(), enable_dr=False))
    env.reset()
    step = env.step(np.array([[0, 0, 1, -1], [1, -1, -1, 0],
                             [1, -1, -1, 0], [1, -1, -1, 0]]))
    assert np.isfinite(step.rewards).all()


def test_tariff_boundaries_and_fixed_leader_views():
    np.testing.assert_allclose(retail_tariff(80, np.array([0., 2.88e7, 4.8e7])),
                               [.078, .067, .060])
    assert retail_tariff(88, np.zeros(1))[0] == .048
    assert retail_tariff(36, np.zeros(1))[0] == .068
    adapter = FixedLeaderDemandAdapter(DemandResponseEnv(profiles()))
    step = adapter.reset(1)
    assert step.observations.shape == (3, 14)
    assert step.observations[0, 13] > 0  # 当前公开 DER offer 可见。
    step = adapter.step(np.zeros((3, 4)))
    assert step.rewards.shape == (3,) and adapter.environment.time == 1
    with pytest.raises(ValueError):
        adapter.environment.step(np.full((4, 4), 2))


def test_data_rejects_missing_negative_and_discontinuous_intervals():
    original = profiles()
    bad = original.load_kw.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match='非负'):
        replace(original, load_kw=bad)
    bad[0, 0] = -1
    with pytest.raises(ValueError):
        replace(original, load_kw=bad)
    utc = original.utc.copy()
    utc[2] += np.timedelta64(1, 'h')
    with pytest.raises(ValueError, match='连续'):
        replace(original, utc=utc)


def test_cumulative_meter_units_local_split_and_no_scale_leakage(tmp_path):
    pd = pytest.importorskip('pandas')
    end = pd.date_range('2016-05-31T21:45Z', '2016-09-30T22:00Z', freq='15min')
    start = end-pd.Timedelta(minutes=15)
    rate = np.tile(np.arange(1., 5.), (len(end), 1))
    rate[start >= pd.Timestamp('2016-07-31T22:00Z')] *= 10
    frame = pd.DataFrame(np.cumsum(rate*.25, axis=0), columns=OPSD_COLUMNS)
    frame['utc_timestamp'] = end.astype(str)
    frame['interpolated'] = ''
    frame.loc[2, 'interpolated'] = OPSD_COLUMNS[0]
    frame.to_csv(tmp_path/OPSD_FILE, index=False)
    manifest = prepare_opsd(tmp_path)
    train = DemandProfiles.load(tmp_path/'train.npz')
    validation = DemandProfiles.load(tmp_path/'validation.npz')
    test = DemandProfiles.load(tmp_path/'test.npz')
    assert len(train.load_kw) == 5856 and len(validation.load_kw) == 2976
    assert len(test.load_kw) == 2880
    np.testing.assert_allclose(train.load_scale, [1, 2, 3])
    np.testing.assert_allclose(test.load_scale, train.load_scale)
    np.testing.assert_allclose(test.load_kw[0], [10, 20, 30])
    assert train.der_scale == 4 and train.imputed[:, 0].sum() == 2
    assert str(train.utc[0]) == '2016-05-31T22:00:00'
    assert len(manifest['sha256']) == 64
