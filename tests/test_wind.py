"""OU 바람 프로세스 검증."""

import numpy as np
import pytest

from rocket_env.wind import WindProcess


def gust(max_speed=12.0, ou_theta=0.15, ou_sigma=3.0) -> WindProcess:
    return WindProcess(mode="gust", max_speed=max_speed,
                       ou_theta=ou_theta, ou_sigma=ou_sigma)


def test_none_mode_is_always_zero():
    w = WindProcess(mode="none", max_speed=0.0, ou_theta=0.0, ou_sigma=0.0)
    rng = np.random.default_rng(0)
    value = w.reset(rng)
    assert value == 0.0
    for _ in range(100):
        value = w.step(value, rng)
        assert value == 0.0


def test_constant_mode_holds_its_reset_value():
    w = WindProcess(mode="constant", max_speed=8.0, ou_theta=0.0, ou_sigma=0.0)
    rng = np.random.default_rng(1)
    value = w.reset(rng)
    assert -8.0 <= value <= 8.0
    for _ in range(100):
        assert w.step(value, rng) == pytest.approx(value)


def test_gust_mode_actually_varies():
    w = gust()
    rng = np.random.default_rng(2)
    value = w.reset(rng)
    series = [value := w.step(value, rng) for _ in range(200)]
    assert np.std(series) > 0.1


def test_gust_stays_within_max_speed():
    w = gust(max_speed=5.0, ou_sigma=50.0)
    rng = np.random.default_rng(3)
    value = w.reset(rng)
    for _ in range(500):
        value = w.step(value, rng)
        assert -5.0 <= value <= 5.0


def test_same_seed_gives_identical_wind_sequence():
    def run(seed):
        w = gust()
        rng = np.random.default_rng(seed)
        value = w.reset(rng)
        return [value := w.step(value, rng) for _ in range(50)]

    assert run(7) == run(7)
    assert run(7) != run(8)


def test_mean_reversion_pulls_toward_zero():
    """ou_sigma=0이면 결정론적 평균회귀만 남아 0으로 수렴한다."""
    w = WindProcess(mode="gust", max_speed=20.0, ou_theta=5.0, ou_sigma=0.0)
    rng = np.random.default_rng(4)
    value = 10.0
    for _ in range(500):
        value = w.step(value, rng)
    assert abs(value) < 0.1
