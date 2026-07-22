"""Gymnasium 파사드 검증: API 준수, 관찰 규격, info 계약, 재현성."""

import math
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import rocket_env  # noqa: F401  — 환경 등록 트리거
from rocket_env.config import PRESETS
from rocket_env.env import OBS_DIM, RocketEnv
from rocket_env.reward import potential
from rocket_env.types import Outcome

NOOP = 1        # 추력 0, 노즐 정지
FULL_UP = 10    # 추력 2g, 노즐 정지


def rollout(env, action, seed):
    env.reset(seed=seed)
    total = 0.0
    while True:
        _, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            return total, info


def test_registered_ids_are_makeable():
    for env_id in ("rocket-v0", "rocket-landing-v0", "rocket-catch-v0"):
        env = gym.make(env_id)
        env.reset(seed=0)
        env.close()


def test_alias_ids_select_the_right_task():
    assert gym.make("rocket-catch-v0").unwrapped.cfg["task"] == "catch"
    assert gym.make("rocket-landing-v0").unwrapped.cfg["task"] == "landing"


def test_alias_id_still_accepts_extra_config():
    env = gym.make("rocket-catch-v0", config={"max_steps": 123})
    assert env.unwrapped.cfg["task"] == "catch"
    assert env.unwrapped.cfg["max_steps"] == 123


def test_passes_gymnasium_env_checker():
    check_env(RocketEnv(), skip_render_check=True)


def test_spaces_match_the_contract():
    env = RocketEnv()
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.n == 12


def test_observation_is_finite_and_correctly_typed():
    env = RocketEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))


def test_unlimited_fuel_shows_full_fuel_fraction():
    env = RocketEnv(config=PRESETS["landing-basic"])
    obs, _ = env.reset(seed=0)
    assert obs[8] == pytest.approx(1.0)


def test_info_contains_every_contract_key():
    env = RocketEnv()
    env.reset(seed=0)
    _, _, _, _, info = env.step(NOOP)
    for key in ("is_success", "outcome", "fuel_left", "fuel_frac",
                "impact_speed", "wind_x", "step"):
        assert key in info


def test_same_seed_reproduces_identical_trajectories():
    """같은 시드는 bit-exact 동일 궤적을, 다른 시드는 다른 궤적을 만든다.

    채점 워커가 reset(seed=base + i) 로 에피소드를 재현하므로, 이 성질이
    깨지면 같은 모델이 실행할 때마다 다른 점수를 받는다.

    에피소드 점수는 대리 지표로 쓸 수 없다 — shaping 이 정확히 상쇄되고
    실패 근접도가 포화되면 궤적이 달라도 점수가 같아진다. 관찰 시퀀스
    전체를 비교한다.
    """
    def trajectory(seed: int) -> np.ndarray:
        env = RocketEnv(config=PRESETS["landing-gust"])
        obs, _ = env.reset(seed=seed)
        frames = [obs]
        while True:
            obs, _, terminated, truncated, _ = env.step(NOOP)
            frames.append(obs)
            if terminated or truncated:
                env.close()
                return np.stack(frames)

    a, b, c = trajectory(123), trajectory(123), trajectory(124)
    assert np.array_equal(a, b)
    assert not (a.shape == c.shape and np.array_equal(a, c))


def test_config_seed_is_not_consumed_by_the_env():
    """cfg['seed']는 호출자 메타데이터다. 환경이 읽으면 학습 시
    모든 에피소드가 동일해지는 버그가 생긴다."""
    env = RocketEnv(config={**PRESETS["landing-descent"], "seed": 7})
    env.reset()
    first = env.unwrapped.state.x
    env.reset()
    assert env.unwrapped.state.x != first


def test_zero_thrust_from_altitude_ends_in_crash():
    env = RocketEnv(config=PRESETS["landing-descent"])
    _, info = rollout(env, NOOP, seed=0)
    assert info["outcome"] == Outcome.CRASH
    assert info["is_success"] is False
    assert info["impact_speed"] is not None


def test_running_out_of_fuel_is_reported_distinctly():
    env = RocketEnv(config={**PRESETS["landing-descent"],
                            "fuel": {"capacity": 1.0}})
    _, info = rollout(env, FULL_UP, seed=0)
    assert info["outcome"] == Outcome.OUT_OF_FUEL


def test_timeout_truncates_rather_than_terminates():
    """추력 1g 부근으로 떠 있으면 max_steps에 걸린다."""
    env = RocketEnv(config={**PRESETS["landing-basic"], "max_steps": 30})
    env.reset(seed=0)
    for _ in range(30):
        obs, reward, terminated, truncated, info = env.step(7)  # 1.0g, 노즐 정지
    assert truncated
    assert not terminated
    assert info["outcome"] == Outcome.TIMEOUT


def test_fuel_never_goes_negative():
    env = RocketEnv(config={**PRESETS["landing-descent"],
                            "fuel": {"capacity": 2.0}})
    env.reset(seed=0)
    for _ in range(200):
        _, _, terminated, truncated, info = env.step(FULL_UP)
        assert info["fuel_left"] >= 0.0
        if terminated or truncated:
            break


def test_wind_disabled_config_keeps_wind_at_zero():
    env = RocketEnv(config=PRESETS["landing-basic"])
    env.reset(seed=0)
    for _ in range(50):
        _, _, terminated, truncated, info = env.step(NOOP)
        assert info["wind_x"] == 0.0
        if terminated or truncated:
            break


def test_step_reward_matches_hand_computed_shaping():
    """스텝 보상이 실제로 Φ 변화를 반영하는지 수치로 확인한다.

    _potential 을 읽기 전에 덮어쓰면 shaping 이 매 스텝 정확히 0이 되는데,
    나머지 테스트는 전부 통과한다 — 아무도 보상값을 보지 않기 때문이다.
    추력 0인 NOOP 을 써서 연료 패널티 항을 0으로 만들고 shaping 만 남긴다.
    """
    env = RocketEnv(config=PRESETS["landing-basic"])
    env.reset(seed=0)
    inner = env.unwrapped
    before = potential(inner.state, inner._target, inner.cfg)

    _, reward, _, _, _ = env.step(NOOP)

    after = potential(inner.state, inner._target, inner.cfg)
    gamma = inner.cfg["reward"]["shaping_gamma"]
    assert reward == pytest.approx(gamma * after - before)
    assert reward != 0.0


def test_observation_places_sin_and_cos_in_the_right_slots():
    """자세 30도면 sin=0.5, cos=0.866 으로 값이 뚜렷이 달라 뒤바뀜이 드러난다."""
    env = RocketEnv(config=PRESETS["landing-descent"])
    env.reset(seed=0)
    inner = env.unwrapped
    inner.state = replace(inner.state, theta=math.radians(30.0))
    obs = inner._observation()
    assert obs[4] == pytest.approx(0.5, abs=1e-6)
    assert obs[5] == pytest.approx(math.sqrt(3.0) / 2.0, abs=1e-6)


def test_observation_uses_distinct_scales_for_position_and_velocity():
    """위치와 속도를 같은 상수로 나누면 물리적으로 다른 양이 같은 값이 된다."""
    env = RocketEnv(config=PRESETS["landing-descent"])
    env.reset(seed=0)
    inner = env.unwrapped
    tx, ty = inner._target
    inner.state = replace(inner.state, x=tx + 450.0, y=ty, vx=100.0, vy=0.0)
    obs = inner._observation()
    assert obs[0] == pytest.approx(0.5)    # 450 / 900
    assert obs[1] == pytest.approx(0.0)
    assert obs[2] == pytest.approx(0.5)    # 100 / 200
    assert obs[3] == pytest.approx(0.0)


def test_observation_reports_time_and_wind_fractions():
    env = RocketEnv(config={**PRESETS["landing-descent"], "max_steps": 100})
    env.reset(seed=0)
    inner = env.unwrapped
    inner.state = replace(inner.state, step=25, wind_x=10.0)
    obs = inner._observation()
    assert obs[9] == pytest.approx(0.5)     # 10 / 20
    assert obs[10] == pytest.approx(0.25)   # 25 / 100


def test_catch_task_can_be_selected_by_config():
    env = RocketEnv(config=PRESETS["catch"])
    _, info = rollout(env, NOOP, seed=0)
    assert info["outcome"] in (Outcome.MISSED, Outcome.CRASH)
