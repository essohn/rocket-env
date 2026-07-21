"""보상 설계 검증.

가장 중요한 것은 shaping 총합이 에피소드 길이와 무관하다는 점과,
실패 보상에 시간 항이 전혀 없다는 점이다 (원본 환경의 '빨리 자폭' 버그 방지).
"""

import math

import pytest

from rocket_env.config import PRESETS, build_config
from rocket_env.reward import (
    distance_to_target,
    potential,
    shaping,
    terminal_reward,
)
from rocket_env.types import Outcome, State

CFG = build_config(PRESETS["landing-normal"])
TARGET = (0.0, 25.0)


def at(**kw) -> State:
    base = dict(x=0.0, y=200.0, vx=0.0, vy=0.0, theta=0.0, omega=0.0,
                phi=0.0, thrust=0.0, fuel=100.0, wind_x=0.0, step=100)
    base.update(kw)
    return State(**base)


def test_potential_is_zero_at_target_and_upright():
    assert potential(at(x=0.0, y=25.0), TARGET, CFG) == pytest.approx(0.0)


def test_potential_is_negative_away_from_target():
    assert potential(at(x=100.0), TARGET, CFG) < 0.0


def test_potential_penalises_tilt():
    upright = potential(at(), TARGET, CFG)
    tilted = potential(at(theta=math.radians(45.0)), TARGET, CFG)
    assert tilted < upright


def test_shaping_sum_telescopes_exactly():
    """shaping_gamma=1.0이면 총합이 정확히 Φ(s_T) - Φ(s_0)다.

    이것이 '목표 근처에서 오래 버티며 점수 쌓기'를 구조적으로 막는다.
    """
    trajectory = [at(x=float(i), y=200.0 - i) for i in range(0, 300)]
    potentials = [potential(s, TARGET, CFG) for s in trajectory]
    total = sum(shaping(potentials[i], potentials[i + 1], CFG)
                for i in range(len(potentials) - 1))
    assert total == pytest.approx(potentials[-1] - potentials[0], abs=1e-9)


def test_shaping_total_is_independent_of_episode_length():
    short = [at(x=0.0, y=200.0), at(x=0.0, y=100.0)]
    long = [at(x=0.0, y=200.0)] + [at(x=0.0, y=150.0)] * 500 + [at(x=0.0, y=100.0)]

    def total(states):
        p = [potential(s, TARGET, CFG) for s in states]
        return sum(shaping(p[i], p[i + 1], CFG) for i in range(len(p) - 1))

    assert total(short) == pytest.approx(total(long), abs=1e-9)


def test_perfect_landing_scores_near_the_maximum():
    r = terminal_reward(Outcome.SUCCESS, at(x=0.0, y=25.0, step=0),
                        TARGET, CFG, d_initial=425.0, fuel_frac=1.0)
    assert r == pytest.approx(250.0, abs=0.5)


def test_marginal_success_still_beats_every_failure():
    """최악의 성공도 최고의 실패보다 커야 한다."""
    s = CFG["success"]
    marginal = terminal_reward(
        Outcome.SUCCESS,
        at(x=s["zone_r"] - 0.01, y=25.0,
           vy=-(s["v_max"] - 0.01),
           theta=math.radians(s["theta_max_deg"] - 0.01),
           step=CFG["max_steps"]),
        TARGET, CFG, d_initial=425.0, fuel_frac=0.0)
    best_failure = CFG["reward"]["failure_max"]
    assert marginal > best_failure


def test_crash_at_start_position_scores_zero():
    """진행이 없으면 부분 점수도 없다 — '빨리 자폭' 전략의 봉쇄."""
    start = at(x=0.0, y=450.0)
    d0 = distance_to_target(start, TARGET)
    r = terminal_reward(Outcome.CRASH, start, TARGET, CFG,
                        d_initial=d0, fuel_frac=0.5)
    assert r == pytest.approx(0.0, abs=1e-9)


def test_crash_at_target_scores_the_failure_maximum():
    r = terminal_reward(Outcome.CRASH, at(x=0.0, y=25.0), TARGET, CFG,
                        d_initial=425.0, fuel_frac=0.0)
    assert r == pytest.approx(CFG["reward"]["failure_max"])


def test_failure_reward_has_no_time_term():
    """같은 상태라면 언제 끝났든 실패 점수는 동일하다.

    원본 환경은 실패 보상에 (max_steps - step)을 곱해서 조기 자폭이
    고득점이 되었다. 이 테스트가 그 회귀를 막는다.
    """
    early = terminal_reward(Outcome.CRASH, at(x=10.0, y=100.0, step=5),
                            TARGET, CFG, d_initial=425.0, fuel_frac=0.9)
    late = terminal_reward(Outcome.CRASH, at(x=10.0, y=100.0, step=790),
                           TARGET, CFG, d_initial=425.0, fuel_frac=0.1)
    assert early == pytest.approx(late)


def test_all_failure_outcomes_share_the_same_formula():
    """네 가지 실패는 동일한 값을 내야 한다.

    각각 범위 안에 있는지만 보면, MISSED 에만 다른 공식이 붙어도 통과한다.
    """
    state = at(x=10.0, y=100.0)
    scores = [
        terminal_reward(outcome, state, TARGET, CFG,
                        d_initial=425.0, fuel_frac=0.5)
        for outcome in (Outcome.CRASH, Outcome.MISSED,
                        Outcome.TIMEOUT, Outcome.OUT_OF_FUEL)
    ]
    assert len(set(scores)) == 1
    assert 0.0 <= scores[0] <= CFG["reward"]["failure_max"]


def test_catch_profile_rewards_slow_contact_much_more_steeply():
    catch_cfg = build_config(PRESETS["catch-normal"])
    target = (0.0, catch_cfg["catch"]["y_arm"])

    def score(speed):
        return terminal_reward(
            Outcome.SUCCESS,
            at(x=0.0, y=target[1], vy=-speed, step=0),
            target, catch_cfg, d_initial=400.0, fuel_frac=1.0)

    assert score(0.0) - score(1.0) > score(3.0) - score(4.0)


def test_crash_without_progress_scores_zero_at_any_initial_distance():
    """진행이 없으면 출발 거리와 무관하게 0점이다.

    예전 구현은 분모를 max(d_initial, 1.0)으로 눌렀다. 그러면 목표
    근처에서 출발했을 때 분모만 1.0으로 올라가고 분자는 작은 실제 거리라서,
    제자리 추락에도 양수 점수가 나온다 — 이 모듈이 막으려는 바로 그
    '빨리 자폭' 꼼수다. 현재 프리셋은 d_initial >= 370 이라 도달할 수
    없지만, 저고도에서 시작하는 라운드를 새로 만들면 되살아난다.
    """
    for d in (0.5, 2.0, 425.0):
        state = at(x=d, y=25.0)          # 목표에서 정확히 d 만큼 떨어진 지점
        r = terminal_reward(Outcome.CRASH, state, TARGET, CFG,
                            d_initial=d, fuel_frac=0.5)
        assert r == pytest.approx(0.0)


def test_zero_initial_distance_scores_zero():
    r = terminal_reward(Outcome.CRASH, at(x=0.0, y=25.0), TARGET, CFG,
                        d_initial=0.0, fuel_frac=0.0)
    assert r == 0.0


def test_shaping_reads_gamma_from_config():
    """shaping 이 cfg 의 감가율을 실제로 읽는지 확인한다.

    gamma 를 1.0으로 하드코딩한 구현도 기본 설정에서는 텔레스코핑
    테스트를 전부 통과한다. 다른 값을 넣어야 비로소 드러난다.
    """
    cfg = build_config({"reward": {"shaping_gamma": 0.5}})
    assert shaping(-2.0, -1.0, cfg) == pytest.approx(0.5 * -1.0 - (-2.0))
