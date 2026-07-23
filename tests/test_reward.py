"""보상 설계 검증.

가장 중요한 것은 shaping 총합이 에피소드 길이와 무관하다는 점과,
실패 보상에 시간 항이 전혀 없다는 점이다 (원본 환경의 '빨리 자폭' 버그 방지).
"""

import math

import pytest

from rocket_env.config import PRESETS, build_config
from rocket_env.reward import potential, shaping, terminal_reward
from rocket_env.types import Outcome, State

CFG = build_config(PRESETS["landing-descent"])
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
    # landing-descent 는 착륙 데미지 채점이 엄격해 success_soft 160 이다.
    # failure_max(40) + success_soft(160) + success_position(20) + success_fuel(20)
    r = terminal_reward(Outcome.SUCCESS, at(x=0.0, y=25.0, step=0),
                        TARGET, CFG, fuel_frac=1.0)
    assert r == pytest.approx(240.0, abs=0.5)


def test_marginal_success_still_beats_every_failure():
    """최악의 성공도 최고의 실패보다 커야 한다."""
    s = CFG["success"]
    marginal = terminal_reward(
        Outcome.SUCCESS,
        at(x=s["zone_r"] - 0.01, y=25.0,
           vy=-(s["v_max"] - 0.01),
           theta=math.radians(s["theta_max_deg"] - 0.01),
           step=CFG["max_steps"]),
        TARGET, CFG, fuel_frac=0.0)
    best_failure = CFG["reward"]["failure_max"]
    assert marginal > best_failure


def test_freefall_impact_scores_low():
    """자유낙하로 지면에 꽂히면 아주 낮은 점수를 받는다(무행동 악용 방지).

    도달도가 전 구간 gradual 이라 종단속도 충돌도 정확히 0은 아니지만, 제어된
    접지보다 훨씬 낮다. 핵심은 '더 느리게 부딪힐수록' 항상 점수가 높아 제동을
    배울 gradient 가 살아 있다는 것 — 예전의 일괄 0점 바닥은 이 신호를 죽였다.
    """
    freefall = terminal_reward(Outcome.CRASH, at(x=0.0, y=25.0, vy=-49.5),
                               TARGET, CFG, fuel_frac=0.5)
    controlled = terminal_reward(Outcome.CRASH, at(x=0.0, y=25.0, vy=-16.0),
                                 TARGET, CFG, fuel_frac=0.5)
    assert 0.0 < freefall < 0.15 * controlled


def test_high_hover_timeout_scores_zero():
    """목표 고도에 도달하지 못하면 다른 조건이 완벽해도 0점이다."""
    hovering = at(x=0.0, y=400.0, vx=0.0, vy=0.0)
    r = terminal_reward(Outcome.TIMEOUT, hovering, TARGET, CFG, fuel_frac=1.0)
    assert r == pytest.approx(0.0)


def test_success_attitude_bonus_is_the_same_for_theta_zero_and_two_pi():
    """관찰이 sin/cos 라 θ=0 과 θ=2π 를 구별 못 하므로 점수도 같아야 한다."""
    upright = terminal_reward(Outcome.SUCCESS, at(x=0.0, y=25.0, theta=0.0),
                              TARGET, CFG, fuel_frac=1.0)
    spun = terminal_reward(Outcome.SUCCESS,
                           at(x=0.0, y=25.0, theta=2.0 * math.pi),
                           TARGET, CFG, fuel_frac=1.0)
    assert spun == pytest.approx(upright)


def test_at_threshold_failure_equals_the_success_floor():
    """모든 축이 정확히 임계값에서 실패하면 성공식의 '바닥'과 같은 값을 받는다.

    _reach 가 임계값에서 1.0이라 closeness=1, 실패 = failure_max + 위치 + 연료.
    성공도 그 값에서 (거의 0인 속도 보너스만 더해) 시작하므로 경계가 연속이다.
    예전의 실패~40 → 성공~80 점프(절벽)가 사라진다.
    """
    s = CFG["success"]
    r_ = CFG["reward"]
    near = at(x=0.0, y=25.0, vy=-s["v_max"])   # 속도만 임계값, 나머지는 목표에서 완벽
    r = terminal_reward(Outcome.CRASH, near, TARGET, CFG, fuel_frac=0.5)
    assert r == pytest.approx(r_["failure_max"]
                              + r_["success_position"]
                              + r_["success_fuel"] * 0.5)


def test_failure_to_success_boundary_is_continuous():
    """경계를 사이에 둔 실패와 성공의 점수 차가 크지 않다(절벽 없음)."""
    s = CFG["success"]
    # 속도가 임계값을 아주 살짝 넘은 실패 vs 아주 살짝 못 넘은 성공
    fail = terminal_reward(Outcome.CRASH,
                           at(x=0.0, y=25.0, vy=-(s["v_max"] + 0.01)),
                           TARGET, CFG, fuel_frac=0.0)
    succ = terminal_reward(Outcome.SUCCESS,
                           at(x=0.0, y=25.0, vy=-(s["v_max"] - 0.01)),
                           TARGET, CFG, fuel_frac=0.0)
    # 성공에는 중심 정렬 보너스(dx=0)가 붙지만, 점프는 예전 절벽(약 180)의
    # 몇 분의 일 수준이어야 한다.
    assert succ >= fail
    assert succ - fail < 40.0


def test_weakest_criterion_determines_the_failure_score():
    """다섯 조건 중 가장 나쁜 것이 점수를 정한다.

    성공하려면 전부 만족해야 하므로 부분 점수도 가장 약한 고리를 따른다.
    평균을 쓰면 '속도만 빼고 완벽'이 높은 점수를 받아 자유낙하가 되살아난다.
    """
    s = CFG["success"]
    r_ = CFG["reward"]
    good = at(x=0.0, y=25.0, vy=-1.0)
    bad_speed = at(x=0.0, y=25.0, vy=-2.0 * s["v_max"])   # 속도만 임계값 2배
    good_score = terminal_reward(Outcome.CRASH, good, TARGET, CFG, fuel_frac=0.0)
    bad_score = terminal_reward(Outcome.CRASH, bad_speed, TARGET, CFG, fuel_frac=0.0)
    # 위치·자세가 완벽해도 속도(가장 약한 고리) 하나가 전체를 끌어내린다.
    assert 0.0 < bad_score < good_score
    # 그 값은 정확히 속도 도달도로 스케일된다(min = 가장 약한 고리).
    assert bad_score == pytest.approx(
        math.exp(-1.0) * (r_["failure_max"] + r_["success_position"]))


def test_potential_penalises_speed():
    """Φ 가 속도를 반영해야 밀집 신호가 성공 기준을 가리킨다."""
    assert potential(at(vy=-40.0), TARGET, CFG) < potential(at(vy=-1.0), TARGET, CFG)


def test_potential_wraps_the_attitude_angle():
    """여러 바퀴 돌아도 Φ 가 무한정 작아지지 않는다.

    물리는 θ 를 감지 않는데 관찰은 sin/cos 라 감김 횟수를 볼 수 없다.
    보상만 그것에 의존하면 관찰로 구분 불가능한 두 상태가 다른 값을 갖는다.
    """
    assert potential(at(theta=0.1), TARGET, CFG) == pytest.approx(
        potential(at(theta=0.1 + 2.0 * math.pi), TARGET, CFG))


def test_failure_reward_has_no_time_term():
    """같은 상태라면 언제 끝났든 실패 점수는 동일하다.

    원본 환경은 실패 보상에 (max_steps - step)을 곱해서 조기 자폭이
    고득점이 되었다. 이 테스트가 그 회귀를 막는다. (연료는 점수 축이므로
    같게 두고, step 만 다르게 해서 시간 의존이 없음을 확인한다.)
    """
    early = terminal_reward(Outcome.CRASH, at(x=10.0, y=100.0, step=5),
                            TARGET, CFG, fuel_frac=0.5)
    late = terminal_reward(Outcome.CRASH, at(x=10.0, y=100.0, step=790),
                           TARGET, CFG, fuel_frac=0.5)
    assert early == pytest.approx(late)


def test_contact_failures_share_the_same_formula():
    """접지·포획실패·연료소진은 같은 공식을 쓴다."""
    state = at(x=10.0, y=100.0)
    scores = [terminal_reward(o, state, TARGET, CFG, fuel_frac=0.5)
              for o in (Outcome.CRASH, Outcome.MISSED, Outcome.OUT_OF_FUEL)]
    assert len(set(scores)) == 1
    r_ = CFG["reward"]
    ceiling = r_["failure_max"] + r_["success_position"] + r_["success_fuel"]
    assert 0.0 <= scores[0] <= ceiling


def test_timeout_scores_zero_even_in_a_near_perfect_state():
    """시도하지 않으면 0점이다.

    목표 바로 위에서 거의 멈춘 상태라도 착륙하지 않았으면 0점이다.
    맴도는 쪽이 착륙을 시도하다 실패하는 쪽보다 높은 점수를 받으면
    최적 전략이 '절대 착륙하지 않기'가 되기 때문이다. 실제로
    landing-easy 의 고도를 낮추자 1.0g 정지 추력 정책이 정확히 그
    상태로 수렴했다.
    """
    almost = at(x=0.0, y=26.0, vx=0.0, vy=-0.5)
    assert terminal_reward(Outcome.TIMEOUT, almost, TARGET, CFG,
                           fuel_frac=1.0) == 0.0


def test_catch_profile_rewards_slow_contact_much_more_steeply():
    catch_cfg = build_config(PRESETS["catch"])
    target = (0.0, catch_cfg["catch"]["y_arm"])

    def score(speed):
        return terminal_reward(
            Outcome.SUCCESS,
            at(x=0.0, y=target[1], vy=-speed, step=0),
            target, catch_cfg, fuel_frac=1.0)

    assert score(0.0) - score(1.0) > score(3.0) - score(4.0)


def test_shaping_reads_gamma_from_config():
    """shaping 이 cfg 의 감가율을 실제로 읽는지 확인한다.

    gamma 를 1.0으로 하드코딩한 구현도 기본 설정에서는 텔레스코핑
    테스트를 전부 통과한다. 다른 값을 넣어야 비로소 드러난다.
    """
    cfg = build_config({"reward": {"shaping_gamma": 0.5}})
    assert shaping(-2.0, -1.0, cfg) == pytest.approx(0.5 * -1.0 - (-2.0))
