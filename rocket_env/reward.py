"""보상 설계.

두 부분으로 나뉜다.

1. 스텝 보상 — 잠재함수 기반 shaping(PBRS).
   shaping_gamma=1.0이면 shaping 총합이 정확히 Φ(s_T) - Φ(s_0)로 접히므로,
   에피소드가 길다고 점수가 쌓이지 않는다.
   연료 패널티(cfg["reward"]["fuel_penalty"])는 실제 소모량을 아는
   env.step()에서 더한다. 이 모듈은 소모량을 모른다.

2. 종료 보상 — 성공은 기본점 + 품질 보너스, 실패는 성공 조건까지의 근접도.
   실패 보상에 시간 항이 전혀 없다는 점이 중요하다. 원본 환경은 실패에도
   남은 스텝 수를 곱해서 '빨리 자폭하기'가 고득점 전략이 되었다.

   예전에는 실패 점수를 목표까지의 직선 거리로 매겼다(1 - d_final/d_initial).
   목표가 지면에 있어 중력이 거리를 공짜로 닫아 주는 탓에, 자유낙하만으로도
   실패 점수의 80~89%를 받았다 — 무행동 정책이 학습된 DQN 보다 높은 점수를
   받는 근본 원인이었다. 지금은 성공 판정에 쓰는 다섯 조건(위치·속도·자세·
   각속도) 각각에 대한 근접도 중 가장 나쁜 것으로 채점한다.
"""

import math

from rocket_env.types import Outcome, State

# 잠재함수의 정규화 상수. config에서 파생하지 않는 환경 상수다.
POTENTIAL_DIST_SCALE = 900.0
POTENTIAL_SPEED_SCALE = 200.0

# 판정 지점에 실제로 도달한 실패. 부분 점수를 받는다.
_CONTACT_FAILURES = frozenset({
    Outcome.CRASH, Outcome.MISSED, Outcome.OUT_OF_FUEL,
})


def _wrap_angle(theta: float) -> float:
    """각도를 (-π, π] 로 접는다.

    물리는 θ 를 감지 않으므로 여러 바퀴 돈 상태에서 |θ| 가 계속 커진다.
    반면 관찰은 sin/cos 라 감김 횟수를 볼 수 없다. 보상만 그것에 의존하면
    관찰로 구분할 수 없는 두 상태가 다른 값을 가져 비마르코프가 된다.

    이미 (-π, π] 안에 있는 값은 그대로 돌려준다. `(theta + π) % 2π - π`는
    수학적으로 항등식이지만, 부동소수점에서는 π를 더했다 빼는 과정에서
    반올림 오차가 생겨 경계값(예: 정확히 theta_max_deg)에서 '같다'가
    '근소하게 작다'로 바뀔 수 있다. 실제로 감아야 하는 경우(여러 바퀴 돈
    상태)에만 모듈로 연산을 타도록 분기해 흔한 경로의 정밀도를 지킨다.
    """
    if -math.pi < theta <= math.pi:
        return theta
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def _reach(value: float, threshold: float) -> float:
    """임계값 안에서 1.0, 그 밖에서는 부드럽게 지수 감쇠하는 도달도(0 바닥 없음).

    임계값에서 정확히 1.0이라 실패↔성공 경계가 연속이다. 임계값을 넘으면
    선형으로 끊지 않고 exp(-(초과분)/임계값)으로 **전 구간 단조 감소**한다.

    예전엔 임계값 2배 지점(예: v_max 15의 2배인 30 m/s)에서 0으로 끊겨, 그
    너머 모든 크래시가 똑같이 0점이었다 — 60 m/s 든 250 m/s 든 구분이 없어
    "더 느리게 부딪히면 낫다"는 gradient 가 사라졌다. 이 평평한 0 바닥이
    제동 학습을 막는 원인이었다(실측: gradual 화만으로 접지속도 76→49 m/s).
    이제 접지가 느릴수록·중심에 가까울수록·수직일수록 항상 점수가 높다.
    """
    if value <= threshold:
        return 1.0
    return math.exp(-(value - threshold) / threshold)


def potential(state: State, target: tuple[float, float], cfg: dict) -> float:
    """Φ(s). 목표에 가깝고, 수직이고, 느릴수록 0에 가깝다. 항상 0 이하."""
    r = cfg["reward"]
    dx = abs(state.x - target[0]) / POTENTIAL_DIST_SCALE
    dy = abs(state.y - target[1]) / POTENTIAL_DIST_SCALE
    tilt = abs(_wrap_angle(state.theta)) / (math.pi / 2.0)
    speed = math.hypot(state.vx, state.vy) / POTENTIAL_SPEED_SCALE
    return -(r["shaping_w_dist"] * (dx + dy)
             + r["shaping_w_attitude"] * tilt
             + r["shaping_w_speed"] * speed)


def shaping(prev_potential: float, cur_potential: float, cfg: dict) -> float:
    """F = γ·Φ(s') - Φ(s)."""
    return cfg["reward"]["shaping_gamma"] * cur_potential - prev_potential


def terminal_reward(outcome: str, state: State, target: tuple[float, float],
                    cfg: dict, fuel_frac: float) -> float:
    """종료 시 한 번 지급되는 보상. 전 구간에서 gradual(연속·단조)하게 설계한다.

      - 실패(접촉): closeness × (failure_max + 위치 + 연료). closeness 는
        5개 축 도달도의 min 이며, 각 도달도가 임계값 밖에서 0으로 끊기지 않고
        지수 감쇠한다([[_reach]]). 그래서 60 m/s 크래시가 250 m/s 크래시보다
        확실히 높은 점수를 받아 "더 느리게 부딪히면 낫다"는 gradient 가 살아
        있다 — 제동 학습의 핵심.
      - 성공: 같은 바닥(failure_max + 위치 + 연료)에서 시작해, 접지 속도가
        낮을수록(success_soft·가장 중요한 축) 위로 더 쌓는다.
    경계(모든 축이 임계값)에서 실패=failure_max+위치+연료, 성공=거기에 거의
    0인 속도 보너스만 더해져 점프가 없다. 예전엔 (1) 실패가 임계값 2배 밖에서
    일괄 0점이라 제동 gradient 가 없었고 (2) 경계에서 실패40→성공80 점프가
    있었다 — 둘 다 이 설계로 사라진다.
    """
    r = cfg["reward"]
    s = cfg["success"]

    if outcome == Outcome.TIMEOUT:
        # 시간이 다 되도록 판정 지점에 가지 않았다면 시도 자체를 하지 않은
        # 것이다. 목표 근처에서 맴도는 쪽이 착륙을 시도하다 실패하는 쪽보다
        # 높은 점수를 받으면, 최적 전략은 "절대 착륙하지 않기"가 된다.
        return 0.0

    speed = math.hypot(state.vx, state.vy)
    dx = abs(state.x - target[0])

    if outcome == Outcome.SUCCESS:
        centered = max(0.0, 1.0 - dx / s["zone_r"])
        return (
            r["failure_max"]
            + r["success_soft"] * math.exp(-speed / r["soft_v_ref"])
            + r["success_position"] * centered
            + r["success_fuel"] * fuel_frac
        )

    if outcome in _CONTACT_FAILURES:
        # 성공은 네 조건을 모두 만족해야 하므로, 실패 점수도 가장 약한
        # 고리(closeness)가 정한다 — 평균을 쓰면 "속도만 빼고 완벽"이 높은
        # 점수를 받아, 중력이 공짜로 만들어 주는 자유낙하 고득점이 되살아난다.
        closeness = min(
            _reach(dx, s["zone_r"]),
            _reach(abs(state.y - target[1]), s["zone_r"]),
            _reach(speed, s["v_max"]),
            _reach(abs(_wrap_angle(state.theta)),
                   math.radians(s["theta_max_deg"])),
            _reach(abs(state.omega), math.radians(s["omega_max_deg"])),
        )
        # 성공과 경계에서 연속이 되도록 위치·연료 보너스도 함께 준다. closeness
        # (speed 도달도 포함)로 스케일하므로 빠른 크래시는 전체가 작아져
        # 자유낙하 고득점 악용은 여전히 막힌다. 경계(closeness=1)에서
        # failure_max + 위치 + 연료 = 성공식의 바닥과 일치한다.
        centered = max(0.0, 1.0 - dx / s["zone_r"])
        return closeness * (r["failure_max"]
                            + r["success_position"] * centered
                            + r["success_fuel"] * fuel_frac)

    raise ValueError(f"종료 보상을 계산할 수 없는 outcome: {outcome!r}")
