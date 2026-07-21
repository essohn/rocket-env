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
POTENTIAL_DIST_SCALE = 300.0
POTENTIAL_SPEED_SCALE = 50.0

# 판정 지점에 실제로 도달한 실패. 부분 점수를 받는다.
_CONTACT_FAILURES = frozenset({
    Outcome.CRASH, Outcome.MISSED, Outcome.OUT_OF_FUEL,
})


def _wrap_angle(theta: float) -> float:
    """각도를 (-π, π] 로 접는다.

    물리는 θ 를 감지 않으므로 여러 바퀴 돈 상태에서 |θ| 가 계속 커진다.
    반면 관찰은 sin/cos 라 감김 횟수를 볼 수 없다. 보상만 그것에 의존하면
    관찰로 구분할 수 없는 두 상태가 다른 값을 가져 비마르코프가 된다.
    """
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def _closeness(value: float, threshold: float) -> float:
    """임계값의 2배 지점에서 0이 되는 선형 근접도.

    임계값 자체에서 0.5 다. 부분 점수를 주되 '거의 성공'과 '한참 멀었음'을
    뚜렷이 가르는 기울기를 만든다.
    """
    return min(max(1.0 - value / (2.0 * threshold), 0.0), 1.0)


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
    """종료 시 한 번 지급되는 보상."""
    r = cfg["reward"]
    s = cfg["success"]

    if outcome == Outcome.SUCCESS:
        speed = math.hypot(state.vx, state.vy)
        dx = abs(state.x - target[0])
        return (
            r["success_base"]
            + r["w_speed"] * math.exp(-speed / r["v_ref"])
            + r["w_position"] * max(0.0, 1.0 - dx / s["zone_r"])
            + r["w_attitude"] * max(
                0.0, 1.0 - abs(state.theta) / math.radians(s["theta_max_deg"]))
            + r["w_fuel"] * fuel_frac
            + r["w_time"] * (1.0 - state.step / cfg["max_steps"])
        )

    if outcome == Outcome.TIMEOUT:
        # 시간이 다 되도록 판정 지점에 가지 않았다면 시도 자체를 하지 않은
        # 것이다. 목표 근처에서 맴도는 쪽이 착륙을 시도하다 실패하는 쪽보다
        # 높은 점수를 받으면, 최적 전략은 "절대 착륙하지 않기"가 된다.
        return 0.0

    if outcome in _CONTACT_FAILURES:
        speed = math.hypot(state.vx, state.vy)
        # 성공은 다섯 조건을 모두 만족해야 한다. 실패 점수도 가장 약한
        # 고리가 정한다 — 평균을 쓰면 "속도만 빼고 완벽"이 높은 점수를
        # 받아, 중력이 공짜로 만들어 주는 자유낙하 고득점이 되살아난다.
        return r["failure_max"] * min(
            _closeness(abs(state.x - target[0]), s["zone_r"]),
            _closeness(abs(state.y - target[1]), s["zone_r"]),
            _closeness(speed, s["v_max"]),
            _closeness(abs(_wrap_angle(state.theta)),
                       math.radians(s["theta_max_deg"])),
            _closeness(abs(state.omega), math.radians(s["omega_max_deg"])),
        )

    raise ValueError(f"종료 보상을 계산할 수 없는 outcome: {outcome!r}")
