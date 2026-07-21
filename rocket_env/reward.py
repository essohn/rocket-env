"""보상 설계.

두 부분으로 나뉜다.

1. 스텝 보상 — 잠재함수 기반 shaping(PBRS).
   shaping_gamma=1.0이면 shaping 총합이 정확히 Φ(s_T) - Φ(s_0)로 접히므로,
   에피소드가 길다고 점수가 쌓이지 않는다.
   연료 패널티(cfg["reward"]["fuel_penalty"])는 실제 소모량을 아는
   env.step()에서 더한다. 이 모듈은 소모량을 모른다.

2. 종료 보상 — 성공은 기본점 + 품질 보너스, 실패는 목표를 향한 '진행도'.
   실패 보상에 시간 항이 전혀 없다는 점이 중요하다. 원본 환경은 실패에도
   남은 스텝 수를 곱해서 '빨리 자폭하기'가 고득점 전략이 되었다.
"""

import math

from rocket_env.types import Outcome, State

# 잠재함수의 거리 정규화 상수. config에서 파생하지 않는 환경 상수다.
POTENTIAL_DIST_SCALE = 300.0

_FAILURE_OUTCOMES = frozenset({
    Outcome.CRASH, Outcome.MISSED, Outcome.TIMEOUT, Outcome.OUT_OF_FUEL,
})


def distance_to_target(state: State, target: tuple[float, float]) -> float:
    return math.hypot(state.x - target[0], state.y - target[1])


def potential(state: State, target: tuple[float, float], cfg: dict) -> float:
    """Φ(s). 목표에 가깝고 수직일수록 0에 가깝고, 항상 0 이하다."""
    r = cfg["reward"]
    dx = abs(state.x - target[0]) / POTENTIAL_DIST_SCALE
    dy = abs(state.y - target[1]) / POTENTIAL_DIST_SCALE
    tilt = abs(state.theta) / (math.pi / 2.0)
    return -(r["shaping_w_dist"] * (dx + dy) + r["shaping_w_attitude"] * tilt)


def shaping(prev_potential: float, cur_potential: float, cfg: dict) -> float:
    """F = γ·Φ(s') - Φ(s)."""
    return cfg["reward"]["shaping_gamma"] * cur_potential - prev_potential


def terminal_reward(outcome: str, state: State, target: tuple[float, float],
                    cfg: dict, d_initial: float, fuel_frac: float) -> float:
    """종료 시 한 번 지급되는 보상."""
    r = cfg["reward"]

    if outcome == Outcome.SUCCESS:
        s = cfg["success"]
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

    if outcome in _FAILURE_OUTCOMES:
        d_final = distance_to_target(state, target)
        if d_initial <= 0.0:
            # 출발점이 목표와 정확히 겹치면 '진행도'가 정의되지 않는다.
            return 0.0
        progress = 1.0 - d_final / d_initial
        return r["failure_max"] * min(max(progress, 0.0), 1.0)

    raise ValueError(f"종료 보상을 계산할 수 없는 outcome: {outcome!r}")
