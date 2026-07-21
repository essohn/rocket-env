"""강체 로켓 동역학.

이 모듈은 태스크와 보상을 전혀 모른다. 물리는 고정이고 보상은 설계 선택이라는
구분을 파일 경계로 드러내기 위한 것이다.

모델: 얇은 막대 강체 + 짐벌 노즐 + 속도에 비례하는 항력.
"""

import math
from dataclasses import replace

from rocket_env.types import State

# --- 물리 상수 (config로 노출하지 않는 잠금 값) ---
G = 9.8                                    # 중력가속도 (m/s^2)
DT = 0.05                                  # 시뮬레이션 시간 간격 (s)
ROCKET_HEIGHT = 50.0                       # 기체 길이 (m)
MOMENT_OF_INERTIA = ROCKET_HEIGHT**2 / 12.0  # 얇은 막대의 관성모멘트 (단위질량)
PHI_MAX = math.radians(20.0)               # 노즐 짐벌 한계 (rad)

# 항력 계수는 종단속도를 설계값으로 두고 역산한다. 무동력 낙하가 평형에
# 이르면 DRAG_RHO * v = G 이므로 DRAG_RHO = G / v_term.
# 계수 자체는 물리 법칙이 정해주지 않는 설계 선택이므로, 의미가 바로 읽히는
# 양(종단속도)으로 고르는 편이 학생에게도 검증하기 쉽다.
TERMINAL_VELOCITY = 49.5                   # 무동력 낙하 종단속도 (m/s)
DRAG_RHO = G / TERMINAL_VELOCITY

# --- 세계 경계 ---
WORLD_X_MIN, WORLD_X_MAX = -300.0, 300.0
WORLD_Y_MIN, WORLD_Y_MAX = 0.0, 570.0

# --- 행동 테이블 ---
# 추력 0을 포함하는 것이 핵심이다. 엔진을 끌 수 있어야 연료 절약이 전략이 된다.
THRUST_LEVELS = (0.0, 0.2 * G, 1.0 * G, 2.0 * G)
NOZZLE_RATES = (-math.radians(30.0), 0.0, math.radians(30.0))

# 인덱스 = thrust_idx * 3 + nozzle_idx
ACTION_TABLE: tuple[tuple[float, float], ...] = tuple(
    (f, rate) for f in THRUST_LEVELS for rate in NOZZLE_RATES
)


def fuel_cost(thrust: float) -> float:
    """이번 스텝의 연료 소모량. 1단위 = 1G 추력으로 1초 분사."""
    return (thrust / G) * DT


def integrate(state: State, thrust: float, nozzle_rate: float,
              wind_x: float) -> State:
    """한 스텝 적분한 새 State를 반환한다.

    추력을 기체 기준 두 성분으로 나눈 뒤 세계 좌표로 회전시킨다.
    접선 성분만 토크를 만들고, 축 성분만 기체를 밀어올린다.

    항력은 지면 기준 속도가 아니라 **공기 기준 상대속도**에 비례한다.
    바람이 새로운 힘 항이 아니라 기존 항력 항의 수정으로 들어가는 이유다.
    """
    theta, phi = state.theta, state.phi

    thrust_tangential = -thrust * math.sin(phi)   # 옆 방향 성분 → 토크
    thrust_axial = thrust * math.cos(phi)         # 기체 축 방향 성분 → 추진

    fx = thrust_tangential * math.cos(theta) - thrust_axial * math.sin(theta)
    fy = thrust_tangential * math.sin(theta) + thrust_axial * math.cos(theta)

    ax = fx - DRAG_RHO * (state.vx - wind_x)
    ay = fy - G - DRAG_RHO * state.vy
    alpha = thrust_tangential * (ROCKET_HEIGHT / 2.0) / MOMENT_OF_INERTIA

    return replace(
        state,
        x=state.x + state.vx * DT + 0.5 * ax * DT**2,
        y=state.y + state.vy * DT + 0.5 * ay * DT**2,
        vx=state.vx + ax * DT,
        vy=state.vy + ay * DT,
        theta=state.theta + state.omega * DT + 0.5 * alpha * DT**2,
        omega=state.omega + alpha * DT,
        phi=min(max(phi + nozzle_rate * DT, -PHI_MAX), PHI_MAX),
        thrust=thrust,
        step=state.step + 1,
    )
