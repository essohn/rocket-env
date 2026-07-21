"""물리 엔진 검증.

물리는 태스크·보상과 무관한 고정 계층이므로, 여기 테스트가 깨지면
환경 전체의 동작이 바뀐 것이다.
"""

import math

import pytest

from rocket_env.physics import (
    ACTION_TABLE,
    DT,
    G,
    PHI_MAX,
    ROCKET_HEIGHT,
    TERMINAL_VELOCITY,
    fuel_cost,
    integrate,
)
from rocket_env.types import State


def make_state(**kw) -> State:
    base = dict(
        x=0.0, y=100.0, vx=0.0, vy=0.0, theta=0.0, omega=0.0,
        phi=0.0, thrust=0.0, fuel=math.inf, wind_x=0.0, step=0,
    )
    base.update(kw)
    return State(**base)


def test_action_table_has_12_entries_in_thrust_major_order():
    assert len(ACTION_TABLE) == 12
    assert ACTION_TABLE[0] == (0.0, -math.radians(30.0))
    assert ACTION_TABLE[1] == (0.0, 0.0)
    assert ACTION_TABLE[11] == (2.0 * G, math.radians(30.0))


def test_single_freefall_step_matches_hand_computation():
    s = integrate(make_state(), thrust=0.0, nozzle_rate=0.0, wind_x=0.0)
    assert s.y == pytest.approx(100.0 - 0.5 * G * DT**2)
    assert s.vy == pytest.approx(-G * DT)
    assert s.step == 1


def test_freefall_reaches_the_designed_terminal_velocity():
    """시뮬레이션이 설계값 TERMINAL_VELOCITY에 실제로 도달하는지 본다.

    DRAG_RHO를 종단속도에서 역산했으므로, 이 테스트는 계수 계산과 적분이
    서로 맞물려 돌아가는지 확인하는 독립적 검증이 된다.
    """
    s = make_state(y=100_000.0)
    for _ in range(2000):
        s = integrate(s, thrust=0.0, nozzle_rate=0.0, wind_x=0.0)
    assert s.vy == pytest.approx(-TERMINAL_VELOCITY, abs=0.01)


def test_drag_vanishes_when_moving_with_the_wind():
    """항력은 공기 기준 상대속도에 비례하므로 바람과 같은 속도면 0이다."""
    s = integrate(make_state(vx=10.0), thrust=0.0, nozzle_rate=0.0, wind_x=10.0)
    assert s.vx == pytest.approx(10.0)


def test_zero_wind_still_decelerates_horizontal_motion():
    s = integrate(make_state(vx=10.0), thrust=0.0, nozzle_rate=0.0, wind_x=0.0)
    assert s.vx < 10.0


def test_upright_full_thrust_gives_net_upward_acceleration_of_g():
    s = integrate(make_state(), thrust=2.0 * G, nozzle_rate=0.0, wind_x=0.0)
    assert s.vy == pytest.approx(G * DT)


def test_gimballed_thrust_produces_torque():
    """얇은 막대(I = H^2/12)의 H/2 지점에 접선력이 걸리면 alpha = 6*ft/H 다.

    프로덕션 코드와 다른 대수 경로로 유도했으므로 독립적인 오라클이다.
    프로덕션 수식을 그대로 재계산하면 지렛대 길이나 관성모멘트를 함께
    잘못 잡은 경우를 잡아낼 수 없다.
    """
    phi = math.radians(10.0)
    s = integrate(make_state(phi=phi), thrust=G, nozzle_rate=0.0, wind_x=0.0)
    thrust_tangential = -G * math.sin(phi)
    expected_alpha = 6.0 * thrust_tangential / ROCKET_HEIGHT
    assert s.omega == pytest.approx(expected_alpha * DT)
    assert s.omega < 0.0


def test_nozzle_angle_is_clipped_to_twenty_degrees():
    s = make_state()
    for _ in range(50):
        s = integrate(s, thrust=0.0, nozzle_rate=math.radians(30.0), wind_x=0.0)
    assert s.phi == pytest.approx(PHI_MAX)


def test_integrate_records_applied_thrust():
    s = integrate(make_state(), thrust=1.5 * G, nozzle_rate=0.0, wind_x=0.0)
    assert s.thrust == pytest.approx(1.5 * G)


def test_fuel_cost_is_one_unit_per_g_second():
    assert fuel_cost(G) == pytest.approx(DT)
    assert fuel_cost(0.0) == 0.0
    assert fuel_cost(2.0 * G) == pytest.approx(2.0 * DT)
