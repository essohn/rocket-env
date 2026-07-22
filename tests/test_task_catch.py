"""젓가락 포획 태스크의 통과 판정.

핵심은 '하강 중 팔 높이를 가로지르는 단 한 스텝'에만 판정이 일어난다는 것이다.
"""

import math

import pytest

from rocket_env.config import PRESETS, build_config
from rocket_env.tasks import make_task
from rocket_env.types import Outcome, State

CFG = build_config(PRESETS["catch"])
TASK = make_task("catch")
Y_ARM = CFG["catch"]["y_arm"]


def at(**kw) -> State:
    base = dict(x=0.0, y=200.0, vx=0.0, vy=-1.0, theta=0.0, omega=0.0,
                phi=0.0, thrust=0.0, fuel=100.0, wind_x=0.0, step=10)
    base.update(kw)
    return State(**base)


def crossing(**kw) -> tuple[State, State]:
    """팔 높이를 아래로 가로지르는 (prev, cur) 쌍."""
    return at(y=Y_ARM + 1.0), at(y=Y_ARM - 0.1, **kw)


def test_target_is_the_tower_arm():
    assert TASK.target(CFG) == (CFG["catch"]["x_tower"], Y_ARM)


def test_far_above_the_arm_is_in_progress():
    assert TASK.evaluate(at(y=300.0), at(y=299.0), CFG) is None


def test_slow_centred_crossing_is_caught():
    prev, cur = crossing(vx=0.0, vy=-1.0)
    assert TASK.evaluate(prev, cur, CFG) == Outcome.SUCCESS


def test_fast_crossing_is_missed():
    prev, cur = crossing(vy=-CFG["success"]["v_max"] - 0.1)
    assert TASK.evaluate(prev, cur, CFG) == Outcome.MISSED


def test_offset_crossing_is_missed():
    prev, cur = crossing(x=CFG["success"]["zone_r"] + 0.1, vy=-1.0)
    assert TASK.evaluate(prev, cur, CFG) == Outcome.MISSED


def test_tilted_crossing_is_missed():
    prev, cur = crossing(vy=-1.0,
                         theta=math.radians(CFG["success"]["theta_max_deg"] + 0.1))
    assert TASK.evaluate(prev, cur, CFG) == Outcome.MISSED


def test_spinning_crossing_is_missed():
    prev, cur = crossing(vy=-1.0,
                         omega=math.radians(CFG["success"]["omega_max_deg"] + 0.1))
    assert TASK.evaluate(prev, cur, CFG) == Outcome.MISSED


def test_crossing_upward_is_not_judged():
    """아래에서 위로 지나가는 것은 포획 시도가 아니다."""
    prev, cur = at(y=Y_ARM - 1.0, vy=+5.0), at(y=Y_ARM + 0.1, vy=+5.0)
    assert TASK.evaluate(prev, cur, CFG) is None


def test_hovering_below_the_arm_does_not_retrigger():
    """한 번 지나간 뒤 팔 아래에서 맴돌아도 다시 판정되지 않는다."""
    assert TASK.evaluate(at(y=Y_ARM - 5.0), at(y=Y_ARM - 6.0), CFG) is None


def test_crossing_exactly_at_arm_height_is_judged():
    """`y_arm >= cur.y` 는 등호를 포함한다 — 정확히 팔 높이에 닿아도 판정한다.

    다른 테스트는 전부 Y_ARM - 0.1 을 쓰므로 등호를 빼도(`>` 로 바꿔도)
    모두 통과한다. 이 테스트가 그 한 칸을 고정한다.
    """
    prev, cur = at(y=Y_ARM + 1.0), at(y=Y_ARM, vy=-1.0)
    assert TASK.evaluate(prev, cur, CFG) == Outcome.SUCCESS


def test_starting_exactly_at_arm_height_is_not_a_crossing():
    """`prev.y > y_arm` 은 strict 다 — 팔 높이에서 출발하면 통과가 아니다.

    등호를 허용하면 팔 높이 부근에서 맴도는 로켓이 매 스텝 재판정된다.
    """
    prev, cur = at(y=Y_ARM, vy=-1.0), at(y=Y_ARM - 0.1, vy=-1.0)
    assert TASK.evaluate(prev, cur, CFG) is None


def test_step_that_reverses_to_upward_is_not_judged():
    """`cur.vy < 0` 가드가 실제로 지키는 유일한 경우.

    한 스텝의 순 변위는 아래쪽인데 끝 속도가 위로 뒤집힌 상태 — 팔 높이
    부근에서 거의 멈췄다가 반등하는 순간이다. test_crossing_upward_is_not_judged
    는 y 순서 조건만으로 이미 걸러져서 이 가드를 전혀 시험하지 못한다.
    """
    prev, cur = at(y=Y_ARM + 0.05, vy=-0.3), at(y=Y_ARM - 0.01, vy=+0.1)
    assert TASK.evaluate(prev, cur, CFG) is None


def test_reaching_the_ground_without_crossing_crashes():
    from rocket_env.tasks.base import GROUND_Y
    assert TASK.evaluate(at(y=GROUND_Y + 1.0),
                         at(y=GROUND_Y - 0.1), CFG) == Outcome.CRASH


def test_offset_tower_shifts_the_capture_zone():
    cfg = build_config({**PRESETS["catch"],
                        "catch": {"x_tower": 100.0, "y_arm": Y_ARM}})
    prev, cur = crossing(x=100.0, vy=-1.0)
    assert TASK.evaluate(prev, cur, cfg) == Outcome.SUCCESS
    prev, cur = crossing(x=0.0, vy=-1.0)
    assert TASK.evaluate(prev, cur, cfg) == Outcome.MISSED
