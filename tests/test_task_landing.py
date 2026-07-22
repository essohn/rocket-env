"""지면 착륙 태스크의 성공/실패 판정."""

import math

import numpy as np
import pytest

from rocket_env.config import PRESETS, build_config
from rocket_env.physics import ROCKET_HEIGHT, WORLD_Y_MAX
from rocket_env.tasks import make_task
from rocket_env.types import Outcome, State

CFG = build_config(PRESETS["landing-gust"])
TASK = make_task("landing")
GROUND = ROCKET_HEIGHT / 2.0


def at(**kw) -> State:
    base = dict(x=0.0, y=200.0, vx=0.0, vy=0.0, theta=0.0, omega=0.0,
                phi=0.0, thrust=0.0, fuel=100.0, wind_x=0.0, step=10)
    base.update(kw)
    return State(**base)


def test_target_is_pad_centre_at_half_rocket_height():
    assert TASK.target(CFG) == (0.0, GROUND)


def test_airborne_state_is_in_progress():
    assert TASK.evaluate(at(y=201.0), at(y=200.0), CFG) is None


def test_perfect_touchdown_succeeds():
    cur = at(y=GROUND - 0.1, vy=-1.0)
    assert TASK.evaluate(at(y=GROUND + 5.0), cur, CFG) == Outcome.SUCCESS


def test_touchdown_too_fast_crashes():
    cur = at(y=GROUND - 0.1, vy=-CFG["success"]["v_max"] - 0.1)
    assert TASK.evaluate(at(y=GROUND + 5.0), cur, CFG) == Outcome.CRASH


def test_touchdown_outside_pad_crashes():
    cur = at(y=GROUND - 0.1, x=CFG["success"]["zone_r"] + 0.1, vy=-1.0)
    assert TASK.evaluate(at(y=GROUND + 5.0), cur, CFG) == Outcome.CRASH


def test_touchdown_tilted_crashes():
    cur = at(y=GROUND - 0.1, vy=-1.0,
             theta=math.radians(CFG["success"]["theta_max_deg"] + 0.1))
    assert TASK.evaluate(at(y=GROUND + 5.0), cur, CFG) == Outcome.CRASH


def test_touchdown_spinning_crashes():
    cur = at(y=GROUND - 0.1, vy=-1.0,
             omega=math.radians(CFG["success"]["omega_max_deg"] + 0.1))
    assert TASK.evaluate(at(y=GROUND + 5.0), cur, CFG) == Outcome.CRASH


def test_flying_off_the_top_crashes():
    cur = at(y=WORLD_Y_MAX - GROUND + 1.0)
    assert TASK.evaluate(at(y=500.0), cur, CFG) == Outcome.CRASH


def test_flying_off_the_side_crashes():
    cur = at(x=901.0)
    assert TASK.evaluate(at(x=899.0), cur, CFG) == Outcome.CRASH


def test_touchdown_exactly_at_speed_threshold_crashes():
    """임계값 비교는 strict `<` 다 — 정확히 임계값이면 실패한다.

    ±0.1로만 찔러보는 테스트는 `<` 와 `<=` 를 구별하지 못한다. 성적을
    만드는 코드에서 이 한 칸이 '겨우 통과'와 '겨우 실패'를 가른다.
    """
    cur = at(y=GROUND - 0.1, vy=-CFG["success"]["v_max"])
    assert TASK.evaluate(at(y=GROUND + 5.0), cur, CFG) == Outcome.CRASH


def test_touchdown_exactly_at_pad_edge_crashes():
    cur = at(y=GROUND - 0.1, x=CFG["success"]["zone_r"], vy=-1.0)
    assert TASK.evaluate(at(y=GROUND + 5.0), cur, CFG) == Outcome.CRASH


def test_touchdown_exactly_at_tilt_threshold_crashes():
    cur = at(y=GROUND - 0.1, vy=-1.0,
             theta=math.radians(CFG["success"]["theta_max_deg"]))
    assert TASK.evaluate(at(y=GROUND + 5.0), cur, CFG) == Outcome.CRASH


def test_touchdown_exactly_at_spin_threshold_crashes():
    cur = at(y=GROUND - 0.1, vy=-1.0,
             omega=math.radians(CFG["success"]["omega_max_deg"]))
    assert TASK.evaluate(at(y=GROUND + 5.0), cur, CFG) == Outcome.CRASH


def test_theta_zero_and_two_pi_give_the_same_verdict():
    """관찰이 sin/cos 라 θ=0 과 θ=2π 를 구별할 수 없다 — 판정도 같아야 한다.

    Task 16 이전에는 성공 판정이 `_wrap_angle` 을 쓰지 않아 두 상태의
    outcome 이 달랐다(성공 vs 실패). 물리는 θ 를 감지 않으므로 여러 바퀴
    돌고 착지하면 |θ| 가 그대로 누적되는데, 관찰로는 절대 구별할 수 없는
    두 상태가 다른 채점을 받는 것은 비마르코프적이다.
    """
    upright = at(y=GROUND - 0.1, vy=-1.0, theta=0.0)
    spun = at(y=GROUND - 0.1, vy=-1.0, theta=2.0 * math.pi)
    prev = at(y=GROUND + 5.0)
    assert (TASK.evaluate(prev, upright, CFG)
            == TASK.evaluate(prev, spun, CFG)
            == Outcome.SUCCESS)


def test_ground_contact_triggers_exactly_at_ground_level():
    """접지 판정만 `<=` 다 — 정확히 지면 높이에 닿으면 접지로 본다."""
    cur = at(y=GROUND, vy=-1.0)
    assert TASK.evaluate(at(y=GROUND + 5.0), cur, CFG) == Outcome.SUCCESS


def test_initial_state_respects_config_ranges():
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = TASK.initial_state(rng, CFG)
        assert CFG["init"]["x_range"][0] <= s.x <= CFG["init"]["x_range"][1]
        assert CFG["init"]["vy_range"][0] <= s.vy <= CFG["init"]["vy_range"][1]
        assert s.y == CFG["init"]["y"]
        assert abs(s.theta) <= math.radians(CFG["init"]["theta_range_deg"][1])
        assert s.fuel == CFG["fuel"]["capacity"]
        assert s.step == 0


def test_unlimited_fuel_config_yields_infinite_fuel():
    cfg = build_config(PRESETS["landing-basic"])
    s = TASK.initial_state(np.random.default_rng(0), cfg)
    assert math.isinf(s.fuel)


def test_make_task_rejects_unknown_name():
    with pytest.raises(ValueError, match="hover"):
        make_task("hover")
