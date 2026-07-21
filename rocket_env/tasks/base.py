"""태스크 인터페이스와 두 태스크가 공유하는 로직.

태스크는 '언제 끝나고 무엇이 성공인가'만 정의한다. 물리도 보상도 모른다.
"""

import math
from typing import Protocol

import numpy as np

from rocket_env.physics import (
    ROCKET_HEIGHT,
    WORLD_X_MAX,
    WORLD_X_MIN,
    WORLD_Y_MAX,
)
from rocket_env.types import State

GROUND_Y = ROCKET_HEIGHT / 2.0     # 로켓 중심이 지면에 닿는 높이
CEILING_Y = WORLD_Y_MAX - ROCKET_HEIGHT / 2.0


class Task(Protocol):
    name: str

    def initial_state(self, rng: np.random.Generator, cfg: dict) -> State:
        """에피소드 시작 상태를 샘플링한다."""
        ...

    def target(self, cfg: dict) -> tuple[float, float]:
        """관찰의 기준이 되는 목표점 (x, y)."""
        ...

    def evaluate(self, prev: State, cur: State, cfg: dict) -> str | None:
        """종료 사유를 반환한다. 아직 진행 중이면 None."""
        ...


def sample_initial_state(rng: np.random.Generator, cfg: dict) -> State:
    """두 태스크가 동일한 초기 조건 분포를 쓴다."""
    init = cfg["init"]
    capacity = cfg["fuel"]["capacity"]
    return State(
        x=float(rng.uniform(*init["x_range"])),
        y=float(init["y"]),
        vx=0.0,
        vy=float(rng.uniform(*init["vy_range"])),
        theta=math.radians(float(rng.uniform(*init["theta_range_deg"]))),
        omega=0.0,
        phi=0.0,
        thrust=0.0,
        fuel=math.inf if capacity is None else float(capacity),
        wind_x=0.0,
        step=0,
    )


def out_of_bounds(state: State) -> bool:
    """세계 밖으로 벗어났는가."""
    return (state.y >= CEILING_Y
            or state.x <= WORLD_X_MIN
            or state.x >= WORLD_X_MAX)


def within_thresholds(state: State, cfg: dict, dx: float) -> bool:
    """속도·자세·각속도·수평 오차가 모두 성공 임계 안인가."""
    s = cfg["success"]
    speed = math.hypot(state.vx, state.vy)
    return (abs(dx) < s["zone_r"]
            and speed < s["v_max"]
            and abs(state.theta) < math.radians(s["theta_max_deg"])
            and abs(state.omega) < math.radians(s["omega_max_deg"]))
