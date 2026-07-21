"""젓가락(타워 팔) 포획 태스크.

실제 Mechazilla처럼, 하강 중 팔 높이를 충분히 느리고 바르게 통과하는
'단 한 순간'에만 포획이 일어난다. 놓치면 즉시 종료된다 — 재시도는 없다.
"""

import numpy as np

from rocket_env.tasks.base import (
    GROUND_Y,
    out_of_bounds,
    sample_initial_state,
    within_thresholds,
)
from rocket_env.types import Outcome, State


class CatchTask:
    name = "catch"

    def initial_state(self, rng: np.random.Generator, cfg: dict) -> State:
        return sample_initial_state(rng, cfg)

    def target(self, cfg: dict) -> tuple[float, float]:
        return (cfg["catch"]["x_tower"], cfg["catch"]["y_arm"])

    def evaluate(self, prev: State, cur: State, cfg: dict) -> str | None:
        if out_of_bounds(cur):
            return Outcome.CRASH

        y_arm = cfg["catch"]["y_arm"]
        descending_through_arm = (
            prev.y > y_arm >= cur.y and cur.vy < 0.0
        )
        if descending_through_arm:
            dx = cur.x - cfg["catch"]["x_tower"]
            return (Outcome.SUCCESS if within_thresholds(cur, cfg, dx)
                    else Outcome.MISSED)

        if cur.y <= GROUND_Y:
            return Outcome.CRASH
        return None
