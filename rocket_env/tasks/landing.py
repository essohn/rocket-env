"""지면 착륙 태스크."""

import numpy as np

from rocket_env.tasks.base import (
    GROUND_Y,
    out_of_bounds,
    sample_initial_state,
    within_thresholds,
)
from rocket_env.types import Outcome, State


class LandingTask:
    name = "landing"

    def initial_state(self, rng: np.random.Generator, cfg: dict) -> State:
        return sample_initial_state(rng, cfg)

    def target(self, cfg: dict) -> tuple[float, float]:
        return (0.0, GROUND_Y)

    def evaluate(self, prev: State, cur: State, cfg: dict) -> str | None:
        if out_of_bounds(cur):
            return Outcome.CRASH
        if cur.y <= GROUND_Y:
            ok = within_thresholds(cur, cfg, dx=cur.x)
            return Outcome.SUCCESS if ok else Outcome.CRASH
        return None
