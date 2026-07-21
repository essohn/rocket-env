"""태스크 팩토리."""

from rocket_env.tasks.base import Task
from rocket_env.tasks.landing import LandingTask

__all__ = ["Task", "LandingTask", "make_task"]

_REGISTRY = {"landing": LandingTask}


def make_task(name: str) -> Task:
    if name not in _REGISTRY:
        raise ValueError(
            f"알 수 없는 task: {name!r} (가능한 값: {sorted(_REGISTRY)})"
        )
    return _REGISTRY[name]()
