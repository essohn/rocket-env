"""rocket-env — 강화학습 수업용 로켓 착륙 / 젓가락 포획 환경.

Inspired by jiupinjia/rocket-recycling (Zhengxia Zou).
물리 모델만 참고한 독립 구현이며, 원본 코드나 에셋을 포함하지 않는다.
"""

from gymnasium.envs.registration import register

from rocket_env.config import PRESETS, build_config, validate_train_config
from rocket_env.env import RocketEnv

__all__ = ["RocketEnv", "PRESETS", "build_config", "validate_train_config"]


def _make_with_task(task: str):
    """별칭 id용 팩토리.

    kwargs로 config를 통째로 넘기면 task 지정이 덮여 사라지므로,
    사용자 config에 task만 채워 넣는 방식으로 만든다.
    """

    def factory(config: dict | None = None, **kwargs) -> RocketEnv:
        merged = dict(config or {})
        merged.setdefault("task", task)
        return RocketEnv(config=merged, **kwargs)

    return factory


_make_landing = _make_with_task("landing")
_make_catch = _make_with_task("catch")

# max_episode_steps는 None이다 — 절단은 환경이 직접 처리한다.
register(id="rocket-v0", entry_point="rocket_env.env:RocketEnv",
         max_episode_steps=None)
register(id="rocket-landing-v0", entry_point="rocket_env:_make_landing",
         max_episode_steps=None)
register(id="rocket-catch-v0", entry_point="rocket_env:_make_catch",
         max_episode_steps=None)
