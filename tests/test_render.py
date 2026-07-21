"""렌더링 smoke 테스트.

픽셀 값을 검증하지는 않는다. 크래시 없이 올바른 형태의 배열이 나오는지,
두 태스크과 모든 종료 상태에서 그려지는지만 본다.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from rocket_env.config import PRESETS, build_config  # noqa: E402
from rocket_env.env import RocketEnv  # noqa: E402
from rocket_env.render import HEIGHT, WIDTH, Renderer  # noqa: E402
from rocket_env.tasks.base import sample_initial_state  # noqa: E402
from rocket_env.types import Outcome  # noqa: E402


def a_state(cfg):
    return sample_initial_state(np.random.default_rng(0), cfg)


@pytest.mark.parametrize("preset", ["landing-normal", "catch-normal"])
def test_rgb_array_has_the_expected_shape_and_dtype(preset):
    env = RocketEnv(config=PRESETS[preset], render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (HEIGHT, WIDTH, 3)
    assert frame.dtype == np.uint8
    env.close()


@pytest.mark.parametrize("preset", ["landing-normal", "catch-normal"])
def test_rendering_survives_a_whole_episode(preset):
    env = RocketEnv(config=PRESETS[preset], render_mode="rgb_array")
    env.reset(seed=0)
    for _ in range(200):
        _, _, terminated, truncated, _ = env.step(1)
        assert env.render().shape == (HEIGHT, WIDTH, 3)
        if terminated or truncated:
            break
    env.close()


@pytest.mark.parametrize("outcome", [
    Outcome.IN_PROGRESS, Outcome.SUCCESS, Outcome.CRASH,
    Outcome.MISSED, Outcome.TIMEOUT, Outcome.OUT_OF_FUEL,
])
def test_every_outcome_banner_draws(outcome):
    cfg = build_config(PRESETS["landing-normal"])
    renderer = Renderer(cfg, "rgb_array")
    frame = renderer.draw(a_state(cfg), (0.0, 25.0), outcome)
    assert frame.shape == (HEIGHT, WIDTH, 3)
    renderer.close()


def test_render_returns_none_when_render_mode_is_none():
    env = RocketEnv()
    env.reset(seed=0)
    assert env.render() is None
    env.close()


def test_reset_clears_the_trail():
    cfg = build_config(PRESETS["landing-normal"])
    renderer = Renderer(cfg, "rgb_array")
    state = a_state(cfg)
    for _ in range(5):
        renderer.draw(state, (0.0, 25.0), Outcome.IN_PROGRESS)
    assert len(renderer.trail) == 5
    renderer.reset()
    assert renderer.trail == []
    renderer.close()
