"""렌더링 smoke 테스트.

픽셀 값을 검증하지는 않는다. 크래시 없이 올바른 형태의 배열이 나오는지,
두 태스크와 모든 종료 상태에서 그려지는지를 본다. 다만 학생이 오해할 수
있는 두 지점 — 포획 창 폭과 렌더러 여러 개의 수명 — 은 따로 고정한다.
"""

import os
from dataclasses import replace

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from rocket_env.config import PRESETS, build_config  # noqa: E402
from rocket_env.env import RocketEnv  # noqa: E402
from rocket_env.physics import G  # noqa: E402
from rocket_env.render import HEIGHT, WIDTH, Renderer  # noqa: E402
from rocket_env.tasks.base import sample_initial_state  # noqa: E402
from rocket_env.types import Outcome  # noqa: E402


def a_state(cfg):
    return sample_initial_state(np.random.default_rng(0), cfg)


@pytest.mark.parametrize("preset", ["landing-descent", "catch"])
def test_rgb_array_has_the_expected_shape_and_dtype(preset):
    env = RocketEnv(config=PRESETS[preset], render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (HEIGHT, WIDTH, 3)
    assert frame.dtype == np.uint8
    env.close()


@pytest.mark.parametrize("preset", ["landing-descent", "catch"])
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
    cfg = build_config(PRESETS["landing-descent"])
    renderer = Renderer(cfg, "rgb_array")
    frame = renderer.draw(a_state(cfg), (0.0, 25.0), outcome)
    assert frame.shape == (HEIGHT, WIDTH, 3)
    renderer.close()


def test_catch_capture_window_matches_the_success_threshold():
    """그려지는 포획 창은 실제 판정 범위와 정확히 같아야 한다.

    팔 구조물은 보이도록 넓게 그리지만, 판정 범위를 따로 표시하지 않으면
    학생은 팔 안쪽으로 지나갔는데 MISSED 가 뜨는 이유를 알 수 없다.
    """
    cfg = build_config(PRESETS["catch"])
    renderer = Renderer(cfg, "rgb_array")
    _, _, arm_half, window_half = renderer._catch_geometry((0.0, 80.0))
    assert window_half == cfg["success"]["zone_r"]
    assert arm_half > window_half
    renderer.close()


def test_closing_one_renderer_does_not_break_another():
    """close() 가 pygame 을 전역 종료하면 살아 있는 다른 렌더러가 깨진다."""
    cfg = build_config(PRESETS["landing-descent"])
    first = Renderer(cfg, "rgb_array")
    second = Renderer(cfg, "rgb_array")
    first.close()
    frame = second.draw(a_state(cfg), (0.0, 25.0), Outcome.IN_PROGRESS)
    assert frame.shape == (HEIGHT, WIDTH, 3)
    second.close()


def test_render_returns_none_when_render_mode_is_none():
    env = RocketEnv()
    env.reset(seed=0)
    assert env.render() is None
    env.close()


def test_reset_clears_the_trail():
    cfg = build_config(PRESETS["landing-descent"])
    renderer = Renderer(cfg, "rgb_array")
    state = a_state(cfg)
    for _ in range(5):
        renderer.draw(state, (0.0, 25.0), Outcome.IN_PROGRESS)
    assert len(renderer.trail) == 5
    renderer.reset()
    assert renderer.trail == []
    renderer.close()


def test_camera_keeps_rocket_and_target_on_screen():
    """카메라가 로켓과 목표를 항상 화면 안에 둔다."""
    cfg = build_config(PRESETS["catch"])
    renderer = Renderer(cfg, "rgb_array")
    target = (cfg["catch"]["x_tower"], cfg["catch"]["y_arm"])
    # 목표에서 멀리 떨어진 상태 — 고정 배율이었다면 화면 밖으로 나갔을 것.
    state = replace(a_state(cfg), x=-120.0, y=350.0)

    renderer.draw(state, target, Outcome.IN_PROGRESS)

    margin = 40
    rx, ry = renderer._to_px(state.x, state.y)
    tx, ty = renderer._to_px(*target)
    assert margin <= rx <= WIDTH - margin
    assert margin <= ry <= HEIGHT - margin
    assert margin <= tx <= WIDTH - margin
    assert margin <= ty <= HEIGHT - margin
    renderer.close()


def test_particles_are_emitted_only_under_thrust():
    """추력 0이면 입자가 생기지 않는다."""
    cfg = build_config(PRESETS["landing-descent"])
    renderer = Renderer(cfg, "rgb_array")

    idle = replace(a_state(cfg), thrust=0.0)
    renderer.draw(idle, (0.0, 25.0), Outcome.IN_PROGRESS)
    assert renderer._particles == []

    thrusting = replace(a_state(cfg), thrust=2.0 * G)
    renderer.draw(thrusting, (0.0, 25.0), Outcome.IN_PROGRESS)
    assert len(renderer._particles) > 0
    renderer.close()


def test_particles_are_cleared_on_reset():
    """에피소드가 바뀌면 이전 연기가 남지 않는다."""
    cfg = build_config(PRESETS["landing-descent"])
    renderer = Renderer(cfg, "rgb_array")
    thrusting = replace(a_state(cfg), thrust=2.0 * G)
    renderer.draw(thrusting, (0.0, 25.0), Outcome.IN_PROGRESS)
    assert len(renderer._particles) > 0

    renderer.reset()
    assert renderer._particles == []
    renderer.close()


def test_caught_rocket_is_drawn_hanging_below_the_arm():
    """포획 프레임에서 그려지는 로켓 중심이 팔보다 아래다.

    물리 상태는 바뀌지 않아야 한다 — 그리기 직전에만 바꾼다.
    """
    cfg = build_config(PRESETS["catch"])
    renderer = Renderer(cfg, "rgb_array")
    y_arm = cfg["catch"]["y_arm"]
    target = (cfg["catch"]["x_tower"], y_arm)
    state = replace(a_state(cfg), y=y_arm, theta=0.3, thrust=1.5 * G)

    frame = renderer.draw(state, target, Outcome.SUCCESS)
    assert frame.shape == (HEIGHT, WIDTH, 3)

    # 원본 state는 그대로다 — frozen dataclass가 이를 강제하지만, 렌더러가
    # 별도 사본을 그리기용으로 만든다는 계약을 명시적으로 고정해 둔다.
    assert state.y == y_arm
    assert state.theta == 0.3

    drawn = renderer._catch_draw_state(state, Outcome.SUCCESS)
    assert drawn is not state
    assert drawn.y < y_arm
    assert drawn.theta == 0.0
    assert drawn.thrust == 0.0

    # SUCCESS가 아니거나 catch 태스크가 아니면 원본을 그대로 돌려준다.
    assert renderer._catch_draw_state(state, Outcome.IN_PROGRESS) is state
    renderer.close()
