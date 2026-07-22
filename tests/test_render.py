"""렌더링 smoke 테스트.

픽셀 값을 검증하지는 않는다. 크래시 없이 올바른 형태의 배열이 나오는지,
두 태스크와 모든 종료 상태에서 그려지는지를 본다. 다만 학생이 오해할 수
있는 두 지점 — 포획 창 폭과 렌더러 여러 개의 수명 — 은 따로 고정한다.
"""

import math
import os
from dataclasses import replace

import numpy as np
import pygame
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


def test_caught_rocket_is_not_teleported():
    """포획 프레임에서 로켓 위치가 그대로여야 한다.

    예전에는 매달린 자리로 내려 그렸는데, 판정이 로켓 중심이 팔 높이를
    지날 때 일어나므로 한 프레임 만에 20 m 넘게 순간이동한 것처럼 보였다.
    관통해 보이는 문제는 앞/뒤 팔을 나눠 로켓을 그 사이에 끼우는 것으로
    푼다 — 위치를 옮겨서가 아니다.
    """
    cfg = build_config(PRESETS["catch"])
    renderer = Renderer(cfg, "rgb_array")
    y_arm = cfg["catch"]["y_arm"]
    target = (cfg["catch"]["x_tower"], y_arm)
    state = replace(a_state(cfg), y=y_arm, theta=0.3, thrust=1.5 * G)

    frame = renderer.draw(state, target, Outcome.SUCCESS)
    assert frame.shape == (HEIGHT, WIDTH, 3)

    drawn = renderer._catch_draw_state(state, Outcome.SUCCESS)
    assert drawn.y == state.y          # 점프 없음
    assert drawn.theta == state.theta  # 자세도 그대로
    assert drawn.thrust == 0.0         # 엔진만 꺼진다

    # 원본은 건드리지 않는다 — env.py 가 살아있는 상태를 넘기기 때문이다.
    assert state.thrust == 1.5 * G
    assert renderer._catch_draw_state(state, Outcome.IN_PROGRESS) is state
    renderer.close()


def test_jaws_close_as_the_rocket_approaches():
    """집게는 잡히는 순간이 아니라 다가오는 동안 점점 오므라든다.

    포획 시점에만 닫히면 동작이 한 프레임에 끝나 보이지 않는다.
    """
    cfg = build_config(PRESETS["catch"])
    renderer = Renderer(cfg, "rgb_array")
    y_arm = cfg["catch"]["y_arm"]
    target = (cfg["catch"]["x_tower"], y_arm)

    far = renderer._approach_grip(replace(a_state(cfg), x=0.0, y=y_arm + 200.0),
                                  target)
    near = renderer._approach_grip(replace(a_state(cfg), x=0.0, y=y_arm + 20.0),
                                   target)
    arrived = renderer._approach_grip(replace(a_state(cfg), x=0.0, y=y_arm),
                                      target)
    assert far == 0.0
    assert 0.0 < near < 1.0
    assert arrived == 1.0

    # 수평으로 크게 빗나간 로켓에는 닫히지 않는다 — 어색하기 때문이다.
    off = renderer._approach_grip(
        replace(a_state(cfg), x=cfg["success"]["zone_r"] * 10.0, y=y_arm), target)
    assert off == 0.0
    renderer.close()


def test_livery_rotates_with_the_hull_not_against_it():
    """도색이 동체와 같은 방향으로 돌아야 한다.

    과거 버그: rotozoom에 -theta를 넘겨 글자가 기체와 반대로 돌았다.
    작은 각도(예: 14도)에서는 부호가 틀려도 거의 비슷해 보여 회귀를
    못 잡는다. ±70도라는 큰 각으로, 그것도 픽셀을 직접 세어 확인한다.

    도색 서피스를 위(노즈 쪽)는 빨강, 아래(핀 쪽)는 파랑인 마커로 바꿔치기
    한 뒤 실제 draw() 로 그려서, 빨강 무게중심이 파랑보다 노즈 방향으로
    더 나아가 있는지를 잰다. 부호가 뒤집히면 이 관계가 깨진다 — 실제로
    수정 전 코드로 이 테스트를 돌려 실패를 확인했다.
    """
    cfg = build_config(PRESETS["landing-descent"])
    renderer = Renderer(cfg, "rgb_array")

    w, h = renderer._livery.get_size()
    marker = pygame.Surface((w, h), pygame.SRCALPHA)
    marker.fill((255, 0, 0, 255), pygame.Rect(0, 0, w, h // 2))
    marker.fill((0, 0, 255, 255), pygame.Rect(0, h // 2, w, h - h // 2))
    renderer._livery = marker

    state = replace(a_state(cfg), x=0.0, y=200.0, thrust=0.0)
    for deg in (70.0, -70.0):
        s = replace(state, theta=math.radians(deg))
        frame = renderer.draw(s, (0.0, 25.0), Outcome.IN_PROGRESS)

        center = np.array(renderer._body_to_px(s, 0.0, 2.0), dtype=float)
        nose_vec = np.array(renderer._body_to_px(s, 0.0, 20.0), dtype=float) - center

        red_ys, red_xs = np.where(np.all(frame == [255, 0, 0], axis=-1))
        blue_ys, blue_xs = np.where(np.all(frame == [0, 0, 255], axis=-1))
        assert red_xs.size > 0 and blue_xs.size > 0

        red_centroid = np.array([red_xs.mean(), red_ys.mean()])
        blue_centroid = np.array([blue_xs.mean(), blue_ys.mean()])
        red_proj = np.dot(red_centroid - center, nose_vec)
        blue_proj = np.dot(blue_centroid - center, nose_vec)
        assert red_proj > blue_proj, (
            f"theta={deg}deg: 노즈 쪽(빨강)이 핀 쪽(파랑)보다 노즈 방향으로 "
            "더 나아가 있어야 한다")
    renderer.close()


def test_clouds_are_deterministic_across_renderers():
    """같은 시드로 만든 구름은 렌더러를 새로 만들어도 같다.

    구름은 에피소드마다, 렌더러 인스턴스마다 다시 만들어지므로(매번
    `_build_clouds`를 호출) 고정 시드가 아니면 매 실행마다 하늘이 달라져
    재현성이 깨진다.
    """
    cfg = build_config(PRESETS["landing-descent"])
    first = Renderer(cfg, "rgb_array")
    second = Renderer(cfg, "rgb_array")
    assert first._clouds == second._clouds
    first.close()
    second.close()


def test_jaws_close_as_grip_increases():
    """grip 이 커질수록 화면에 그려지는 턱의 안쪽 간격이 좁아진다.

    내부 공식을 재계산하지 않고, 실제로 그려진 JAW_COLOR 픽셀을 스캔해
    두 턱의 가장 안쪽(중앙에 가장 가까운) 픽셀 사이 거리를 잰다 — 그래야
    `_draw_jaws`가 실제로 무엇을 그리는지 검증한 것이 된다.
    """
    from rocket_env.render import JAW_COLOR

    cfg = build_config(PRESETS["catch"])
    renderer = Renderer(cfg, "rgb_array")
    target = (cfg["catch"]["x_tower"], cfg["catch"]["y_arm"])
    # 로켓을 팔에서 충분히 떨어뜨려 두어 몸통이 턱 픽셀을 가리지 않게 한다.
    state = replace(a_state(cfg), x=target[0], y=target[1] + 80.0, theta=0.0)

    def inner_gap_px(grip: float) -> int:
        frame = renderer.draw(state, target, Outcome.IN_PROGRESS, grip=grip)
        ys, xs = np.where(np.all(frame == JAW_COLOR, axis=-1))
        assert xs.size > 0, "턱이 화면에 그려지지 않았다"
        cx = renderer._to_px(*target)[0]
        left_xs, right_xs = xs[xs < cx], xs[xs >= cx]
        assert left_xs.size > 0 and right_xs.size > 0
        return int(right_xs.min() - left_xs.max())

    gaps = [inner_gap_px(g) for g in (0.0, 0.5, 1.0)]
    assert gaps[0] > gaps[1] > gaps[2], gaps
    renderer.close()


def test_tower_mast_is_drawn_beside_the_capture_point():
    """마스트 x 좌표가 x_tower 보다 왼쪽이다 — 로켓이 타워 위가 아니라
    옆에 잡힌다."""
    cfg = build_config(PRESETS["catch"])
    renderer = Renderer(cfg, "rgb_array")
    target = (cfg["catch"]["x_tower"], cfg["catch"]["y_arm"])
    x_tower, _, _, _ = renderer._catch_geometry(target)

    from rocket_env.render import TOWER_OFFSET
    assert TOWER_OFFSET > 0.0

    mast_x = x_tower - TOWER_OFFSET
    assert mast_x < x_tower
    renderer.close()
