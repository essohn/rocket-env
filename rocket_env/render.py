"""pygame 렌더링.

전부 벡터로 그린다 — 외부 이미지 에셋이 없다. 원본 저장소의 배경 이미지가
CC BY-NC-SA라서 쓸 수 없기도 하고, 벡터가 해상도와 무관하게 깔끔하다.
"""

import math

import numpy as np
import pygame

from rocket_env.physics import (
    G,
    ROCKET_HEIGHT,
    WORLD_X_MAX,
    WORLD_X_MIN,
)
from rocket_env.types import Outcome, State

WIDTH, HEIGHT = 640, 960
SCALE = WIDTH / (WORLD_X_MAX - WORLD_X_MIN)   # px per meter
GROUND_PX = 60                                # 화면 하단에서 지면까지 여백

SKY_TOP = (12, 18, 40)
SKY_BOTTOM = (70, 96, 140)
GROUND_COLOR = (38, 40, 44)
PAD_COLOR = (200, 190, 90)
TOWER_COLOR = (150, 155, 165)
ARM_COLOR = (230, 120, 60)
BODY_COLOR = (232, 234, 238)
FIN_COLOR = (120, 125, 135)
TRAIL_COLOR = (90, 160, 220)
HUD_COLOR = (225, 230, 240)

BANNER_TEXT = {
    Outcome.SUCCESS: ("LANDED / CAUGHT", (110, 220, 130)),
    Outcome.CRASH: ("CRASHED", (230, 90, 80)),
    Outcome.MISSED: ("MISSED", (240, 170, 60)),
    Outcome.TIMEOUT: ("OUT OF TIME", (200, 200, 200)),
    Outcome.OUT_OF_FUEL: ("OUT OF FUEL", (240, 120, 200)),
}


class Renderer:
    def __init__(self, cfg: dict, render_mode: str):
        pygame.init()
        pygame.font.init()
        self.cfg = cfg
        self.render_mode = render_mode
        self.font = pygame.font.SysFont("monospace", 15)
        self.banner_font = pygame.font.SysFont("monospace", 34, bold=True)
        self.trail: list[tuple[int, int]] = []

        if render_mode == "human":
            self.surface = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("rocket-env")
            self.clock = pygame.time.Clock()
        else:
            self.surface = pygame.Surface((WIDTH, HEIGHT))
            self.clock = None

        # 하늘 그라디언트는 매 프레임 960줄을 다시 그릴 이유가 없다.
        # 한 번 만들어두고 blit한다.
        self._sky_surface = self._build_sky()

    # --- 공개 API ---

    def reset(self) -> None:
        self.trail = []

    def draw(self, state: State, target: tuple[float, float],
             outcome: str):
        self.surface.blit(self._sky_surface, (0, 0))
        self._ground()
        self._structure(target)

        self.trail.append(self._to_px(state.x, state.y))
        self._trail()
        self._rocket(state)
        self._hud(state)

        if outcome != Outcome.IN_PROGRESS:
            self._banner(outcome)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(20)
            return None
        return np.transpose(pygame.surfarray.array3d(self.surface), (1, 0, 2))

    def close(self) -> None:
        """디스플레이만 닫는다.

        pygame.quit() 은 프로세스 전역이다. 노트북에서 렌더링 환경을 둘
        이상 띄워둔 채 하나를 닫으면 나머지의 폰트와 서피스까지 무효가 된다.
        """
        if self.render_mode == "human":
            pygame.display.quit()

    # --- 좌표 변환 ---

    def _to_px(self, x: float, y: float) -> tuple[int, int]:
        return (int(WIDTH / 2 + x * SCALE),
                int(HEIGHT - GROUND_PX - y * SCALE))

    # --- 그리기 ---

    def _build_sky(self) -> pygame.Surface:
        sky = pygame.Surface((WIDTH, HEIGHT))
        for row in range(HEIGHT):
            t = row / HEIGHT
            color = tuple(int(SKY_TOP[i] + (SKY_BOTTOM[i] - SKY_TOP[i]) * t)
                          for i in range(3))
            pygame.draw.line(sky, color, (0, row), (WIDTH, row))
        return sky

    def _ground(self) -> None:
        pygame.draw.rect(self.surface, GROUND_COLOR,
                         pygame.Rect(0, HEIGHT - GROUND_PX, WIDTH, GROUND_PX))

    def _structure(self, target: tuple[float, float]) -> None:
        if self.cfg["task"] == "landing":
            radius = self.cfg["success"]["zone_r"]
            left = self._to_px(-radius, 0.0)
            right = self._to_px(radius, 0.0)
            pygame.draw.line(self.surface, PAD_COLOR, left, right, 6)
            return

        x_tower, y_arm, arm_half, window_half = self._catch_geometry(target)
        base = self._to_px(x_tower, 0.0)
        top = self._to_px(x_tower, y_arm * 1.25)
        pygame.draw.line(self.surface, TOWER_COLOR, base, top, 8)

        # 팔 구조물. 포획 창보다 넓게 그린다 — 6 m 는 화면에서 13 px 밖에
        # 안 돼서 구조물로 알아보기 어렵다.
        pygame.draw.line(self.surface, TOWER_COLOR,
                         self._to_px(x_tower - arm_half, y_arm),
                         self._to_px(x_tower + arm_half, y_arm), 7)
        # 실제 포획 판정 범위(±zone_r). 이걸 따로 그리지 않으면 학생은
        # 팔 안쪽으로 잘 지나간 것처럼 보이는데 MISSED 가 뜨는 이유를
        # 알 수 없다. 디버깅하라고 만든 그림이 디버깅을 방해하게 된다.
        pygame.draw.line(self.surface, ARM_COLOR,
                         self._to_px(x_tower - window_half, y_arm),
                         self._to_px(x_tower + window_half, y_arm), 11)

    def _catch_geometry(self, target: tuple[float, float]):
        """타워 x, 팔 높이, 팔 반폭, 실제 포획 창 반폭을 돌려준다.

        구조물 폭과 판정 폭을 한곳에서 분리해 두어야, 렌더러가 임계값을
        잘못 그리는 일이 생기지 않고 테스트로 고정할 수도 있다.
        """
        x_tower, y_arm = target
        window_half = self.cfg["success"]["zone_r"]
        return x_tower, y_arm, max(window_half * 3.0, 18.0), window_half

    def _trail(self) -> None:
        if len(self.trail) < 2:
            return
        pygame.draw.lines(self.surface, TRAIL_COLOR, False, self.trail[-400:], 2)

    def _rocket(self, state: State) -> None:
        half = ROCKET_HEIGHT / 2.0
        body = [(-4.0, -half), (4.0, -half), (4.0, half - 10.0),
                (0.0, half), (-4.0, half - 10.0)]
        pygame.draw.polygon(
            self.surface, BODY_COLOR,
            [self._body_to_px(state, bx, by) for bx, by in body])

        fins = [(-4.0, -half + 4.0), (-11.0, -half - 3.0), (-4.0, -half + 12.0)]
        pygame.draw.polygon(
            self.surface, FIN_COLOR,
            [self._body_to_px(state, bx, by) for bx, by in fins])
        pygame.draw.polygon(
            self.surface, FIN_COLOR,
            [self._body_to_px(state, -bx, by) for bx, by in fins])

        self._flame(state)

    def _flame(self, state: State) -> None:
        if state.thrust <= 0.0:
            return
        length = 6.0 + 22.0 * (state.thrust / (2.0 * G))
        nozzle = (0.0, -ROCKET_HEIGHT / 2.0)
        tip = (length * math.sin(state.phi),
               -ROCKET_HEIGHT / 2.0 - length * math.cos(state.phi))
        color = (255, 210, 90) if state.thrust < 1.5 * G else (255, 140, 60)
        points = [
            self._body_to_px(state, nozzle[0] - 3.0, nozzle[1]),
            self._body_to_px(state, nozzle[0] + 3.0, nozzle[1]),
            self._body_to_px(state, tip[0], tip[1]),
        ]
        pygame.draw.polygon(self.surface, color, points)

    def _body_to_px(self, state: State, bx: float, by: float) -> tuple[int, int]:
        """기체 좌표(bx, by)를 화면 픽셀로. 자세각 theta만큼 회전한다."""
        cos_t, sin_t = math.cos(state.theta), math.sin(state.theta)
        wx = state.x + bx * cos_t - by * sin_t
        wy = state.y + bx * sin_t + by * cos_t
        return self._to_px(wx, wy)

    def _hud(self, state: State) -> None:
        speed = math.hypot(state.vx, state.vy)
        capacity = self.cfg["fuel"]["capacity"]
        fuel_text = ("inf" if capacity is None
                     else f"{state.fuel:6.1f}/{capacity:.0f}")
        lines = [
            f"alt   {state.y - ROCKET_HEIGHT / 2:7.1f} m",
            f"speed {speed:7.1f} m/s   (vx {state.vx:6.1f}  vy {state.vy:6.1f})",
            f"tilt  {math.degrees(state.theta):7.1f} deg",
            f"fuel  {fuel_text}",
            f"wind  {state.wind_x:7.1f} m/s",
            f"step  {state.step:5d} / {self.cfg['max_steps']}",
        ]
        for i, line in enumerate(lines):
            self.surface.blit(
                self.font.render(line, True, HUD_COLOR), (12, 12 + i * 19))

    def _banner(self, outcome: str) -> None:
        text, color = BANNER_TEXT[outcome]
        surface = self.banner_font.render(text, True, color)
        rect = surface.get_rect(center=(WIDTH // 2, HEIGHT // 3))
        self.surface.blit(surface, rect)
