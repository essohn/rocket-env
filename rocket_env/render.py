"""pygame 렌더링.

전부 벡터로 그린다 — 외부 이미지 에셋이 없다. 원본 저장소의 배경 이미지가
CC BY-NC-SA라서 쓸 수 없기도 하고, 벡터가 해상도와 무관하게 깔끔하다.

카메라는 로켓과 목표점을 함께 화면에 담도록 매 프레임 목표 중심·배율을
계산한 뒤 지수 평활로 따라간다. 세계가 570 m 인데 포획 팔은 80 m 높이라,
고정 배율로는 화면 대부분이 빈 하늘이 되기 때문이다.
"""

import math
from dataclasses import replace

import numpy as np
import pygame

from rocket_env.physics import (
    DT,
    G,
    ROCKET_HEIGHT,
    WORLD_X_MAX,
    WORLD_X_MIN,
)
from rocket_env.types import Outcome, State

WIDTH, HEIGHT = 640, 960

# --- 카메라 ---
MIN_SCALE = WIDTH / (WORLD_X_MAX - WORLD_X_MIN)   # 세계 전체 폭이 보이는 최소 배율
MAX_SCALE = 6.0                                    # 최대 줌
CAMERA_SMOOTHING = 0.12                            # 0에 가까울수록 부드럽다(느리다)

# --- 연기 입자 ---
MAX_PARTICLES = 500
PARTICLE_LIFE = 1.2          # 초
SMOKE_COLOR = (215, 215, 225)

# --- 구름 ---
# 배경이 단색 그라디언트뿐이면 34 m/s 로 떨어지든 2 m/s 로 기어가든 화면상
# 차이가 없다. 구름을 월드 좌표에 고정해 두면 카메라가 내려가며 스쳐
# 지나가서 하강 속도가 눈으로 느껴진다.
CLOUD_COUNT = 34
CLOUD_COLOR = (238, 242, 250)
CLOUD_SEED = 20260722

# --- 타워(포획) ---
# 마스트를 포획 지점(x_tower)에서 왼쪽으로 물려 그린다 — 물리 목표는
# 그대로 x_tower 다, 그림만 옆으로 옮긴다. 실제 Mechazilla가 타워 옆에서
# 팔을 뻗어 붙잡는 모습을 흉내낸다.
TOWER_OFFSET = 55.0
TOWER_HEIGHT_FACTOR = 1.6

SKY_TOP = (12, 18, 40)
SKY_BOTTOM = (70, 96, 140)
GROUND_COLOR = (38, 40, 44)
PAD_COLOR = (200, 190, 90)
TOWER_COLOR = (150, 155, 165)
ARM_COLOR = (230, 120, 60)
JAW_COLOR = (255, 175, 60)
JAW_CLOSED_HALF = 5.0     # 다 물었을 때 턱 안쪽 간격 (m)
BODY_COLOR = (232, 234, 238)
FIN_COLOR = (120, 125, 135)
TRAIL_COLOR = (90, 160, 220)
HUD_COLOR = (225, 230, 240)
LIVERY_COLOR = (40, 60, 120)

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

        # (중심x, 중심y, 배율). reset()에서 None으로 돌려 다음 에피소드가
        # 평활 없이 즉시 맞춰지게 한다.
        self._cam: tuple[float, float, float] | None = None
        self._particles: list[dict] = []
        self._rng = np.random.default_rng()

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
        # 기체 도색도 마찬가지로 한 번만 만들어 캐시한다.
        self._livery = self._build_livery()
        # 구름도 고정 시드로 한 번 만들어 재사용한다 — 매 에피소드, 매
        # 렌더러 인스턴스에서 같은 하늘이 보여야 한다(재현성).
        self._clouds = self._build_clouds()

    # --- 공개 API ---

    def reset(self) -> None:
        self.trail = []
        self._cam = None
        self._particles = []

    def draw(self, state: State, target: tuple[float, float],
             outcome: str, grip: float = 0.0):
        """한 프레임을 그린다.

        `grip`은 순전히 연출용 상태다(0=벌어짐, 1=다 묾) — 물리에는
        전혀 관여하지 않는다. 기본값 0.0이라 기존 호출부(테스트 등)는
        그대로 동작한다.
        """
        draw_state = self._catch_draw_state(state, outcome)

        self._update_camera(draw_state, target)
        self._update_particles(draw_state)

        self.surface.blit(self._sky_surface, (0, 0))
        self._draw_clouds()
        self._ground()
        self._structure(target, grip)
        self._draw_particles()

        self.trail.append(self._to_px(draw_state.x, draw_state.y))
        self._trail()
        self._rocket(draw_state)
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

    # --- 포획 연출 ---

    def _catch_draw_state(self, state: State, outcome: str) -> State:
        """포획 성공 프레임에서 그리기 전용으로 쓸 상태를 만든다.

        실제 판정은 로켓 중심이 팔 높이를 지나는 순간 일어나지만, 로켓은
        50 m 라 그 상태 그대로 그리면 몸통이 팔을 관통한 것처럼 보인다.
        상단이 팔 바로 위에 걸리도록 중심을 내려 매달린 모습으로 그린다.

        `dataclasses.replace`로 만든 로컬 사본만 바꾼다 — 원본 `state`는
        건드리지 않는다. env.py가 살아있는 에피소드 상태를 넘기므로, 여기서
        물리 상태를 고치면 에피소드가 corrupt된다.
        """
        if self.cfg["task"] != "catch" or outcome != Outcome.SUCCESS:
            return state
        y_arm = self.cfg["catch"]["y_arm"]
        return replace(state, y=y_arm - ROCKET_HEIGHT / 2.0 + 4.0,
                       theta=0.0, thrust=0.0)

    # --- 카메라 ---

    def _camera_target(self, state: State, target: tuple[float, float]):
        """로켓과 목표가 모두 여유 있게 들어오는 (중심x, 중심y, 배율)."""
        cx = (state.x + target[0]) / 2.0
        cy = (state.y + target[1]) / 2.0
        span_x = abs(state.x - target[0]) + 6.0 * ROCKET_HEIGHT
        span_y = abs(state.y - target[1]) + 6.0 * ROCKET_HEIGHT
        scale = min(WIDTH / span_x, HEIGHT / span_y)
        return cx, cy, min(max(scale, MIN_SCALE), MAX_SCALE)

    def _update_camera(self, state: State, target: tuple[float, float]) -> None:
        tx, ty, tscale = self._camera_target(state, target)
        if self._cam is None:
            self._cam = (tx, ty, tscale)
            return
        cx, cy, cscale = self._cam
        k = CAMERA_SMOOTHING
        self._cam = (cx + (tx - cx) * k, cy + (ty - cy) * k,
                     cscale + (tscale - cscale) * k)

    # --- 좌표 변환 ---

    def _to_px(self, x: float, y: float) -> tuple[int, int]:
        cx, cy, scale = self._cam
        return (int(WIDTH / 2 + (x - cx) * scale),
                int(HEIGHT / 2 - (y - cy) * scale))

    def _body_to_world(self, state: State, bx: float, by: float) -> tuple[float, float]:
        """기체 좌표(bx, by)를 자세각 theta만큼 회전한 세계 좌표로."""
        cos_t, sin_t = math.cos(state.theta), math.sin(state.theta)
        return (state.x + bx * cos_t - by * sin_t,
                state.y + bx * sin_t + by * cos_t)

    def _body_to_px(self, state: State, bx: float, by: float) -> tuple[int, int]:
        return self._to_px(*self._body_to_world(state, bx, by))

    # --- 그리기 ---

    def _build_sky(self) -> pygame.Surface:
        sky = pygame.Surface((WIDTH, HEIGHT))
        for row in range(HEIGHT):
            t = row / HEIGHT
            color = tuple(int(SKY_TOP[i] + (SKY_BOTTOM[i] - SKY_TOP[i]) * t)
                          for i in range(3))
            pygame.draw.line(sky, color, (0, row), (WIDTH, row))
        return sky

    def _build_clouds(self) -> list[dict]:
        """월드 좌표에 고정된 구름. 카메라가 내려가면 스쳐 지나가며
        하강 속도를 눈으로 알 수 있게 한다."""
        rng = np.random.default_rng(CLOUD_SEED)
        clouds = []
        for _ in range(CLOUD_COUNT):
            # 로켓(50 m)보다 확실히 작게 잡는다. 반경이 40 m 까지 커지면
            # 화면을 덮고 HUD 뒤로 겹쳐 값을 읽기 어려워진다. 작고 많은
            # 쪽이 같은 속도 단서를 주면서 하늘처럼 보인다.
            r = float(rng.uniform(7.0, 20.0))
            clouds.append({
                "x": float(rng.uniform(WORLD_X_MIN - 120.0, WORLD_X_MAX + 120.0)),
                "y": float(rng.uniform(70.0, 540.0)),
                "r": r,
                # 알파가 낮으면 어두운 남색 하늘 위에서 흰 구름이 탁한 회색
                # 덩어리가 되어 먹구름처럼 읽힌다. 충분히 올려야 의도한
                # 밝은 색이 나온다.
                "alpha": int(rng.integers(150, 215)),
                # 뭉게구름처럼 보이도록 원을 몇 개 겹친다
                "puffs": [(float(rng.uniform(-1.3, 1.3)) * r,
                           float(rng.uniform(-0.35, 0.35)) * r,
                           float(rng.uniform(0.55, 1.0)) * r) for _ in range(5)],
            })
        return clouds

    def _draw_clouds(self) -> None:
        # 연기 입자와 같은 이유로, 알파 서피스 한 장에 전부 그린 뒤 한 번만
        # blit한다 — 구름마다 서피스를 새로 만들면 18개를 매 프레임 blit
        # 하게 되어 낭비다.
        _, _, scale = self._cam
        overlay = None
        for cloud in self._clouds:
            cx, cy = self._to_px(cloud["x"], cloud["y"])
            reach = int(cloud["r"] * 1.5 * scale) + 20
            if cx + reach < 0 or cx - reach > WIDTH or cy + reach < 0 or cy - reach > HEIGHT:
                continue  # 화면 밖 구름은 건너뛴다
            if overlay is None:
                overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            color = (*CLOUD_COLOR, cloud["alpha"])
            for dx, dy, pr in cloud["puffs"]:
                center = (cx + int(dx * scale), cy + int(dy * scale))
                radius = max(2, int(pr * scale))
                pygame.draw.circle(overlay, color, center, radius)
        if overlay is not None:
            self.surface.blit(overlay, (0, 0))

    def _ground(self) -> None:
        # 카메라가 움직이므로 지면은 더 이상 화면 하단 고정 띠가 아니다.
        # y=0 이 화면의 어디에 오는지 계산해 그 아래를 채운다 — 안 그러면
        # 카메라가 올라갈 때 지면이 세계에서 분리되어 보인다.
        _, ground_top = self._to_px(0.0, 0.0)
        ground_top = max(0, min(HEIGHT, ground_top))
        if ground_top < HEIGHT:
            pygame.draw.rect(
                self.surface, GROUND_COLOR,
                pygame.Rect(0, ground_top, WIDTH, HEIGHT - ground_top))

    def _structure(self, target: tuple[float, float], grip: float) -> None:
        if self.cfg["task"] == "landing":
            radius = self.cfg["success"]["zone_r"]
            left = self._to_px(-radius, 0.0)
            right = self._to_px(radius, 0.0)
            pygame.draw.line(self.surface, PAD_COLOR, left, right,
                             self._scaled_width(6.0))
            return

        x_tower, y_arm, arm_half, window_half = self._catch_geometry(target)
        # 마스트는 포획 지점 정중앙이 아니라 왼쪽으로 물러난 자리에 선다 —
        # 그래야 로켓이 타워 "위"가 아니라 "옆"에 잡힌 것처럼 보인다.
        mast_x = x_tower - TOWER_OFFSET
        mast_top = y_arm * TOWER_HEIGHT_FACTOR

        pygame.draw.line(self.surface, TOWER_COLOR,
                         self._to_px(mast_x, 0.0), self._to_px(mast_x, mast_top),
                         self._scaled_width(8.0))
        # 격자 살대 — 기둥 하나만 그리면 마스트가 아니라 깃대처럼 보인다.
        for frac in (0.2, 0.45, 0.7, 0.92):
            y = mast_top * frac
            pygame.draw.line(self.surface, TOWER_COLOR,
                             self._to_px(mast_x - 7.0, y),
                             self._to_px(mast_x + 7.0, y),
                             self._scaled_width(3.0, min_px=1))

        # 가로보(cantilever): 마스트에서 포획 지점 너머까지 옆으로 뻗는다.
        pygame.draw.line(self.surface, TOWER_COLOR,
                         self._to_px(mast_x, y_arm),
                         self._to_px(x_tower + arm_half, y_arm),
                         self._scaled_width(7.0))

        # 실제 포획 판정 범위(±zone_r). 이걸 따로 그리지 않으면 학생은
        # 팔 안쪽으로 잘 지나간 것처럼 보이는데 MISSED 가 뜨는 이유를
        # 알 수 없다. 디버깅하라고 만든 그림이 디버깅을 방해하게 된다.
        pygame.draw.line(self.surface, ARM_COLOR,
                         self._to_px(x_tower - window_half, y_arm),
                         self._to_px(x_tower + window_half, y_arm),
                         self._scaled_width(11.0))

        # 젓가락은 가로보 위에 그려야 구조물에 달린 것처럼 보인다 — 가로보를
        # 먼저 그리고 그 위에 덧그린다.
        self._draw_jaws(x_tower, y_arm, window_half, grip)

    def _draw_jaws(self, x_tower: float, y_arm: float, window_half: float,
                   grip: float) -> None:
        """포획 창 자리에 달린 두 턱. grip 0(벌어짐) -> 1(다 묾)으로 오므라든다."""
        inner = window_half + (JAW_CLOSED_HALF - window_half) * grip
        width = self._scaled_width(9.0, min_px=3)
        for sign in (-1, 1):
            root = self._to_px(x_tower + sign * (window_half + 12.0), y_arm)
            tip = self._to_px(x_tower + sign * inner, y_arm)
            pygame.draw.line(self.surface, JAW_COLOR, root, tip, width)

    def _scaled_width(self, base_px: float, min_px: int = 2) -> int:
        """카메라 배율에 맞춰 선 굵기를 조정한다.

        MIN_SCALE(세계 전체가 보이는 최소 배율)에서 base_px로 보이도록
        맞춘 값이라, 줌인할수록 그만큼 굵어져야 로켓 크기 대비 두께가
        일정하게 유지된다. 줌인 폭이 커진 지금은 고정 px로는 하이라인이
        되어버려서 최소값으로 하한을 둔다.
        """
        _, _, scale = self._cam
        return max(min_px, int(round(base_px * scale / MIN_SCALE)))

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
        pygame.draw.lines(self.surface, TRAIL_COLOR, False, self.trail[-400:],
                          self._scaled_width(2.0, min_px=1))

    # --- 연기 입자 ---

    def _update_particles(self, state: State) -> None:
        self._emit_particles(state)
        for p in self._particles:
            p["x"] += p["vx"] * DT
            p["y"] += p["vy"] * DT
            p["vx"] *= 0.88
            p["vy"] *= 0.88
            p["age"] += DT
        self._particles = [p for p in self._particles
                           if p["age"] <= PARTICLE_LIFE][-MAX_PARTICLES:]

    def _emit_particles(self, state: State) -> None:
        if state.thrust <= 0.0:
            return
        n = int(10 * state.thrust / (2.0 * G))
        if n <= 0:
            return

        nozzle_x, nozzle_y = self._body_to_world(state, 0.0, -ROCKET_HEIGHT / 2.0)
        # 배기 방향: 기체 아래쪽을 phi만큼 꺾은 방향(기체 좌표)을 theta로
        # 세계 좌표로 회전한다. 위치가 아니라 방향 벡터라 평행이동 성분은
        # 빼고 회전 성분만 적용한다.
        bxd, byd = math.sin(state.phi), -math.cos(state.phi)
        cos_t, sin_t = math.cos(state.theta), math.sin(state.theta)
        dir_x = bxd * cos_t - byd * sin_t
        dir_y = bxd * sin_t + byd * cos_t
        base_angle = math.atan2(dir_y, dir_x)

        for _ in range(n):
            speed = self._rng.uniform(40.0, 70.0)
            angle = base_angle + self._rng.uniform(-0.25, 0.25)
            self._particles.append({
                "x": nozzle_x, "y": nozzle_y,
                "vx": speed * math.cos(angle), "vy": speed * math.sin(angle),
                "age": 0.0,
            })

    def _draw_particles(self) -> None:
        if not self._particles:
            return
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        _, _, scale = self._cam
        for p in self._particles:
            frac = p["age"] / PARTICLE_LIFE
            radius_px = max(1, int((2.0 + 14.0 * frac) * scale))
            alpha = int(170 * (1.0 - frac) ** 1.5)
            pygame.draw.circle(overlay, (*SMOKE_COLOR, alpha),
                               self._to_px(p["x"], p["y"]), radius_px)
        self.surface.blit(overlay, (0, 0))

    # --- 로켓 ---

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

        self._paint_livery(state)
        self._flame(state)

    def _build_livery(self) -> pygame.Surface:
        """세로로 읽히는 기체 도색. 한 번만 만든다."""
        font = pygame.font.SysFont("helvetica", 40, bold=True)
        glyphs = [font.render(ch, True, LIVERY_COLOR) for ch in "YONSEI"]
        w = max(g.get_width() for g in glyphs)
        h = sum(g.get_height() for g in glyphs)
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        y = 0
        for g in glyphs:
            surf.blit(g, ((w - g.get_width()) // 2, y))
            y += g.get_height()
        return surf

    def _paint_livery(self, state: State) -> None:
        _, _, scale = self._cam
        zoom = (ROCKET_HEIGHT * 0.55 * scale) / self._livery.get_height()
        if zoom <= 0.05:
            return
        # body_to_px가 만드는 노즈 방향(화면 벡터)은 (-sinθ, -cosθ)다.
        # pygame.transform.rotozoom(φ)은 이미지의 "위"를 화면 벡터
        # (-sinφ, -cosφ)로 돌린다(양의 φ가 화면에서 반시계 방향). 두 벡터가
        # 같으려면 φ = θ 여야 한다 — 부호를 뒤집으면 글자가 기체와 반대로
        # 돌아 큰 각도(±30° 이상)에서 뚜렷하게 어긋난다. ±30°/±70°로 실제
        # 렌더링해 확인했다(계산이 아니라 픽셀로).
        art = pygame.transform.rotozoom(self._livery, math.degrees(state.theta), zoom)
        self.surface.blit(art, art.get_rect(
            center=self._body_to_px(state, 0.0, 2.0)))

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
