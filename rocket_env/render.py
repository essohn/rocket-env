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
MAX_SCALE = 14.0                                   # 최대 줌
CAMERA_SMOOTHING = 0.12                            # 0에 가까울수록 부드럽다(느리다)

# --- 연기 입자 ---
MAX_PARTICLES = 700
PARTICLE_LIFE = 1.2          # 초 — 연기
SMOKE_COLOR = (215, 215, 225)

# 부스러기/먼지. 연기와 달리 빠르게 튀어나가 금방 사라진다. 연기만 있으면
# 배기가 뭉근하게 보여서 추력의 격렬함이 전달되지 않는다.
DEBRIS_LIFE = 0.35           # 초
DEBRIS_COLOR = (250, 226, 190)

# --- 구름 ---
# 배경이 단색 그라디언트뿐이면 34 m/s 로 떨어지든 2 m/s 로 기어가든 화면상
# 차이가 없다. 구름을 월드 좌표에 고정해 두면 카메라가 내려가며 스쳐
# 지나가서 하강 속도가 눈으로 느껴진다.
CLOUD_COUNT = 34
CLOUD_COLOR = (252, 232, 224)   # 노을빛을 받은 구름
CLOUD_SEED = 20260722
# 일부 구름은 카메라에 더 가깝게 둔다 — 로켓보다 앞에 그리고, 카메라
# 중심에서의 화면 오프셋을 시차 계수만큼 부풀린다. 원경 구름만 있으면
# 배경이 평평해서 깊이가 느껴지지 않는다.
CLOUD_FRONT_RATIO = 0.3
CLOUD_FRONT_PARALLAX = 1.7

# --- 타워(포획) ---
# 마스트를 포획 지점(x_tower)에서 왼쪽으로 물려 그린다 — 물리 목표는
# 그대로 x_tower 다, 그림만 옆으로 옮긴다. 실제 Mechazilla가 타워 옆에서
# 팔을 뻗어 붙잡는 모습을 흉내낸다.
# 실물 비율: Super Heavy 71 m, 발사탑 146 m(2.06배), 팔 길이 약 40 m(0.56배),
# 부스터 중심선과 타워 간격 약 30 m(0.42배). 로켓이 50 m 이므로 각각
# 103 m / 28 m / 21 m 가 된다. 예전 값(이격 55 m, 가로보 총 100 m)은 팔이
# 로켓 길이의 2배로, 실물의 3.5배였다.
TRUSS_HALF_W = 4.5        # 트러스 폭의 절반 (m)
TRUSS_BAY = 9.0           # 브레이스 한 칸 길이 (m)
TOWER_OFFSET = 21.0
TOWER_HEIGHT_FACTOR = 1.3

# 석양. 천정의 짙은 보라에서 지평선의 주황까지 3단으로 섞는다 —
# 2단으로는 노을 특유의 붉은 중간대가 나오지 않는다.
SKY_TOP = (28, 26, 66)
SKY_MID = (188, 88, 84)
SKY_BOTTOM = (255, 178, 98)
GROUND_COLOR = (38, 40, 44)
PAD_COLOR = (200, 190, 90)
TOWER_COLOR = (150, 155, 165)
ARM_COLOR = (230, 120, 60)
JAW_COLOR = (255, 175, 60)
JAW_BACK_COLOR = (196, 128, 42)   # 뒤쪽 팔 — 그늘져 어둡게
JAW_CLOSED_HALF = 5.0     # 다 물었을 때 턱 안쪽 간격 (m)
# 카메라가 약간 위에서 내려다본다고 가정한다. 뒤쪽 팔은 화면에서 위로,
# 앞쪽 팔은 아래로 어긋나게 그려 둘이 구분되고, 로켓이 그 사이에 물린다.
# 젓가락은 마스트의 피봇에서 회전한다. 벌어지면 팁이 위아래로 갈라지고
# 조이면 수평으로 모인다 — 평행이동이 아니라 회전이라야 경첩처럼 보인다.
JAW_OPEN_SPREAD = 15.0    # 벌어졌을 때 팁의 수직 벌어짐 (m)
JAW_CLOSE_SPREAD = 4.0    # 다 조였을 때 (m)
JAW_TRUSS_HALF = 1.5      # 젓가락 트러스 폭의 절반 (m) — 본체보다 가늘다
JAW_TRUSS_BAY = 6.0
# 젓가락을 판정 높이보다 조금 위에 그린다. 순전히 연출용 오프셋이며,
# 판정 높이 자체는 그대로다 — 실제 포획 범위 표시선은 아래 y_arm 에
# 그대로 남겨 "그려진 것과 판정되는 것"이 어긋나지 않게 한다.
JAW_Y_OFFSET = 5.0
# 로켓이 이만큼 위로 접근하면 집게가 오므라들기 시작한다. 잡히는 순간에만
# 닫히면 동작이 보이지 않는다.
GRIP_APPROACH_RANGE = 75.0
# 접근 중에 대부분 닫는다. 예전 값(0.45)은 절반 이상을 포획 후 연출로
# 미뤄서, 로켓이 멎은 다음에야 젓가락이 좁혀지는 것처럼 보였다. 실제
# Mechazilla 도 부스터가 다가오는 동안 팔을 미리 좁힌다. 나머지 15%는
# 걸린 뒤 마저 조여 "고정"되는 느낌을 준다.
GRIP_APPROACH_MAX = 0.85
GRIP_ALIGN_RANGE = 3.0    # 수평 정렬이 이 배수 안이어야 닫기 시작한다
BODY_COLOR = (232, 234, 238)
BODY_HALF_W = 4.2          # 원통 반지름 (m)
# 상단 걸림 구조(리프트 포인트). 젓가락이 이 돌기에 걸려 기체를 든다.
PIN_Y = ROCKET_HEIGHT / 2.0 - 2.0   # 기체 맨 위 (걸림쇠는 최상단에 있다)
PIN_OUT = 3.4              # 몸통 밖으로 튀어나온 길이
PIN_THICK = 1.6
PIN_COLOR = (176, 182, 194)
# 포획 후 미끄러져 내려앉는 거리. 걸림쇠 높이(PIN_Y) 전체만큼 내리면
# 기체가 팔 한참 아래로 매달려 "너무 밑에서 걸린" 것처럼 보인다.
# 걸린 자리가 상단 가까이 남도록 짧게 잡는다.
SETTLE_DROP = 4.5
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
             outcome: str, grip: float | None = None, settle: float = 0.0):
        """한 프레임을 그린다.

        `grip`은 순전히 연출용 상태다(0=벌어짐, 1=다 묾) — 물리에는
        전혀 관여하지 않는다. 기본값 0.0이라 기존 호출부(테스트 등)는
        그대로 동작한다.
        """
        draw_state = self._catch_draw_state(state, outcome, settle)
        if grip is None:
            # 접근할수록 오므라든다. 잡히면 완전히 문다.
            grip = (1.0 if outcome == Outcome.SUCCESS
                    else self._approach_grip(draw_state, target))

        self._update_camera(draw_state, target)
        self._update_particles(draw_state)

        self.surface.blit(self._sky_surface, (0, 0))
        self._draw_clouds(front=False)      # 원경 구름 — 배경
        self._ground()
        self._structure(target, grip)
        self._draw_particles()

        self.trail.append(self._to_px(draw_state.x, draw_state.y))
        self._trail()
        self._rocket(draw_state)
        self._front_arm(target, grip)
        # 전경 구름은 로켓보다 카메라에 가까우므로 기체 위에 덮인다.
        # 가끔 로켓을 스쳐 지나가며 깊이감을 만든다.
        self._draw_clouds(front=True)
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

    def _catch_draw_state(self, state: State, outcome: str,
                          settle: float = 0.0) -> State:
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
        # 순간이동은 하지 않는다. 대신 settle(0~1)에 따라 걸림 구조가
        # 팔에 얹힐 때까지 PIN_Y 만큼 부드럽게 미끄러져 내려간다. 판정은
        # 로켓 중심이 팔 높이를 지날 때 일어나므로, 그 지점에서 pin이
        # 팔에 닿으려면 중심이 PIN_Y 만큼 더 내려가야 한다.
        # ease-out 지수를 3에서 2로 낮춰 초반 가속을 줄였다. 3제곱은 첫
        # 몇 프레임에 대부분을 내려와 버려서 "미끄러진다"기보다 튄다.
        eased = 1.0 - (1.0 - min(max(settle, 0.0), 1.0)) ** 2
        return replace(state, y=state.y - SETTLE_DROP * eased, thrust=0.0)

    # --- 카메라 ---

    def _camera_target(self, state: State, target: tuple[float, float]):
        """로켓과 목표가 모두 여유 있게 들어오는 (중심x, 중심y, 배율)."""
        cx = (state.x + target[0]) / 2.0
        cy = (state.y + target[1]) / 2.0
        # 여백이 로켓 6개 길이(300 m)나 되면 로켓이 화면에서 점만 해진다.
        # 2.2배(110 m)로 좁혀 기체를 크게 잡고, 배경이 빠르게 흘러 속도감도
        # 커진다.
        span_x = abs(state.x - target[0]) + 2.2 * ROCKET_HEIGHT
        span_y = abs(state.y - target[1]) + 2.2 * ROCKET_HEIGHT
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
            if t < 0.55:
                u = t / 0.55
                a, b = SKY_TOP, SKY_MID
            else:
                u = (t - 0.55) / 0.45
                a, b = SKY_MID, SKY_BOTTOM
            color = tuple(int(a[i] + (b[i] - a[i]) * u) for i in range(3))
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
                "y": float(rng.uniform(120.0, 1650.0)),
                "r": r,
                # 알파가 낮으면 어두운 남색 하늘 위에서 흰 구름이 탁한 회색
                # 덩어리가 되어 먹구름처럼 읽힌다. 충분히 올려야 의도한
                # 밝은 색이 나온다.
                "alpha": int(rng.integers(70, 125)),
                # 앞쪽 구름은 더 크고 진하게 — 가까이 있다는 인상을 준다
                "front": bool(rng.random() < CLOUD_FRONT_RATIO),
                # 뭉게구름처럼 보이도록 원을 몇 개 겹친다
                "puffs": [(float(rng.uniform(-1.3, 1.3)) * r,
                           float(rng.uniform(-0.35, 0.35)) * r,
                           float(rng.uniform(0.55, 1.0)) * r) for _ in range(5)],
            })
        return clouds

    def _cloud_px(self, cloud: dict) -> tuple[int, int]:
        """구름의 화면 좌표. 앞쪽 구름은 카메라 중심에서의 오프셋을
        시차 계수만큼 부풀려, 카메라가 움직일 때 더 빨리 흐르게 한다 —
        가까이 있는 물체가 빨리 지나가는 실제 시차를 흉내낸다."""
        px, py = self._to_px(cloud["x"], cloud["y"])
        if not cloud["front"]:
            return px, py
        k = CLOUD_FRONT_PARALLAX
        return (int(WIDTH / 2 + (px - WIDTH / 2) * k),
                int(HEIGHT / 2 + (py - HEIGHT / 2) * k))

    def _draw_clouds(self, front: bool = False) -> None:
        # 연기 입자와 같은 이유로, 알파 서피스 한 장에 전부 그린 뒤 한 번만
        # blit한다 — 구름마다 서피스를 새로 만들면 18개를 매 프레임 blit
        # 하게 되어 낭비다.
        _, _, scale = self._cam
        overlay = None
        for cloud in self._clouds:
            if cloud["front"] != front:
                continue
            cx, cy = self._cloud_px(cloud)
            size = cloud["r"] * (1.6 if front else 1.0)
            reach = int(size * 1.5 * scale) + 20
            if cx + reach < 0 or cx - reach > WIDTH or cy + reach < 0 or cy - reach > HEIGHT:
                continue  # 화면 밖 구름은 건너뛴다
            if overlay is None:
                overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            grow = 1.6 if front else 1.0
            alpha = min(255, int(cloud["alpha"] * (1.25 if front else 1.0)))
            color = (*CLOUD_COLOR, alpha)
            for dx, dy, pr in cloud["puffs"]:
                center = (cx + int(dx * grow * scale), cy + int(dy * grow * scale))
                radius = max(2, int(pr * grow * scale))
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

        # 가로보는 그리지 않는다. 젓가락 자체가 마스트에서 뻗어 나오므로
        # 뒤에 수평 트러스를 하나 더 두면 의미 없이 겹쳐 보인다.
        self._truss_mast(mast_x, mast_top)

        # 실제 포획 판정 범위(±zone_r). 이걸 따로 그리지 않으면 학생은
        # 팔 안쪽으로 잘 지나간 것처럼 보이는데 MISSED 가 뜨는 이유를
        # 알 수 없다. 디버깅하라고 만든 그림이 디버깅을 방해하게 된다.
        # 실제 포획 판정 범위(±zone_r)를 얇게 표시한다. 구조물처럼 굵게
        # 그리면 젓가락과 헷갈리지만, 아예 지우면 학생이 "팔 안쪽으로
        # 지나간 것 같은데 왜 MISSED 인지"를 알 수 없다.
        pygame.draw.line(self.surface, ARM_COLOR,
                         self._to_px(x_tower - window_half, y_arm),
                         self._to_px(x_tower + window_half, y_arm),
                         self._scaled_width(2.5, min_px=1))

        # 뒤쪽 팔만 여기서 그린다. 앞쪽 팔은 로켓을 그린 뒤에 덧그려야
        # 로켓이 두 팔 사이에 물린 것처럼 보인다 — 한 겹으로 그리면
        # 팔이 기체를 관통한 모습이 된다.
        self._draw_jaws(x_tower, y_arm, window_half, grip, front=False)

    def _truss_mast(self, mast_x: float, top: float) -> None:
        """수직 트러스. 기둥 하나로 그리면 깃대처럼 보이고, 실제 발사탑은
        두 줄기 사이를 X 브레이스로 엮은 격자 구조다."""
        rail = self._scaled_width(4.0, min_px=2)
        brace = self._scaled_width(2.0, min_px=1)
        left, right = mast_x - TRUSS_HALF_W, mast_x + TRUSS_HALF_W
        for rx in (left, right):
            pygame.draw.line(self.surface, TOWER_COLOR,
                             self._to_px(rx, 0.0), self._to_px(rx, top), rail)
        bays = max(6, int(top / TRUSS_BAY))
        for i in range(bays):
            y0 = top * i / bays
            y1 = top * (i + 1) / bays
            pygame.draw.line(self.surface, TOWER_COLOR,
                             self._to_px(left, y1), self._to_px(right, y1), brace)
            pygame.draw.line(self.surface, TOWER_COLOR,
                             self._to_px(left, y0), self._to_px(right, y1), brace)
            pygame.draw.line(self.surface, TOWER_COLOR,
                             self._to_px(right, y0), self._to_px(left, y1), brace)

    def _truss_beam(self, x0: float, x1: float, y: float) -> None:
        """수평 트러스 가로보. 위아래 두 줄기와 지그재그 브레이스."""
        rail = self._scaled_width(3.5, min_px=2)
        brace = self._scaled_width(2.0, min_px=1)
        top, bot = y + TRUSS_HALF_W * 0.8, y - TRUSS_HALF_W * 0.8
        for yy in (top, bot):
            pygame.draw.line(self.surface, TOWER_COLOR,
                             self._to_px(x0, yy), self._to_px(x1, yy), rail)
        bays = max(3, int(abs(x1 - x0) / TRUSS_BAY))
        for i in range(bays):
            a = x0 + (x1 - x0) * i / bays
            b = x0 + (x1 - x0) * (i + 1) / bays
            lo, hi = (bot, top) if i % 2 == 0 else (top, bot)
            pygame.draw.line(self.surface, TOWER_COLOR,
                             self._to_px(a, lo), self._to_px(b, hi), brace)

    def _truss_line(self, x0: float, y0: float, x1: float, y1: float,
                    half_w: float, bay: float, color) -> None:
        """임의 방향 트러스. 두 줄기와 지그재그 브레이스로 엮는다."""
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return
        # 진행 방향의 법선으로 줄기를 벌린다
        nx, ny = -dy / length * half_w, dx / length * half_w
        rail = self._scaled_width(2.2, min_px=1)
        brace = self._scaled_width(1.4, min_px=1)
        for sign in (-1, 1):
            pygame.draw.line(
                self.surface, color,
                self._to_px(x0 + sign * nx, y0 + sign * ny),
                self._to_px(x1 + sign * nx, y1 + sign * ny), rail)
        bays = max(3, int(length / bay))
        for i in range(bays):
            ta, tb = i / bays, (i + 1) / bays
            ax, ay = x0 + dx * ta, y0 + dy * ta
            bx, by = x0 + dx * tb, y0 + dy * tb
            s0, s1 = (1, -1) if i % 2 == 0 else (-1, 1)
            pygame.draw.line(
                self.surface, color,
                self._to_px(ax + s0 * nx, ay + s0 * ny),
                self._to_px(bx + s1 * nx, by + s1 * ny), brace)

    def _draw_jaws(self, x_tower: float, y_arm: float, window_half: float,
                   grip: float, *, front: bool) -> None:
        """마스트의 피봇에서 회전하는 젓가락 한 짝.

        평행이동이 아니라 회전이다 — 피봇은 타워 쪽에 고정돼 있고, 팁만
        위아래로 갈라졌다 모인다. 벌어지면 로켓이 들어올 틈이 생기고,
        조이면 기체를 사이에 문다. 본체 트러스보다 가늘게 그려 구조물과
        구분되게 한다.
        """
        pivot_x = x_tower - TOWER_OFFSET
        tip_x = x_tower + max(window_half * 1.4, 12.0)
        spread = JAW_OPEN_SPREAD + (JAW_CLOSE_SPREAD - JAW_OPEN_SPREAD) * grip
        sign = -1.0 if front else 1.0
        color = JAW_COLOR if front else JAW_BACK_COLOR
        base_y = y_arm + JAW_Y_OFFSET
        self._truss_line(pivot_x, base_y, tip_x, base_y + sign * spread,
                         JAW_TRUSS_HALF, JAW_TRUSS_BAY, color)

    def _front_arm(self, target: tuple[float, float], grip: float) -> None:
        """앞쪽 팔. 로켓을 그린 뒤 호출해 기체 위에 덮는다."""
        if self.cfg["task"] != "catch":
            return
        x_tower, y_arm, _, window_half = self._catch_geometry(target)
        self._draw_jaws(x_tower, y_arm, window_half, grip, front=True)

    def _scaled_width(self, base_px: float, min_px: int = 2) -> int:
        """카메라 배율에 맞춰 선 굵기를 조정한다.

        MIN_SCALE(세계 전체가 보이는 최소 배율)에서 base_px로 보이도록
        맞춘 값이라, 줌인할수록 그만큼 굵어져야 로켓 크기 대비 두께가
        일정하게 유지된다. 줌인 폭이 커진 지금은 고정 px로는 하이라인이
        되어버려서 최소값으로 하한을 둔다.
        """
        _, _, scale = self._cam
        return max(min_px, int(round(base_px * scale / MIN_SCALE)))

    def _approach_grip(self, state: State, target: tuple[float, float]) -> float:
        """로켓이 팔에 다가온 정도(0~1). 집게가 오므라드는 양이다.

        수직으로 가까울수록, 그리고 수평으로 대충 정렬돼 있을 때만 닫는다 —
        멀찍이 빗나가는 로켓에도 집게가 닫히면 어색하다.
        """
        if self.cfg["task"] != "catch":
            return 0.0
        x_tower, y_arm = target
        window_half = self.cfg["success"]["zone_r"]
        if abs(state.x - x_tower) > window_half * GRIP_ALIGN_RANGE:
            return 0.0
        above = state.y - y_arm
        raw = min(max(1.0 - above / GRIP_APPROACH_RANGE, 0.0), 1.0)
        return raw * GRIP_APPROACH_MAX

    def _catch_geometry(self, target: tuple[float, float]):
        """타워 x, 팔 높이, 팔 반폭, 실제 포획 창 반폭을 돌려준다.

        구조물 폭과 판정 폭을 한곳에서 분리해 두어야, 렌더러가 임계값을
        잘못 그리는 일이 생기지 않고 테스트로 고정할 수도 있다.
        """
        x_tower, y_arm = target
        window_half = self.cfg["success"]["zone_r"]
        return x_tower, y_arm, max(window_half * 1.4, 12.0), window_half

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
            drag = 0.88 if p["kind"] == "smoke" else 0.97
            p["vx"] *= drag
            p["vy"] *= drag
            p["age"] += DT
        self._particles = [p for p in self._particles
                           if p["age"] <= p["life"]][-MAX_PARTICLES:]

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
                "age": 0.0, "kind": "smoke", "life": PARTICLE_LIFE,
            })

        # 부스러기: 훨씬 빠르고 넓게 튀며 금방 사라진다. 연기만 있으면
        # 배기가 뭉근해 보여 추력의 격렬함이 전달되지 않는다.
        for _ in range(max(1, n // 2)):
            speed = self._rng.uniform(110.0, 190.0)
            angle = base_angle + self._rng.uniform(-0.75, 0.75)
            self._particles.append({
                "x": nozzle_x, "y": nozzle_y,
                "vx": speed * math.cos(angle), "vy": speed * math.sin(angle),
                "age": 0.0, "kind": "debris", "life": DEBRIS_LIFE,
            })

    def _draw_particles(self) -> None:
        if not self._particles:
            return
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        _, _, scale = self._cam
        for p in self._particles:
            frac = p["age"] / p["life"]
            if p["kind"] == "smoke":
                radius_px = max(1, int((2.0 + 14.0 * frac) * scale))
                color = (*SMOKE_COLOR, int(170 * (1.0 - frac) ** 1.5))
            else:
                # 부스러기는 커지지 않고 작게 유지되며 빠르게 흐려진다
                radius_px = max(1, int(1.2 * scale))
                color = (*DEBRIS_COLOR, int(240 * (1.0 - frac)))
            pygame.draw.circle(overlay, color,
                               self._to_px(p["x"], p["y"]), radius_px)
        self.surface.blit(overlay, (0, 0))

    # --- 로켓 ---

    def _rocket(self, state: State) -> None:
        """Starship 부스터를 흉내낸 원통형 기체.

        뾰족한 노즈콘이 아니라 위가 평평한 원통이고, 상단 옆으로 걸림
        구조(리프트 포인트)가 튀어나와 있다 — 젓가락이 그 돌기에 걸려
        기체를 든다. 판정 자체는 물리가 하지만, 어디에 걸리는지가 보여야
        학생이 포획을 이해한다.
        """
        half = ROCKET_HEIGHT / 2.0
        poly = self._body_to_px

        # 엔진 스커트: 아래가 살짝 넓다
        skirt = [(-BODY_HALF_W - 1.2, -half), (BODY_HALF_W + 1.2, -half),
                 (BODY_HALF_W, -half + 5.0), (-BODY_HALF_W, -half + 5.0)]
        pygame.draw.polygon(self.surface, FIN_COLOR,
                            [poly(state, bx, by) for bx, by in skirt])

        # 원통 몸통. 꼭대기는 평평하되 모서리만 살짝 깎는다.
        body = [(-BODY_HALF_W, -half + 4.0), (BODY_HALF_W, -half + 4.0),
                (BODY_HALF_W, half - 1.5), (BODY_HALF_W - 1.4, half),
                (-BODY_HALF_W + 1.4, half), (-BODY_HALF_W, half - 1.5)]
        pygame.draw.polygon(self.surface, BODY_COLOR,
                            [poly(state, bx, by) for bx, by in body])

        # 걸림 구조(리프트 포인트) — 젓가락이 여기에 걸린다
        for sign in (-1, 1):
            pin = [(sign * BODY_HALF_W, PIN_Y - PIN_THICK),
                   (sign * (BODY_HALF_W + PIN_OUT), PIN_Y - PIN_THICK),
                   (sign * (BODY_HALF_W + PIN_OUT), PIN_Y + PIN_THICK),
                   (sign * BODY_HALF_W, PIN_Y + PIN_THICK)]
            pygame.draw.polygon(self.surface, PIN_COLOR,
                                [poly(state, bx, by) for bx, by in pin])

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
        """추력 단계에 따라 길이·굵기·색이 뚜렷이 달라지는 화염.

        행동 테이블의 추력은 0 / 0.2G / 1.0G / 2.0G 네 단계다. 길이를 선형으로
        잡으면 0.2G와 1.0G가 비슷해 보여 단계가 구분되지 않는다. 지수를
        0.75로 눌러 낮은 단계도 짧게, 최대 단계는 확실히 길게 만든다.
        """
        if state.thrust <= 0.0:
            return
        ratio = state.thrust / (2.0 * G)
        length = 5.0 + 40.0 * ratio ** 0.75
        half_w = 1.6 + 2.4 * ratio
        # 약한 분사는 노란빛, 강한 분사는 흰빛이 도는 주황
        color = ((255, 214, 92) if ratio < 0.3
                 else (255, 158, 66) if ratio < 0.7 else (255, 236, 190))
        base = -ROCKET_HEIGHT / 2.0
        tip = (length * math.sin(state.phi),
               base - length * math.cos(state.phi))
        pygame.draw.polygon(self.surface, color, [
            self._body_to_px(state, -half_w, base),
            self._body_to_px(state, half_w, base),
            self._body_to_px(state, tip[0], tip[1]),
        ])

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
        # 석양 배경은 밝아서 흰 글씨가 묻힌다. 반투명 판을 먼저 깐다.
        panel = pygame.Surface((330, 20 + len(lines) * 19), pygame.SRCALPHA)
        panel.fill((10, 12, 24, 150))
        self.surface.blit(panel, (6, 6))
        for i, line in enumerate(lines):
            self.surface.blit(
                self.font.render(line, True, HUD_COLOR), (12, 12 + i * 19))

    def _banner(self, outcome: str) -> None:
        text, color = BANNER_TEXT[outcome]
        surface = self.banner_font.render(text, True, color)
        rect = surface.get_rect(center=(WIDTH // 2, HEIGHT // 3))
        self.surface.blit(surface, rect)
