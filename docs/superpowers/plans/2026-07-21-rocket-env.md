# rocket-env Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** YCS1003의 highway-env를 대체할, Gymnasium 호환 로켓 착륙/젓가락 포획 환경 패키지 `rocket_env`를 만든다.

**Architecture:** 계층 분리 — `physics.py`(순수 동역학) / `wind.py`(OU 바람) / `config.py`(설정·프리셋·검증) / `tasks/`(태스크별 성공 판정) / `reward.py`(PBRS + 종료 보상) / `env.py`(Gymnasium 파사드) / `render.py`(pygame). 각 계층은 아래 계층만 알고 위 계층을 모른다.

**Tech Stack:** Python 3.12(개발) / 3.10+(지원), Gymnasium, NumPy, pygame, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-07-21-rocket-env-design.md`

## Global Constraints

- **런타임 의존성은 정확히 3개**: `gymnasium>=0.29`, `numpy>=1.24`, `pygame>=2.5`. `torch`/`stable-baselines3`는 **절대 런타임 의존성에 넣지 않는다** (optional extra `sb3`로만).
- **Python `requires-python = ">=3.10"`**. 개발 venv는 `python3.12`(서버 워커와 동일).
- **라이선스 MIT.** 원본 `jiupinjia/rocket-recycling`의 코드·이미지 에셋을 **일절 복사하지 않는다.** 물리 방정식만 스펙에서 재작성한다.
- **정규화 상수는 전부 모듈 레벨 하드코딩**이며 config에서 파생하지 않는다 (`300.0`, `50.0`, `π/2`, `PHI_MAX`, `WIND_OBS_SCALE=20.0`).
- **패키지명 `rocket_env`**, 환경 id `rocket-v0` / `rocket-landing-v0` / `rocket-catch-v0`.
- **`shaping_gamma` 기본값은 `1.0`** (평가 설정 고정값). γ<1이면 shaping 총합에 에피소드 길이 편향이 생긴다.
- **식별자·타입명은 영어, docstring과 주석은 한국어.** 학생이 읽는 코드이며 submit-server 코드베이스 관례와 일치한다.
- **모든 작업은 브랜치 `feat/rocket-env`에서** 진행한다.
- 커밋 메시지 말미에 `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` 를 붙인다.

## File Structure

| 파일 | 책임 |
|------|------|
| `pyproject.toml` | 패키지 메타데이터, 의존성, pytest 설정 |
| `LICENSE` | MIT |
| `README.md` | 설치·사용법·출처 표기 |
| `rocket_env/types.py` | `State` 데이터클래스, `Outcome` 상수 |
| `rocket_env/physics.py` | 물리 상수, `ACTION_TABLE`, `integrate()`, `fuel_cost()` |
| `rocket_env/wind.py` | `WindProcess` (OU 과정) |
| `rocket_env/config.py` | `DEFAULT_CONFIG`, `CATCH_OVERRIDES`, `PRESETS`, `build_config()`, `validate_train_config()`, `ConfigError` |
| `rocket_env/tasks/base.py` | `Task` 프로토콜, 공용 경계 검사 |
| `rocket_env/tasks/landing.py` | `LandingTask` |
| `rocket_env/tasks/catch.py` | `CatchTask` |
| `rocket_env/tasks/__init__.py` | `make_task()` 팩토리 |
| `rocket_env/reward.py` | `potential()`, `shaping()`, `terminal_reward()` |
| `rocket_env/env.py` | `RocketEnv(gym.Env)` |
| `rocket_env/render.py` | `Renderer` (pygame) |
| `rocket_env/__init__.py` | gymnasium 환경 등록 |

---

### Task 1: 패키지 스캐폴딩 + 물리 엔진

**Files:**
- Create: `pyproject.toml`, `rocket_env/__init__.py`, `rocket_env/types.py`, `rocket_env/physics.py`
- Test: `tests/test_physics.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `rocket_env.types.State` — frozen dataclass, 필드 `x, y, vx, vy, theta, omega, phi, thrust, fuel, wind_x, step`
  - `rocket_env.types.Outcome` — 상수 `IN_PROGRESS, SUCCESS, CRASH, MISSED, TIMEOUT, OUT_OF_FUEL`
  - `rocket_env.physics.integrate(state: State, thrust: float, nozzle_rate: float, wind_x: float) -> State`
  - `rocket_env.physics.fuel_cost(thrust: float) -> float`
  - 상수 `G, DT, ROCKET_HEIGHT, MOMENT_OF_INERTIA, TERMINAL_VELOCITY, DRAG_RHO, PHI_MAX, ACTION_TABLE, WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX`

- [ ] **Step 1: 브랜치 생성과 개발 환경 준비**

```bash
git checkout -b feat/rocket-env
uv venv --python 3.12
uv pip install gymnasium numpy pygame pytest
```

기대 출력: `.venv` 생성 후 4개 패키지 설치 완료.

- [ ] **Step 2: `pyproject.toml` 작성**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "rocket-env"
version = "0.1.0"
description = "Gymnasium-compatible rocket landing and tower-catch environment for RL coursework"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
dependencies = [
    "gymnasium>=0.29",
    "numpy>=1.24",
    "pygame>=2.5",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]
sb3 = ["stable-baselines3>=2.3", "torch"]

[tool.hatch.build.targets.wheel]
packages = ["rocket_env"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: 학습을 수반하는 느린 통합 테스트"]
```

- [ ] **Step 3: 빈 패키지를 만들고 설치 확인**

`rocket_env/__init__.py` 를 빈 파일로 생성한다 (환경 등록은 Task 7에서 추가).

```bash
mkdir -p rocket_env tests
touch rocket_env/__init__.py tests/__init__.py
uv pip install -e .
uv run python -c "import rocket_env; print('ok')"
```

기대 출력: `ok`

- [ ] **Step 4: 실패하는 물리 테스트 작성**

`tests/test_physics.py`:

```python
"""물리 엔진 검증.

물리는 태스크·보상과 무관한 고정 계층이므로, 여기 테스트가 깨지면
환경 전체의 동작이 바뀐 것이다.
"""

import math

import pytest

from rocket_env.physics import (
    ACTION_TABLE,
    DT,
    G,
    PHI_MAX,
    ROCKET_HEIGHT,
    TERMINAL_VELOCITY,
    fuel_cost,
    integrate,
)
from rocket_env.types import State


def make_state(**kw) -> State:
    base = dict(
        x=0.0, y=100.0, vx=0.0, vy=0.0, theta=0.0, omega=0.0,
        phi=0.0, thrust=0.0, fuel=math.inf, wind_x=0.0, step=0,
    )
    base.update(kw)
    return State(**base)


def test_action_table_has_12_entries_in_thrust_major_order():
    assert len(ACTION_TABLE) == 12
    assert ACTION_TABLE[0] == (0.0, -math.radians(30.0))
    assert ACTION_TABLE[1] == (0.0, 0.0)
    assert ACTION_TABLE[11] == (2.0 * G, math.radians(30.0))


def test_single_freefall_step_matches_hand_computation():
    s = integrate(make_state(), thrust=0.0, nozzle_rate=0.0, wind_x=0.0)
    assert s.y == pytest.approx(100.0 - 0.5 * G * DT**2)
    assert s.vy == pytest.approx(-G * DT)
    assert s.step == 1


def test_freefall_reaches_the_designed_terminal_velocity():
    """시뮬레이션이 설계값 TERMINAL_VELOCITY에 실제로 도달하는지 본다.

    DRAG_RHO를 종단속도에서 역산했으므로, 이 테스트는 계수 계산과 적분이
    서로 맞물려 돌아가는지 확인하는 독립적 검증이 된다.
    """
    s = make_state(y=100_000.0)
    for _ in range(2000):
        s = integrate(s, thrust=0.0, nozzle_rate=0.0, wind_x=0.0)
    assert s.vy == pytest.approx(-TERMINAL_VELOCITY, abs=0.01)


def test_drag_vanishes_when_moving_with_the_wind():
    """항력은 공기 기준 상대속도에 비례하므로 바람과 같은 속도면 0이다."""
    s = integrate(make_state(vx=10.0), thrust=0.0, nozzle_rate=0.0, wind_x=10.0)
    assert s.vx == pytest.approx(10.0)


def test_zero_wind_still_decelerates_horizontal_motion():
    s = integrate(make_state(vx=10.0), thrust=0.0, nozzle_rate=0.0, wind_x=0.0)
    assert s.vx < 10.0


def test_upright_full_thrust_gives_net_upward_acceleration_of_g():
    s = integrate(make_state(), thrust=2.0 * G, nozzle_rate=0.0, wind_x=0.0)
    assert s.vy == pytest.approx(G * DT)


def test_gimballed_thrust_produces_torque():
    """얇은 막대(I = H^2/12)의 H/2 지점에 접선력이 걸리면 alpha = 6*ft/H 다.

    프로덕션 코드와 다른 대수 경로로 유도했으므로 독립적인 오라클이다.
    프로덕션 수식을 그대로 재계산하면 지렛대 길이나 관성모멘트를 함께
    잘못 잡은 경우를 잡아낼 수 없다.
    """
    phi = math.radians(10.0)
    s = integrate(make_state(phi=phi), thrust=G, nozzle_rate=0.0, wind_x=0.0)
    thrust_tangential = -G * math.sin(phi)
    expected_alpha = 6.0 * thrust_tangential / ROCKET_HEIGHT
    assert s.omega == pytest.approx(expected_alpha * DT)
    assert s.omega < 0.0


def test_nozzle_angle_is_clipped_to_twenty_degrees():
    s = make_state()
    for _ in range(50):
        s = integrate(s, thrust=0.0, nozzle_rate=math.radians(30.0), wind_x=0.0)
    assert s.phi == pytest.approx(PHI_MAX)


def test_integrate_records_applied_thrust():
    s = integrate(make_state(), thrust=1.5 * G, nozzle_rate=0.0, wind_x=0.0)
    assert s.thrust == pytest.approx(1.5 * G)


def test_fuel_cost_is_one_unit_per_g_second():
    assert fuel_cost(G) == pytest.approx(DT)
    assert fuel_cost(0.0) == 0.0
    assert fuel_cost(2.0 * G) == pytest.approx(2.0 * DT)
```

- [ ] **Step 5: 테스트가 실패하는지 확인**

```bash
uv run pytest tests/test_physics.py -v
```

기대: `ModuleNotFoundError: No module named 'rocket_env.physics'` 로 collection 실패.

- [ ] **Step 6: `rocket_env/types.py` 구현**

```python
"""환경의 상태 표현.

State는 frozen dataclass다. 물리 적분이 새 State를 반환하는 방식이라
이전 스텝의 상태가 실수로 덮어써지지 않는다 — 젓가락 포획 판정이
'이전 스텝'과 '현재 스텝'을 비교하기 때문에 이 불변성이 중요하다.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class State:
    x: float           # 수평 위치 (m)
    y: float           # 고도, 로켓 중심 기준 (m)
    vx: float          # 수평 속도 (m/s)
    vy: float          # 수직 속도 (m/s)
    theta: float       # 기체 자세각, 수직 기준 (rad)
    omega: float       # 각속도 (rad/s)
    phi: float         # 노즐 짐벌각 (rad)
    thrust: float      # 직전 스텝에 실제 적용된 추력 (m/s^2). 렌더링용
    fuel: float        # 잔여 연료 (단위). 무한 연료면 math.inf
    wind_x: float      # 현재 수평 바람 (m/s)
    step: int          # 경과 스텝 수


class Outcome:
    """에피소드 종료 사유. 문자열 상수를 쓰는 이유는 그대로 info dict에
    실려 서버·리더보드로 전달되기 때문이다 (JSON 직렬화 가능)."""

    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    CRASH = "crash"
    MISSED = "missed"
    TIMEOUT = "timeout"
    OUT_OF_FUEL = "out_of_fuel"
```

- [ ] **Step 7: `rocket_env/physics.py` 구현**

```python
"""강체 로켓 동역학.

이 모듈은 태스크와 보상을 전혀 모른다. 물리는 고정이고 보상은 설계 선택이라는
구분을 파일 경계로 드러내기 위한 것이다.

모델: 얇은 막대 강체 + 짐벌 노즐 + 속도에 비례하는 항력.
"""

import math
from dataclasses import replace

from rocket_env.types import State

# --- 물리 상수 (config로 노출하지 않는 잠금 값) ---
G = 9.8                                    # 중력가속도 (m/s^2)
DT = 0.05                                  # 시뮬레이션 시간 간격 (s)
ROCKET_HEIGHT = 50.0                       # 기체 길이 (m)
MOMENT_OF_INERTIA = ROCKET_HEIGHT**2 / 12.0  # 얇은 막대의 관성모멘트 (단위질량)
PHI_MAX = math.radians(20.0)               # 노즐 짐벌 한계 (rad)

# 항력 계수는 종단속도를 설계값으로 두고 역산한다. 무동력 낙하가 평형에
# 이르면 DRAG_RHO * v = G 이므로 DRAG_RHO = G / v_term.
# 계수 자체는 물리 법칙이 정해주지 않는 설계 선택이므로, 의미가 바로 읽히는
# 양(종단속도)으로 고르는 편이 학생에게도 검증하기 쉽다.
TERMINAL_VELOCITY = 49.5                   # 무동력 낙하 종단속도 (m/s)
DRAG_RHO = G / TERMINAL_VELOCITY

# --- 세계 경계 ---
WORLD_X_MIN, WORLD_X_MAX = -300.0, 300.0
WORLD_Y_MIN, WORLD_Y_MAX = 0.0, 570.0

# --- 행동 테이블 ---
# 추력 0을 포함하는 것이 핵심이다. 엔진을 끌 수 있어야 연료 절약이 전략이 된다.
THRUST_LEVELS = (0.0, 0.2 * G, 1.0 * G, 2.0 * G)
NOZZLE_RATES = (-math.radians(30.0), 0.0, math.radians(30.0))

# 인덱스 = thrust_idx * 3 + nozzle_idx
ACTION_TABLE: tuple[tuple[float, float], ...] = tuple(
    (f, rate) for f in THRUST_LEVELS for rate in NOZZLE_RATES
)


def fuel_cost(thrust: float) -> float:
    """이번 스텝의 연료 소모량. 1단위 = 1G 추력으로 1초 분사."""
    return (thrust / G) * DT


def integrate(state: State, thrust: float, nozzle_rate: float,
              wind_x: float) -> State:
    """한 스텝 적분한 새 State를 반환한다.

    추력을 기체 기준 두 성분으로 나눈 뒤 세계 좌표로 회전시킨다.
    접선 성분만 토크를 만들고, 축 성분만 기체를 밀어올린다.

    항력은 지면 기준 속도가 아니라 **공기 기준 상대속도**에 비례한다.
    바람이 새로운 힘 항이 아니라 기존 항력 항의 수정으로 들어가는 이유다.
    """
    theta, phi = state.theta, state.phi

    thrust_tangential = -thrust * math.sin(phi)   # 옆 방향 성분 → 토크
    thrust_axial = thrust * math.cos(phi)         # 기체 축 방향 성분 → 추진

    fx = thrust_tangential * math.cos(theta) - thrust_axial * math.sin(theta)
    fy = thrust_tangential * math.sin(theta) + thrust_axial * math.cos(theta)

    ax = fx - DRAG_RHO * (state.vx - wind_x)
    ay = fy - G - DRAG_RHO * state.vy
    alpha = thrust_tangential * (ROCKET_HEIGHT / 2.0) / MOMENT_OF_INERTIA

    return replace(
        state,
        x=state.x + state.vx * DT + 0.5 * ax * DT**2,
        y=state.y + state.vy * DT + 0.5 * ay * DT**2,
        vx=state.vx + ax * DT,
        vy=state.vy + ay * DT,
        theta=state.theta + state.omega * DT + 0.5 * alpha * DT**2,
        omega=state.omega + alpha * DT,
        phi=min(max(phi + nozzle_rate * DT, -PHI_MAX), PHI_MAX),
        thrust=thrust,
        step=state.step + 1,
    )
```

- [ ] **Step 8: 테스트 통과 확인**

```bash
uv run pytest tests/test_physics.py -v
```

기대: 10 passed.

- [ ] **Step 9: 커밋**

```bash
git add pyproject.toml rocket_env/ tests/
git commit -m "$(cat <<'EOF'
feat: 패키지 스캐폴딩과 강체 물리 엔진

State/Outcome 타입과 순수 함수 integrate()를 추가. 항력을 공기 기준
상대속도에 비례시켜 바람이 별도 힘 항 없이 들어가도록 했다.
추력 0을 행동 테이블에 포함해 연료 절약이 전략이 되게 했다.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: 바람 프로세스

**Files:**
- Create: `rocket_env/wind.py`
- Test: `tests/test_wind.py`

**Interfaces:**
- Consumes: `rocket_env.physics.DT`
- Produces:
  - `rocket_env.wind.WindProcess(mode: str, max_speed: float, ou_theta: float, ou_sigma: float)`
  - `.reset(rng: numpy.random.Generator) -> float`
  - `.step(wind_x: float, rng: numpy.random.Generator) -> float`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_wind.py`:

```python
"""OU 바람 프로세스 검증."""

import numpy as np
import pytest

from rocket_env.wind import WindProcess


def gust(max_speed=12.0, ou_theta=0.15, ou_sigma=3.0) -> WindProcess:
    return WindProcess(mode="gust", max_speed=max_speed,
                       ou_theta=ou_theta, ou_sigma=ou_sigma)


def test_none_mode_is_always_zero():
    w = WindProcess(mode="none", max_speed=0.0, ou_theta=0.0, ou_sigma=0.0)
    rng = np.random.default_rng(0)
    value = w.reset(rng)
    assert value == 0.0
    for _ in range(100):
        value = w.step(value, rng)
        assert value == 0.0


def test_constant_mode_holds_its_reset_value():
    w = WindProcess(mode="constant", max_speed=8.0, ou_theta=0.0, ou_sigma=0.0)
    rng = np.random.default_rng(1)
    value = w.reset(rng)
    assert -8.0 <= value <= 8.0
    for _ in range(100):
        assert w.step(value, rng) == pytest.approx(value)


def test_gust_mode_actually_varies():
    w = gust()
    rng = np.random.default_rng(2)
    value = w.reset(rng)
    series = [value := w.step(value, rng) for _ in range(200)]
    assert np.std(series) > 0.1


def test_gust_stays_within_max_speed():
    w = gust(max_speed=5.0, ou_sigma=50.0)
    rng = np.random.default_rng(3)
    value = w.reset(rng)
    for _ in range(500):
        value = w.step(value, rng)
        assert -5.0 <= value <= 5.0


def test_same_seed_gives_identical_wind_sequence():
    def run(seed):
        w = gust()
        rng = np.random.default_rng(seed)
        value = w.reset(rng)
        return [value := w.step(value, rng) for _ in range(50)]

    assert run(7) == run(7)
    assert run(7) != run(8)


def test_mean_reversion_pulls_toward_zero():
    """ou_sigma=0이면 결정론적 평균회귀만 남아 0으로 수렴한다."""
    w = WindProcess(mode="gust", max_speed=20.0, ou_theta=5.0, ou_sigma=0.0)
    rng = np.random.default_rng(4)
    value = 10.0
    for _ in range(500):
        value = w.step(value, rng)
    assert abs(value) < 0.1
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_wind.py -v
```

기대: `ModuleNotFoundError: No module named 'rocket_env.wind'`

- [ ] **Step 3: `rocket_env/wind.py` 구현**

```python
"""수평 바람의 Ornstein–Uhlenbeck 과정.

세 가지 모드가 모두 하나의 수식으로 표현된다:
  - "none"     : max_speed = 0  → 클리핑 결과가 항상 0
  - "constant" : ou_theta = ou_sigma = 0 → reset 값이 그대로 유지
  - "gust"     : 평균회귀 + 확률적 변동

분기 없는 한 경로라서, 모드를 바꿔도 검증할 코드가 늘지 않는다.
"""

import math

import numpy as np

from rocket_env.physics import DT


class WindProcess:
    def __init__(self, mode: str, max_speed: float,
                 ou_theta: float, ou_sigma: float):
        self.mode = mode
        self.max_speed = float(max_speed)
        self.ou_theta = float(ou_theta)
        self.ou_sigma = float(ou_sigma)

    def reset(self, rng: np.random.Generator) -> float:
        """에피소드 시작 바람. max_speed=0이면 0.0."""
        return float(rng.uniform(-self.max_speed, self.max_speed))

    def step(self, wind_x: float, rng: np.random.Generator) -> float:
        """한 스텝 진행한 바람 값."""
        drift = self.ou_theta * (0.0 - wind_x) * DT
        noise = self.ou_sigma * math.sqrt(DT) * rng.standard_normal()
        value = wind_x + drift + noise
        return float(min(max(value, -self.max_speed), self.max_speed))
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/test_wind.py -v
```

기대: 6 passed.

- [ ] **Step 5: 커밋**

```bash
git add rocket_env/wind.py tests/test_wind.py
git commit -m "$(cat <<'EOF'
feat: OU 기반 바람 프로세스

none/constant/gust 세 모드를 분기 없는 단일 수식으로 표현.
max_speed=0이면 클리핑이 none 모드를, ou_theta=ou_sigma=0이면
constant 모드를 자동으로 만든다.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Config 시스템

**Files:**
- Create: `rocket_env/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: 없음 (순수 딕셔너리 조작)
- Produces:
  - `rocket_env.config.ConfigError(ValueError)`
  - `rocket_env.config.DEFAULT_CONFIG: dict`
  - `rocket_env.config.CATCH_OVERRIDES: dict`
  - `rocket_env.config.PRESETS: dict[str, dict]`
  - `rocket_env.config.LOCKED_KEYS: frozenset[str]`
  - `rocket_env.config.build_config(user_config: dict | None) -> dict`
  - `rocket_env.config.validate_train_config(train_cfg: dict, eval_cfg: dict) -> tuple[bool, list[str], list[str]]` — `(ok, warnings, errors)`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_config.py`:

```python
"""Config 병합·검증·프리셋."""

import pytest

from rocket_env.config import (
    DEFAULT_CONFIG,
    PRESETS,
    ConfigError,
    build_config,
    validate_train_config,
)


def test_build_config_with_none_returns_defaults():
    cfg = build_config(None)
    assert cfg["task"] == "landing"
    assert cfg["max_steps"] == 800
    assert cfg["reward"]["shaping_gamma"] == 1.0


def test_build_config_does_not_mutate_defaults():
    build_config({"max_steps": 42})
    assert DEFAULT_CONFIG["max_steps"] == 800


def test_partial_nested_override_keeps_sibling_defaults():
    cfg = build_config({"reward": {"success_base": 200.0}})
    assert cfg["reward"]["success_base"] == 200.0
    assert cfg["reward"]["w_speed"] == 40.0


def test_catch_task_swaps_in_catch_profile():
    cfg = build_config({"task": "catch"})
    assert cfg["success"]["v_max"] == 5.0
    assert cfg["success"]["zone_r"] == 6.0
    assert cfg["reward"]["w_speed"] == 60.0
    assert cfg["reward"]["v_ref"] == 2.0


def test_explicit_user_value_beats_catch_profile():
    cfg = build_config({"task": "catch", "reward": {"w_speed": 5.0}})
    assert cfg["reward"]["w_speed"] == 5.0
    assert cfg["reward"]["v_ref"] == 2.0


def test_locked_key_raises_config_error():
    for key in ("dt", "g", "H", "observation", "action"):
        with pytest.raises(ConfigError, match=key):
            build_config({key: 1})


def test_unknown_task_raises_config_error():
    with pytest.raises(ConfigError, match="task"):
        build_config({"task": "hover"})


def test_negative_max_steps_raises_config_error():
    with pytest.raises(ConfigError, match="max_steps"):
        build_config({"max_steps": 0})


def test_negative_fuel_capacity_raises_config_error():
    with pytest.raises(ConfigError, match="capacity"):
        build_config({"fuel": {"capacity": -1.0}})


def test_none_fuel_capacity_is_allowed():
    assert build_config({"fuel": {"capacity": None}})["fuel"]["capacity"] is None


@pytest.mark.parametrize("name", [
    "landing-easy", "landing-normal", "landing-hard",
    "catch-normal", "catch-hard",
])
def test_every_preset_builds(name):
    cfg = build_config(PRESETS[name])
    assert cfg["task"] in ("landing", "catch")
    assert cfg["reward"]["shaping_gamma"] == 1.0


def test_unknown_top_level_key_raises_config_error():
    with pytest.raises(ConfigError, match="wnid"):
        build_config({"wnid": {"mode": "none"}})


def test_unknown_nested_key_raises_config_error():
    """오타 난 키가 조용히 무시되면 설정 없이 학습이 끝난다."""
    with pytest.raises(ConfigError, match="fuel.capacty"):
        build_config({"fuel": {"capacty": 50.0}})


@pytest.mark.parametrize("name,path,expected", [
    ("landing-easy", ("wind", "max_speed"), 0.0),
    ("landing-easy", ("fuel", "capacity"), None),
    ("landing-normal", ("wind", "max_speed"), 8.0),
    ("landing-normal", ("fuel", "capacity"), 120.0),
    ("landing-hard", ("wind", "ou_sigma"), 3.0),
    ("landing-hard", ("fuel", "capacity"), 90.0),
    ("landing-hard", ("success", "zone_r"), 30.0),
    ("catch-normal", ("fuel", "capacity"), 140.0),
    ("catch-normal", ("success", "zone_r"), 6.0),
    ("catch-normal", ("reward", "w_speed"), 60.0),
    ("catch-hard", ("wind", "max_speed"), 12.0),
    ("catch-hard", ("fuel", "capacity"), 110.0),
])
def test_preset_literal_values(name, path, expected):
    """프리셋 리터럴을 고정한다.

    이 값들이 각 라운드의 난이도와 배점을 정한다. 자리 하나가 바뀌어도
    나머지 테스트는 전부 통과하므로, 값 자체를 단언하는 곳이 필요하다.
    """
    value = build_config(PRESETS[name])
    for key in path:
        value = value[key]
    assert value == expected


def test_reward_change_is_free_and_produces_no_warning():
    eval_cfg = build_config(PRESETS["landing-normal"])
    train_cfg = build_config({**PRESETS["landing-normal"],
                              "reward": {"success_base": 999.0}})
    ok, warnings, errors = validate_train_config(train_cfg, eval_cfg)
    assert ok
    assert errors == []
    assert warnings == []


def test_success_threshold_change_warns_but_passes():
    eval_cfg = build_config(PRESETS["landing-normal"])
    train_cfg = build_config({**PRESETS["landing-normal"],
                              "success": {"v_max": 99.0}})
    ok, warnings, errors = validate_train_config(train_cfg, eval_cfg)
    assert ok
    assert errors == []
    assert any("success.v_max" in w for w in warnings)


def test_task_mismatch_is_an_error():
    eval_cfg = build_config(PRESETS["landing-normal"])
    train_cfg = build_config(PRESETS["catch-normal"])
    ok, warnings, errors = validate_train_config(train_cfg, eval_cfg)
    assert not ok
    assert any("task" in e for e in errors)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_config.py -v
```

기대: `ModuleNotFoundError: No module named 'rocket_env.config'`

- [ ] **Step 3: `rocket_env/config.py` 구현**

```python
"""환경 설정: 기본값, 태스크 프로파일, 라운드 프리셋, 검증.

설계 원칙 — 학생은 **학습용** 설정을 자유롭게 바꿀 수 있고, 채점은 **평가용**
설정으로 이뤄진다. 이 어긋남 자체가 수업의 핵심 교보재다: 내가 최적화하는 것과
내가 평가받는 것은 다르다.

다만 물리 상수처럼 바꿔봐야 다른 문제를 푸는 셈인 키는 잠근다.
"""

import copy
from typing import Any

# 물리·관찰·행동 상수는 config로 노출하지 않는다.
LOCKED_KEYS = frozenset({
    "dt", "g", "H", "rocket_height", "observation", "action",
})

# 평가와 다르면 경고만 내는 키 경로 (학습 시 바꿔볼 가치는 있다).
WARN_PATHS = ("success", "init", "catch")

DEFAULT_CONFIG: dict[str, Any] = {
    "task": "landing",
    "max_steps": 800,
    # seed는 환경이 소비하지 않는다. 라운드 시스템이 reset(seed=...)에 쓰는
    # 호출자 측 메타데이터다. 환경이 이 값을 읽으면 학습 시 모든 에피소드가
    # 동일해지는 버그가 생긴다.
    "seed": None,

    "wind": {
        "mode": "constant",   # "none" | "constant" | "gust"
        "max_speed": 8.0,
        "ou_theta": 0.0,
        "ou_sigma": 0.0,
    },

    "fuel": {"capacity": 120.0},   # None이면 무한

    "init": {
        "y": 450.0,
        "vy_range": [-60.0, -50.0],
        "x_range": [-150.0, 150.0],
        "theta_range_deg": [-45.0, 45.0],
    },

    "success": {
        "v_max": 15.0,
        "theta_max_deg": 10.0,
        "omega_max_deg": 10.0,
        "zone_r": 50.0,
    },

    "catch": {"x_tower": 0.0, "y_arm": 80.0},

    "reward": {
        "success_base": 100.0,
        "w_speed": 40.0,
        "v_ref": 5.0,
        "w_position": 30.0,
        "w_attitude": 20.0,
        "w_fuel": 30.0,
        "w_time": 30.0,
        "failure_max": 40.0,
        # 1.0이어야 shaping 총합이 정확히 Φ(s_T) - Φ(s_0)로 접힌다.
        # γ<1이면 에피소드가 길수록 shaping 총합이 커지는 편향이 생긴다.
        "shaping_gamma": 1.0,
        "shaping_w_dist": 1.0,
        "shaping_w_attitude": 0.5,
        "fuel_penalty": 0.05,
    },
}

# task="catch"일 때 갈아끼우는 프로파일. 사용자가 명시한 값은 덮지 않는다.
CATCH_OVERRIDES: dict[str, Any] = {
    "success": {
        "v_max": 5.0,
        "theta_max_deg": 5.0,
        "omega_max_deg": 5.0,
        "zone_r": 6.0,
    },
    "reward": {
        "w_speed": 60.0,
        "v_ref": 2.0,
        "w_fuel": 20.0,
        "w_time": 20.0,
    },
}

PRESETS: dict[str, dict[str, Any]] = {
    "landing-easy": {
        "task": "landing",
        "wind": {"mode": "none", "max_speed": 0.0},
        "fuel": {"capacity": None},
        "init": {"y": 450.0, "vy_range": [-30.0, -30.0],
                 "x_range": [-100.0, 100.0], "theta_range_deg": [-15.0, 15.0]},
    },
    "landing-normal": {
        "task": "landing",
        "wind": {"mode": "constant", "max_speed": 8.0},
        "fuel": {"capacity": 120.0},
        "init": {"y": 450.0, "vy_range": [-60.0, -50.0],
                 "x_range": [-150.0, 150.0], "theta_range_deg": [-45.0, 45.0]},
    },
    "landing-hard": {
        "task": "landing",
        "wind": {"mode": "gust", "max_speed": 15.0,
                 "ou_theta": 0.15, "ou_sigma": 3.0},
        "fuel": {"capacity": 90.0},
        "init": {"y": 450.0, "vy_range": [-70.0, -60.0],
                 "x_range": [-200.0, 200.0], "theta_range_deg": [-85.0, 85.0]},
        "success": {"zone_r": 30.0},
    },
    "catch-normal": {
        "task": "catch",
        "wind": {"mode": "constant", "max_speed": 5.0},
        "fuel": {"capacity": 140.0},
        "init": {"y": 450.0, "vy_range": [-50.0, -40.0],
                 "x_range": [-100.0, 100.0], "theta_range_deg": [-30.0, 30.0]},
    },
    "catch-hard": {
        "task": "catch",
        "wind": {"mode": "gust", "max_speed": 12.0,
                 "ou_theta": 0.15, "ou_sigma": 3.0},
        "fuel": {"capacity": 110.0},
        "init": {"y": 450.0, "vy_range": [-60.0, -55.0],
                 "x_range": [-150.0, 150.0], "theta_range_deg": [-60.0, 60.0]},
    },
}


class ConfigError(ValueError):
    """설정이 잘못되었거나 잠긴 키를 건드렸을 때."""


def _deep_merge(base: dict, overlay: dict) -> dict:
    """overlay를 base 위에 재귀적으로 얹는다. base는 변경하지 않는다."""
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _reject_unknown_keys(user: dict, schema: dict, path: str = "") -> None:
    """스키마에 없는 키를 거부한다.

    오타 난 키가 조용히 무시되는 것이 가장 나쁜 실패 모드다.
    `{"fuel": {"capacty": 50.0}}` 처럼 한 글자만 틀려도 병합은 성공하고,
    의도한 설정이 전혀 적용되지 않은 채로 몇 시간짜리 학습이 끝난다.
    라운드 설정을 쓰는 조교도 학생도 이 실수는 즉시 알아야 한다.
    """
    for key, value in user.items():
        full = f"{path}{key}"
        if key not in schema:
            raise ConfigError(
                f"알 수 없는 설정 키: {full!r}. "
                f"사용 가능한 키: {sorted(schema)}"
            )
        if isinstance(value, dict) and isinstance(schema[key], dict):
            _reject_unknown_keys(value, schema[key], path=f"{full}.")


def build_config(user_config: dict | None) -> dict:
    """기본값 → 태스크 프로파일 → 사용자 설정 순으로 병합한 완전한 설정."""
    user = user_config or {}

    locked = LOCKED_KEYS & set(user)
    if locked:
        raise ConfigError(
            f"다음 키는 환경 상수라 변경할 수 없습니다: {sorted(locked)}"
        )

    _reject_unknown_keys(user, DEFAULT_CONFIG)

    task = user.get("task", DEFAULT_CONFIG["task"])
    if task not in ("landing", "catch"):
        raise ConfigError(f"task는 'landing' 또는 'catch'여야 합니다: {task!r}")

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if task == "catch":
        cfg = _deep_merge(cfg, CATCH_OVERRIDES)
    cfg = _deep_merge(cfg, user)

    _validate_ranges(cfg)
    return cfg


def _validate_ranges(cfg: dict) -> None:
    if cfg["max_steps"] <= 0:
        raise ConfigError(f"max_steps는 양수여야 합니다: {cfg['max_steps']}")

    capacity = cfg["fuel"]["capacity"]
    if capacity is not None and capacity <= 0:
        raise ConfigError(f"fuel.capacity는 양수이거나 None이어야 합니다: {capacity}")

    if cfg["wind"]["max_speed"] < 0:
        raise ConfigError(
            f"wind.max_speed는 음수일 수 없습니다: {cfg['wind']['max_speed']}"
        )
    if cfg["wind"]["mode"] not in ("none", "constant", "gust"):
        raise ConfigError(f"wind.mode가 올바르지 않습니다: {cfg['wind']['mode']!r}")

    for key, value in cfg["success"].items():
        if value <= 0:
            raise ConfigError(f"success.{key}는 양수여야 합니다: {value}")


def validate_train_config(train_cfg: dict,
                          eval_cfg: dict) -> tuple[bool, list[str], list[str]]:
    """학습 설정을 평가 설정과 대조한다.

    Returns:
        (ok, warnings, errors) — errors가 비어 있으면 ok=True.
        task 불일치만 오류다. 정책 자체가 다른 문제를 풀도록 학습되기 때문이다.
    """
    warnings: list[str] = []
    errors: list[str] = []

    if train_cfg["task"] != eval_cfg["task"]:
        errors.append(
            f"task 불일치: 학습={train_cfg['task']!r} 평가={eval_cfg['task']!r}"
        )

    for section in WARN_PATHS:
        for key in eval_cfg.get(section, {}):
            train_value = train_cfg.get(section, {}).get(key)
            eval_value = eval_cfg[section][key]
            if train_value != eval_value:
                warnings.append(
                    f"{section}.{key}가 평가와 다릅니다: "
                    f"학습={train_value} 평가={eval_value}"
                )

    return (not errors), warnings, errors
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/test_config.py -v
```

기대: 32 passed (parametrize 5 + 12건 포함).

- [ ] **Step 5: 커밋**

```bash
git add rocket_env/config.py tests/test_config.py
git commit -m "$(cat <<'EOF'
feat: config 병합·검증·라운드 프리셋

잠금/경고/자유 3단계 정책을 구현. 학생은 reward/wind/fuel을 자유롭게
바꿀 수 있고, 물리 상수는 ConfigError로 막힌다. task 불일치만 오류로
처리하고 나머지 차이는 경고로 남긴다.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Landing 태스크

**Files:**
- Create: `rocket_env/tasks/__init__.py`, `rocket_env/tasks/base.py`, `rocket_env/tasks/landing.py`
- Test: `tests/test_task_landing.py`

**Interfaces:**
- Consumes: `rocket_env.physics` 상수, `rocket_env.types.State`/`Outcome`
- Produces:
  - `rocket_env.tasks.base.Task` — 프로토콜: `name: str`, `initial_state(rng, cfg) -> State`, `target(cfg) -> tuple[float, float]`, `evaluate(prev: State, cur: State, cfg) -> str | None`
  - `rocket_env.tasks.base.sample_initial_state(rng, cfg) -> State` — 두 태스크 공용
  - `rocket_env.tasks.base.out_of_bounds(state: State) -> bool`
  - `rocket_env.tasks.landing.LandingTask`
  - `rocket_env.tasks.make_task(name: str) -> Task`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_task_landing.py`:

```python
"""지면 착륙 태스크의 성공/실패 판정."""

import math

import numpy as np
import pytest

from rocket_env.config import PRESETS, build_config
from rocket_env.physics import ROCKET_HEIGHT, WORLD_Y_MAX
from rocket_env.tasks import make_task
from rocket_env.types import Outcome, State

CFG = build_config(PRESETS["landing-normal"])
TASK = make_task("landing")
GROUND = ROCKET_HEIGHT / 2.0


def at(**kw) -> State:
    base = dict(x=0.0, y=200.0, vx=0.0, vy=0.0, theta=0.0, omega=0.0,
                phi=0.0, thrust=0.0, fuel=100.0, wind_x=0.0, step=10)
    base.update(kw)
    return State(**base)


def test_target_is_pad_centre_at_half_rocket_height():
    assert TASK.target(CFG) == (0.0, GROUND)


def test_airborne_state_is_in_progress():
    assert TASK.evaluate(at(y=201.0), at(y=200.0), CFG) is None


def test_perfect_touchdown_succeeds():
    cur = at(y=GROUND - 0.1, vy=-1.0)
    assert TASK.evaluate(at(y=GROUND + 5.0), cur, CFG) == Outcome.SUCCESS


def test_touchdown_too_fast_crashes():
    cur = at(y=GROUND - 0.1, vy=-CFG["success"]["v_max"] - 0.1)
    assert TASK.evaluate(at(y=GROUND + 5.0), cur, CFG) == Outcome.CRASH


def test_touchdown_outside_pad_crashes():
    cur = at(y=GROUND - 0.1, x=CFG["success"]["zone_r"] + 0.1, vy=-1.0)
    assert TASK.evaluate(at(y=GROUND + 5.0), cur, CFG) == Outcome.CRASH


def test_touchdown_tilted_crashes():
    cur = at(y=GROUND - 0.1, vy=-1.0,
             theta=math.radians(CFG["success"]["theta_max_deg"] + 0.1))
    assert TASK.evaluate(at(y=GROUND + 5.0), cur, CFG) == Outcome.CRASH


def test_touchdown_spinning_crashes():
    cur = at(y=GROUND - 0.1, vy=-1.0,
             omega=math.radians(CFG["success"]["omega_max_deg"] + 0.1))
    assert TASK.evaluate(at(y=GROUND + 5.0), cur, CFG) == Outcome.CRASH


def test_flying_off_the_top_crashes():
    cur = at(y=WORLD_Y_MAX - GROUND + 1.0)
    assert TASK.evaluate(at(y=500.0), cur, CFG) == Outcome.CRASH


def test_flying_off_the_side_crashes():
    cur = at(x=301.0)
    assert TASK.evaluate(at(x=299.0), cur, CFG) == Outcome.CRASH


def test_touchdown_exactly_at_speed_threshold_crashes():
    """임계값 비교는 strict `<` 다 — 정확히 임계값이면 실패한다.

    ±0.1로만 찔러보는 테스트는 `<` 와 `<=` 를 구별하지 못한다. 성적을
    만드는 코드에서 이 한 칸이 '겨우 통과'와 '겨우 실패'를 가른다.
    """
    cur = at(y=GROUND - 0.1, vy=-CFG["success"]["v_max"])
    assert TASK.evaluate(at(y=GROUND + 5.0), cur, CFG) == Outcome.CRASH


def test_touchdown_exactly_at_pad_edge_crashes():
    cur = at(y=GROUND - 0.1, x=CFG["success"]["zone_r"], vy=-1.0)
    assert TASK.evaluate(at(y=GROUND + 5.0), cur, CFG) == Outcome.CRASH


def test_touchdown_exactly_at_tilt_threshold_crashes():
    cur = at(y=GROUND - 0.1, vy=-1.0,
             theta=math.radians(CFG["success"]["theta_max_deg"]))
    assert TASK.evaluate(at(y=GROUND + 5.0), cur, CFG) == Outcome.CRASH


def test_touchdown_exactly_at_spin_threshold_crashes():
    cur = at(y=GROUND - 0.1, vy=-1.0,
             omega=math.radians(CFG["success"]["omega_max_deg"]))
    assert TASK.evaluate(at(y=GROUND + 5.0), cur, CFG) == Outcome.CRASH


def test_ground_contact_triggers_exactly_at_ground_level():
    """접지 판정만 `<=` 다 — 정확히 지면 높이에 닿으면 접지로 본다."""
    cur = at(y=GROUND, vy=-1.0)
    assert TASK.evaluate(at(y=GROUND + 5.0), cur, CFG) == Outcome.SUCCESS


def test_initial_state_respects_config_ranges():
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = TASK.initial_state(rng, CFG)
        assert CFG["init"]["x_range"][0] <= s.x <= CFG["init"]["x_range"][1]
        assert CFG["init"]["vy_range"][0] <= s.vy <= CFG["init"]["vy_range"][1]
        assert s.y == CFG["init"]["y"]
        assert abs(s.theta) <= math.radians(CFG["init"]["theta_range_deg"][1])
        assert s.fuel == CFG["fuel"]["capacity"]
        assert s.step == 0


def test_unlimited_fuel_config_yields_infinite_fuel():
    cfg = build_config(PRESETS["landing-easy"])
    s = TASK.initial_state(np.random.default_rng(0), cfg)
    assert math.isinf(s.fuel)


def test_make_task_rejects_unknown_name():
    with pytest.raises(ValueError, match="hover"):
        make_task("hover")
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_task_landing.py -v
```

기대: `ModuleNotFoundError: No module named 'rocket_env.tasks'`

- [ ] **Step 3: `rocket_env/tasks/base.py` 구현**

```python
"""태스크 인터페이스와 두 태스크가 공유하는 로직.

태스크는 '언제 끝나고 무엇이 성공인가'만 정의한다. 물리도 보상도 모른다.
"""

import math
from typing import Protocol

import numpy as np

from rocket_env.physics import (
    ROCKET_HEIGHT,
    WORLD_X_MAX,
    WORLD_X_MIN,
    WORLD_Y_MAX,
)
from rocket_env.types import State

GROUND_Y = ROCKET_HEIGHT / 2.0     # 로켓 중심이 지면에 닿는 높이
CEILING_Y = WORLD_Y_MAX - ROCKET_HEIGHT / 2.0


class Task(Protocol):
    name: str

    def initial_state(self, rng: np.random.Generator, cfg: dict) -> State:
        """에피소드 시작 상태를 샘플링한다."""
        ...

    def target(self, cfg: dict) -> tuple[float, float]:
        """관찰의 기준이 되는 목표점 (x, y)."""
        ...

    def evaluate(self, prev: State, cur: State, cfg: dict) -> str | None:
        """종료 사유를 반환한다. 아직 진행 중이면 None."""
        ...


def sample_initial_state(rng: np.random.Generator, cfg: dict) -> State:
    """두 태스크가 동일한 초기 조건 분포를 쓴다."""
    init = cfg["init"]
    capacity = cfg["fuel"]["capacity"]
    return State(
        x=float(rng.uniform(*init["x_range"])),
        y=float(init["y"]),
        vx=0.0,
        vy=float(rng.uniform(*init["vy_range"])),
        theta=math.radians(float(rng.uniform(*init["theta_range_deg"]))),
        omega=0.0,
        phi=0.0,
        thrust=0.0,
        fuel=math.inf if capacity is None else float(capacity),
        wind_x=0.0,
        step=0,
    )


def out_of_bounds(state: State) -> bool:
    """세계 밖으로 벗어났는가."""
    return (state.y >= CEILING_Y
            or state.x <= WORLD_X_MIN
            or state.x >= WORLD_X_MAX)


def within_thresholds(state: State, cfg: dict, dx: float) -> bool:
    """속도·자세·각속도·수평 오차가 모두 성공 임계 안인가."""
    s = cfg["success"]
    speed = math.hypot(state.vx, state.vy)
    return (abs(dx) < s["zone_r"]
            and speed < s["v_max"]
            and abs(state.theta) < math.radians(s["theta_max_deg"])
            and abs(state.omega) < math.radians(s["omega_max_deg"]))
```

- [ ] **Step 4: `rocket_env/tasks/landing.py` 구현**

```python
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
```

- [ ] **Step 5: `rocket_env/tasks/__init__.py` 구현**

```python
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
```

- [ ] **Step 6: 테스트 통과 확인**

```bash
uv run pytest tests/test_task_landing.py -v
```

기대: 17 passed.

- [ ] **Step 7: 커밋**

```bash
git add rocket_env/tasks/ tests/test_task_landing.py
git commit -m "$(cat <<'EOF'
feat: Task 인터페이스와 지면 착륙 태스크

태스크는 종료 판정만 담당하고 물리·보상은 모른다. 초기 상태 샘플링과
경계 검사는 base에 두어 캐치 태스크와 공유한다.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Catch 태스크

**Files:**
- Create: `rocket_env/tasks/catch.py`
- Modify: `rocket_env/tasks/__init__.py` (레지스트리에 `catch` 추가)
- Test: `tests/test_task_catch.py`

**Interfaces:**
- Consumes: Task 4의 `sample_initial_state`, `out_of_bounds`, `within_thresholds`, `GROUND_Y`
- Produces: `rocket_env.tasks.catch.CatchTask` — `make_task("catch")`로 얻는다

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_task_catch.py`:

```python
"""젓가락 포획 태스크의 통과 판정.

핵심은 '하강 중 팔 높이를 가로지르는 단 한 스텝'에만 판정이 일어난다는 것이다.
"""

import math

import pytest

from rocket_env.config import PRESETS, build_config
from rocket_env.tasks import make_task
from rocket_env.types import Outcome, State

CFG = build_config(PRESETS["catch-normal"])
TASK = make_task("catch")
Y_ARM = CFG["catch"]["y_arm"]


def at(**kw) -> State:
    base = dict(x=0.0, y=200.0, vx=0.0, vy=-1.0, theta=0.0, omega=0.0,
                phi=0.0, thrust=0.0, fuel=100.0, wind_x=0.0, step=10)
    base.update(kw)
    return State(**base)


def crossing(**kw) -> tuple[State, State]:
    """팔 높이를 아래로 가로지르는 (prev, cur) 쌍."""
    return at(y=Y_ARM + 1.0), at(y=Y_ARM - 0.1, **kw)


def test_target_is_the_tower_arm():
    assert TASK.target(CFG) == (CFG["catch"]["x_tower"], Y_ARM)


def test_far_above_the_arm_is_in_progress():
    assert TASK.evaluate(at(y=300.0), at(y=299.0), CFG) is None


def test_slow_centred_crossing_is_caught():
    prev, cur = crossing(vx=0.0, vy=-1.0)
    assert TASK.evaluate(prev, cur, CFG) == Outcome.SUCCESS


def test_fast_crossing_is_missed():
    prev, cur = crossing(vy=-CFG["success"]["v_max"] - 0.1)
    assert TASK.evaluate(prev, cur, CFG) == Outcome.MISSED


def test_offset_crossing_is_missed():
    prev, cur = crossing(x=CFG["success"]["zone_r"] + 0.1, vy=-1.0)
    assert TASK.evaluate(prev, cur, CFG) == Outcome.MISSED


def test_tilted_crossing_is_missed():
    prev, cur = crossing(vy=-1.0,
                         theta=math.radians(CFG["success"]["theta_max_deg"] + 0.1))
    assert TASK.evaluate(prev, cur, CFG) == Outcome.MISSED


def test_spinning_crossing_is_missed():
    prev, cur = crossing(vy=-1.0,
                         omega=math.radians(CFG["success"]["omega_max_deg"] + 0.1))
    assert TASK.evaluate(prev, cur, CFG) == Outcome.MISSED


def test_crossing_upward_is_not_judged():
    """아래에서 위로 지나가는 것은 포획 시도가 아니다."""
    prev, cur = at(y=Y_ARM - 1.0, vy=+5.0), at(y=Y_ARM + 0.1, vy=+5.0)
    assert TASK.evaluate(prev, cur, CFG) is None


def test_hovering_below_the_arm_does_not_retrigger():
    """한 번 지나간 뒤 팔 아래에서 맴돌아도 다시 판정되지 않는다."""
    assert TASK.evaluate(at(y=Y_ARM - 5.0), at(y=Y_ARM - 6.0), CFG) is None


def test_crossing_exactly_at_arm_height_is_judged():
    """`y_arm >= cur.y` 는 등호를 포함한다 — 정확히 팔 높이에 닿아도 판정한다.

    다른 테스트는 전부 Y_ARM - 0.1 을 쓰므로 등호를 빼도(`>` 로 바꿔도)
    모두 통과한다. 이 테스트가 그 한 칸을 고정한다.
    """
    prev, cur = at(y=Y_ARM + 1.0), at(y=Y_ARM, vy=-1.0)
    assert TASK.evaluate(prev, cur, CFG) == Outcome.SUCCESS


def test_starting_exactly_at_arm_height_is_not_a_crossing():
    """`prev.y > y_arm` 은 strict 다 — 팔 높이에서 출발하면 통과가 아니다.

    등호를 허용하면 팔 높이 부근에서 맴도는 로켓이 매 스텝 재판정된다.
    """
    prev, cur = at(y=Y_ARM, vy=-1.0), at(y=Y_ARM - 0.1, vy=-1.0)
    assert TASK.evaluate(prev, cur, CFG) is None


def test_step_that_reverses_to_upward_is_not_judged():
    """`cur.vy < 0` 가드가 실제로 지키는 유일한 경우.

    한 스텝의 순 변위는 아래쪽인데 끝 속도가 위로 뒤집힌 상태 — 팔 높이
    부근에서 거의 멈췄다가 반등하는 순간이다. test_crossing_upward_is_not_judged
    는 y 순서 조건만으로 이미 걸러져서 이 가드를 전혀 시험하지 못한다.
    """
    prev, cur = at(y=Y_ARM + 0.05, vy=-0.3), at(y=Y_ARM - 0.01, vy=+0.1)
    assert TASK.evaluate(prev, cur, CFG) is None


def test_reaching_the_ground_without_crossing_crashes():
    from rocket_env.tasks.base import GROUND_Y
    assert TASK.evaluate(at(y=GROUND_Y + 1.0),
                         at(y=GROUND_Y - 0.1), CFG) == Outcome.CRASH


def test_offset_tower_shifts_the_capture_zone():
    cfg = build_config({**PRESETS["catch-normal"],
                        "catch": {"x_tower": 100.0, "y_arm": Y_ARM}})
    prev, cur = crossing(x=100.0, vy=-1.0)
    assert TASK.evaluate(prev, cur, cfg) == Outcome.SUCCESS
    prev, cur = crossing(x=0.0, vy=-1.0)
    assert TASK.evaluate(prev, cur, cfg) == Outcome.MISSED
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_task_catch.py -v
```

기대: `ValueError: 알 수 없는 task: 'catch'`

- [ ] **Step 3: `rocket_env/tasks/catch.py` 구현**

```python
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
```

- [ ] **Step 4: `rocket_env/tasks/__init__.py` 수정**

```python
"""태스크 팩토리."""

from rocket_env.tasks.base import Task
from rocket_env.tasks.catch import CatchTask
from rocket_env.tasks.landing import LandingTask

__all__ = ["Task", "LandingTask", "CatchTask", "make_task"]

_REGISTRY = {"landing": LandingTask, "catch": CatchTask}


def make_task(name: str) -> Task:
    if name not in _REGISTRY:
        raise ValueError(
            f"알 수 없는 task: {name!r} (가능한 값: {sorted(_REGISTRY)})"
        )
    return _REGISTRY[name]()
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
uv run pytest tests/test_task_catch.py tests/test_task_landing.py -v
```

기대: 31 passed.

- [ ] **Step 6: 커밋**

```bash
git add rocket_env/tasks/ tests/test_task_catch.py
git commit -m "$(cat <<'EOF'
feat: 젓가락 포획 태스크

하강 중 팔 높이를 가로지르는 단일 스텝에만 판정한다. 상승 통과와
팔 아래 체공은 판정하지 않으며, 놓치면 즉시 종료된다.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: 보상 함수

**Files:**
- Create: `rocket_env/reward.py`
- Test: `tests/test_reward.py`

**Interfaces:**
- Consumes: `rocket_env.types.State`/`Outcome`, cfg 딕셔너리
- Produces:
  - `rocket_env.reward.potential(state: State, target: tuple[float, float], cfg: dict) -> float`
  - `rocket_env.reward.shaping(prev_potential: float, cur_potential: float, cfg: dict) -> float`
  - `rocket_env.reward.terminal_reward(outcome: str, state: State, target: tuple[float, float], cfg: dict, d_initial: float, fuel_frac: float) -> float`
  - `rocket_env.reward.distance_to_target(state: State, target: tuple[float, float]) -> float`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_reward.py`:

```python
"""보상 설계 검증.

가장 중요한 것은 shaping 총합이 에피소드 길이와 무관하다는 점과,
실패 보상에 시간 항이 전혀 없다는 점이다 (원본 환경의 '빨리 자폭' 버그 방지).
"""

import math

import pytest

from rocket_env.config import PRESETS, build_config
from rocket_env.reward import (
    distance_to_target,
    potential,
    shaping,
    terminal_reward,
)
from rocket_env.types import Outcome, State

CFG = build_config(PRESETS["landing-normal"])
TARGET = (0.0, 25.0)


def at(**kw) -> State:
    base = dict(x=0.0, y=200.0, vx=0.0, vy=0.0, theta=0.0, omega=0.0,
                phi=0.0, thrust=0.0, fuel=100.0, wind_x=0.0, step=100)
    base.update(kw)
    return State(**base)


def test_potential_is_zero_at_target_and_upright():
    assert potential(at(x=0.0, y=25.0), TARGET, CFG) == pytest.approx(0.0)


def test_potential_is_negative_away_from_target():
    assert potential(at(x=100.0), TARGET, CFG) < 0.0


def test_potential_penalises_tilt():
    upright = potential(at(), TARGET, CFG)
    tilted = potential(at(theta=math.radians(45.0)), TARGET, CFG)
    assert tilted < upright


def test_shaping_sum_telescopes_exactly():
    """shaping_gamma=1.0이면 총합이 정확히 Φ(s_T) - Φ(s_0)다.

    이것이 '목표 근처에서 오래 버티며 점수 쌓기'를 구조적으로 막는다.
    """
    trajectory = [at(x=float(i), y=200.0 - i) for i in range(0, 300)]
    potentials = [potential(s, TARGET, CFG) for s in trajectory]
    total = sum(shaping(potentials[i], potentials[i + 1], CFG)
                for i in range(len(potentials) - 1))
    assert total == pytest.approx(potentials[-1] - potentials[0], abs=1e-9)


def test_shaping_total_is_independent_of_episode_length():
    short = [at(x=0.0, y=200.0), at(x=0.0, y=100.0)]
    long = [at(x=0.0, y=200.0)] + [at(x=0.0, y=150.0)] * 500 + [at(x=0.0, y=100.0)]

    def total(states):
        p = [potential(s, TARGET, CFG) for s in states]
        return sum(shaping(p[i], p[i + 1], CFG) for i in range(len(p) - 1))

    assert total(short) == pytest.approx(total(long), abs=1e-9)


def test_perfect_landing_scores_near_the_maximum():
    r = terminal_reward(Outcome.SUCCESS, at(x=0.0, y=25.0, step=0),
                        TARGET, CFG, d_initial=425.0, fuel_frac=1.0)
    assert r == pytest.approx(250.0, abs=0.5)


def test_marginal_success_still_beats_every_failure():
    """최악의 성공도 최고의 실패보다 커야 한다."""
    s = CFG["success"]
    marginal = terminal_reward(
        Outcome.SUCCESS,
        at(x=s["zone_r"] - 0.01, y=25.0,
           vy=-(s["v_max"] - 0.01),
           theta=math.radians(s["theta_max_deg"] - 0.01),
           step=CFG["max_steps"]),
        TARGET, CFG, d_initial=425.0, fuel_frac=0.0)
    best_failure = CFG["reward"]["failure_max"]
    assert marginal > best_failure


def test_crash_at_start_position_scores_zero():
    """진행이 없으면 부분 점수도 없다 — '빨리 자폭' 전략의 봉쇄."""
    start = at(x=0.0, y=450.0)
    d0 = distance_to_target(start, TARGET)
    r = terminal_reward(Outcome.CRASH, start, TARGET, CFG,
                        d_initial=d0, fuel_frac=0.5)
    assert r == pytest.approx(0.0, abs=1e-9)


def test_crash_at_target_scores_the_failure_maximum():
    r = terminal_reward(Outcome.CRASH, at(x=0.0, y=25.0), TARGET, CFG,
                        d_initial=425.0, fuel_frac=0.0)
    assert r == pytest.approx(CFG["reward"]["failure_max"])


def test_failure_reward_has_no_time_term():
    """같은 상태라면 언제 끝났든 실패 점수는 동일하다.

    원본 환경은 실패 보상에 (max_steps - step)을 곱해서 조기 자폭이
    고득점이 되었다. 이 테스트가 그 회귀를 막는다.
    """
    early = terminal_reward(Outcome.CRASH, at(x=10.0, y=100.0, step=5),
                            TARGET, CFG, d_initial=425.0, fuel_frac=0.9)
    late = terminal_reward(Outcome.CRASH, at(x=10.0, y=100.0, step=790),
                           TARGET, CFG, d_initial=425.0, fuel_frac=0.1)
    assert early == pytest.approx(late)


@pytest.mark.parametrize("outcome", [
    Outcome.CRASH, Outcome.MISSED, Outcome.TIMEOUT, Outcome.OUT_OF_FUEL,
])
def test_all_failure_outcomes_share_the_same_formula(outcome):
    r = terminal_reward(outcome, at(x=10.0, y=100.0), TARGET, CFG,
                        d_initial=425.0, fuel_frac=0.5)
    assert 0.0 <= r <= CFG["reward"]["failure_max"]


def test_catch_profile_rewards_slow_contact_much_more_steeply():
    catch_cfg = build_config(PRESETS["catch-normal"])
    target = (0.0, catch_cfg["catch"]["y_arm"])

    def score(speed):
        return terminal_reward(
            Outcome.SUCCESS,
            at(x=0.0, y=target[1], vy=-speed, step=0),
            target, catch_cfg, d_initial=400.0, fuel_frac=1.0)

    assert score(0.0) - score(1.0) > score(3.0) - score(4.0)


def test_zero_initial_distance_does_not_divide_by_zero():
    r = terminal_reward(Outcome.CRASH, at(x=0.0, y=25.0), TARGET, CFG,
                        d_initial=0.0, fuel_frac=0.0)
    assert math.isfinite(r)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_reward.py -v
```

기대: `ModuleNotFoundError: No module named 'rocket_env.reward'`

- [ ] **Step 3: `rocket_env/reward.py` 구현**

```python
"""보상 설계.

두 부분으로 나뉜다.

1. 스텝 보상 — 잠재함수 기반 shaping(PBRS) + 연료 패널티.
   shaping_gamma=1.0이면 shaping 총합이 정확히 Φ(s_T) - Φ(s_0)로 접히므로,
   에피소드가 길다고 점수가 쌓이지 않는다.

2. 종료 보상 — 성공은 기본점 + 품질 보너스, 실패는 목표를 향한 '진행도'.
   실패 보상에 시간 항이 전혀 없다는 점이 중요하다. 원본 환경은 실패에도
   남은 스텝 수를 곱해서 '빨리 자폭하기'가 고득점 전략이 되었다.
"""

import math

from rocket_env.types import Outcome, State

# 잠재함수의 거리 정규화 상수. config에서 파생하지 않는 환경 상수다.
POTENTIAL_DIST_SCALE = 300.0

_FAILURE_OUTCOMES = frozenset({
    Outcome.CRASH, Outcome.MISSED, Outcome.TIMEOUT, Outcome.OUT_OF_FUEL,
})


def distance_to_target(state: State, target: tuple[float, float]) -> float:
    return math.hypot(state.x - target[0], state.y - target[1])


def potential(state: State, target: tuple[float, float], cfg: dict) -> float:
    """Φ(s). 목표에 가깝고 수직일수록 0에 가깝고, 항상 0 이하다."""
    r = cfg["reward"]
    dx = abs(state.x - target[0]) / POTENTIAL_DIST_SCALE
    dy = abs(state.y - target[1]) / POTENTIAL_DIST_SCALE
    tilt = abs(state.theta) / (math.pi / 2.0)
    return -(r["shaping_w_dist"] * (dx + dy) + r["shaping_w_attitude"] * tilt)


def shaping(prev_potential: float, cur_potential: float, cfg: dict) -> float:
    """F = γ·Φ(s') - Φ(s)."""
    return cfg["reward"]["shaping_gamma"] * cur_potential - prev_potential


def terminal_reward(outcome: str, state: State, target: tuple[float, float],
                    cfg: dict, d_initial: float, fuel_frac: float) -> float:
    """종료 시 한 번 지급되는 보상."""
    r = cfg["reward"]

    if outcome == Outcome.SUCCESS:
        s = cfg["success"]
        speed = math.hypot(state.vx, state.vy)
        dx = abs(state.x - target[0])
        return (
            r["success_base"]
            + r["w_speed"] * math.exp(-speed / r["v_ref"])
            + r["w_position"] * max(0.0, 1.0 - dx / s["zone_r"])
            + r["w_attitude"] * max(
                0.0, 1.0 - abs(state.theta) / math.radians(s["theta_max_deg"]))
            + r["w_fuel"] * fuel_frac
            + r["w_time"] * (1.0 - state.step / cfg["max_steps"])
        )

    if outcome in _FAILURE_OUTCOMES:
        d_final = distance_to_target(state, target)
        progress = 1.0 - d_final / max(d_initial, 1.0)
        return r["failure_max"] * min(max(progress, 0.0), 1.0)

    raise ValueError(f"종료 보상을 계산할 수 없는 outcome: {outcome!r}")
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/test_reward.py -v
```

기대: 16 passed (parametrize 4건 포함).

- [ ] **Step 5: 커밋**

```bash
git add rocket_env/reward.py tests/test_reward.py
git commit -m "$(cat <<'EOF'
feat: PBRS 기반 보상과 종료 보상

shaping_gamma=1.0으로 shaping 총합이 정확히 텔레스코핑되어 에피소드
길이 편향이 0이다. 실패 보상은 시간 항 없이 목표 진행도에만 비례하므로
'빨리 자폭' 전략이 0점을 받는다.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: RocketEnv (Gymnasium 파사드)

**Files:**
- Create: `rocket_env/env.py`
- Modify: `rocket_env/__init__.py` (환경 등록)
- Test: `tests/test_env.py`

**Interfaces:**
- Consumes: Task 1–6의 모든 공개 인터페이스
- Produces:
  - `rocket_env.env.RocketEnv(config: dict | None = None, render_mode: str | None = None)`
  - `rocket_env.env.OBS_DIM = 11`, `WIND_OBS_SCALE = 20.0`, `POS_OBS_SCALE = 300.0`, `VEL_OBS_SCALE = 50.0`
  - 등록된 환경 id: `rocket-v0`, `rocket-landing-v0`, `rocket-catch-v0`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_env.py`:

```python
"""Gymnasium 파사드 검증: API 준수, 관찰 규격, info 계약, 재현성."""

import math

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import rocket_env  # noqa: F401  — 환경 등록 트리거
from rocket_env.config import PRESETS
from rocket_env.env import OBS_DIM, RocketEnv
from rocket_env.types import Outcome

NOOP = 1        # 추력 0, 노즐 정지
FULL_UP = 10    # 추력 2g, 노즐 정지


def rollout(env, action, seed):
    env.reset(seed=seed)
    total = 0.0
    while True:
        _, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            return total, info


def test_registered_ids_are_makeable():
    for env_id in ("rocket-v0", "rocket-landing-v0", "rocket-catch-v0"):
        env = gym.make(env_id)
        env.reset(seed=0)
        env.close()


def test_alias_ids_select_the_right_task():
    assert gym.make("rocket-catch-v0").unwrapped.cfg["task"] == "catch"
    assert gym.make("rocket-landing-v0").unwrapped.cfg["task"] == "landing"


def test_alias_id_still_accepts_extra_config():
    env = gym.make("rocket-catch-v0", config={"max_steps": 123})
    assert env.unwrapped.cfg["task"] == "catch"
    assert env.unwrapped.cfg["max_steps"] == 123


def test_passes_gymnasium_env_checker():
    check_env(RocketEnv(), skip_render_check=True)


def test_spaces_match_the_contract():
    env = RocketEnv()
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.n == 12


def test_observation_is_finite_and_correctly_typed():
    env = RocketEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))


def test_unlimited_fuel_shows_full_fuel_fraction():
    env = RocketEnv(config=PRESETS["landing-easy"])
    obs, _ = env.reset(seed=0)
    assert obs[8] == pytest.approx(1.0)


def test_info_contains_every_contract_key():
    env = RocketEnv()
    env.reset(seed=0)
    _, _, _, _, info = env.step(NOOP)
    for key in ("is_success", "outcome", "fuel_left", "fuel_frac",
                "impact_speed", "wind_x", "step"):
        assert key in info


def test_same_seed_reproduces_identical_trajectories():
    env = RocketEnv(config=PRESETS["landing-hard"])
    a, _ = rollout(env, NOOP, seed=123)
    b, _ = rollout(env, NOOP, seed=123)
    c, _ = rollout(env, NOOP, seed=124)
    assert a == b
    assert a != c


def test_config_seed_is_not_consumed_by_the_env():
    """cfg['seed']는 호출자 메타데이터다. 환경이 읽으면 학습 시
    모든 에피소드가 동일해지는 버그가 생긴다."""
    env = RocketEnv(config={**PRESETS["landing-normal"], "seed": 7})
    env.reset()
    first = env.unwrapped.state.x
    env.reset()
    assert env.unwrapped.state.x != first


def test_zero_thrust_from_altitude_ends_in_crash():
    env = RocketEnv(config=PRESETS["landing-normal"])
    _, info = rollout(env, NOOP, seed=0)
    assert info["outcome"] == Outcome.CRASH
    assert info["is_success"] is False
    assert info["impact_speed"] is not None


def test_running_out_of_fuel_is_reported_distinctly():
    env = RocketEnv(config={**PRESETS["landing-normal"],
                            "fuel": {"capacity": 1.0}})
    _, info = rollout(env, FULL_UP, seed=0)
    assert info["outcome"] == Outcome.OUT_OF_FUEL


def test_timeout_truncates_rather_than_terminates():
    """추력 1g 부근으로 떠 있으면 max_steps에 걸린다."""
    env = RocketEnv(config={**PRESETS["landing-easy"], "max_steps": 30})
    env.reset(seed=0)
    for _ in range(30):
        obs, reward, terminated, truncated, info = env.step(7)  # 1.0g, 노즐 정지
    assert truncated
    assert not terminated
    assert info["outcome"] == Outcome.TIMEOUT


def test_fuel_never_goes_negative():
    env = RocketEnv(config={**PRESETS["landing-normal"],
                            "fuel": {"capacity": 2.0}})
    env.reset(seed=0)
    for _ in range(200):
        _, _, terminated, truncated, info = env.step(FULL_UP)
        assert info["fuel_left"] >= 0.0
        if terminated or truncated:
            break


def test_wind_disabled_config_keeps_wind_at_zero():
    env = RocketEnv(config=PRESETS["landing-easy"])
    env.reset(seed=0)
    for _ in range(50):
        _, _, terminated, truncated, info = env.step(NOOP)
        assert info["wind_x"] == 0.0
        if terminated or truncated:
            break


def test_catch_task_can_be_selected_by_config():
    env = RocketEnv(config=PRESETS["catch-normal"])
    _, info = rollout(env, NOOP, seed=0)
    assert info["outcome"] in (Outcome.MISSED, Outcome.CRASH)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_env.py -v
```

기대: `ModuleNotFoundError: No module named 'rocket_env.env'`

- [ ] **Step 3: `rocket_env/env.py` 구현**

```python
"""Gymnasium 파사드.

물리·바람·태스크·보상 계층을 조립해 표준 Env 인터페이스로 노출한다.
서버 평가 워커는 이 파일의 계약만 알면 된다.
"""

import math
from dataclasses import replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rocket_env.config import build_config
from rocket_env.physics import ACTION_TABLE, DT, PHI_MAX, fuel_cost, integrate
from rocket_env.reward import (
    distance_to_target,
    potential,
    shaping,
    terminal_reward,
)
from rocket_env.tasks import make_task
from rocket_env.types import Outcome
from rocket_env.wind import WindProcess

OBS_DIM = 11

# 관찰 정규화 상수. 전부 환경 고정값이며 config에서 파생하지 않는다.
# config에서 파생하면 같은 관찰값이 라운드마다 다른 물리량을 뜻하게 되어
# 정책이 라운드 간에 전이되지 않는다.
POS_OBS_SCALE = 300.0
VEL_OBS_SCALE = 50.0
WIND_OBS_SCALE = 20.0


class RocketEnv(gym.Env):
    """로켓 착륙 / 젓가락 포획 환경.

    Args:
        config: 부분 설정 딕셔너리. `rocket_env.config.build_config`로 병합된다.
        render_mode: "human" | "rgb_array" | None
    """

    metadata = {"render_modes": ["human", "rgb_array"],
                "render_fps": int(round(1.0 / DT))}

    def __init__(self, config: dict | None = None,
                 render_mode: str | None = None):
        super().__init__()
        self.cfg = build_config(config)
        self.task = make_task(self.cfg["task"])
        self.wind = WindProcess(**self.cfg["wind"])

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTION_TABLE))

        self.render_mode = render_mode
        self._renderer = None

        self.state = None
        self._target = self.task.target(self.cfg)
        self._d_initial = 1.0
        self._potential = 0.0
        self._outcome = Outcome.IN_PROGRESS

    # --- Gymnasium API ---

    def reset(self, *, seed=None, options=None):
        # cfg["seed"]는 의도적으로 읽지 않는다. 라운드 시스템이 에피소드마다
        # reset(seed=base_seed + i)로 넘겨주는 호출자 측 메타데이터다.
        super().reset(seed=seed)

        state = self.task.initial_state(self.np_random, self.cfg)
        wind_x = self.wind.reset(self.np_random)
        self.state = replace(state, wind_x=wind_x)

        self._target = self.task.target(self.cfg)
        self._d_initial = distance_to_target(self.state, self._target)
        self._potential = potential(self.state, self._target, self.cfg)
        self._outcome = Outcome.IN_PROGRESS

        if self._renderer is not None:
            self._renderer.reset()

        return self._observation(), self._info(impact_speed=None)

    def step(self, action):
        thrust, nozzle_rate = ACTION_TABLE[int(action)]

        # 연료가 바닥나면 엔진이 꺼진다.
        if self.state.fuel <= 0.0:
            thrust = 0.0
        used = fuel_cost(thrust)
        fuel_left = self.state.fuel - used
        if math.isfinite(fuel_left):
            fuel_left = max(fuel_left, 0.0)

        wind_x = self.wind.step(self.state.wind_x, self.np_random)

        prev = self.state
        cur = integrate(prev, thrust, nozzle_rate, wind_x)
        cur = replace(cur, fuel=fuel_left, wind_x=wind_x)
        self.state = cur

        outcome = self.task.evaluate(prev, cur, self.cfg)
        truncated = False
        if outcome is None and cur.step >= self.cfg["max_steps"]:
            outcome = Outcome.TIMEOUT
            truncated = True
        # 연료 소진이 다른 실패 사유보다 우선한다 — 디버깅에 가장 유용하다.
        if outcome == Outcome.CRASH and self._fuel_frac() <= 0.0:
            outcome = Outcome.OUT_OF_FUEL

        terminated = outcome is not None and not truncated

        cur_potential = potential(cur, self._target, self.cfg)
        reward = (shaping(self._potential, cur_potential, self.cfg)
                  - self.cfg["reward"]["fuel_penalty"] * used)
        self._potential = cur_potential

        impact_speed = None
        if outcome is not None:
            self._outcome = outcome
            reward += terminal_reward(outcome, cur, self._target, self.cfg,
                                      self._d_initial, self._fuel_frac())
            if outcome != Outcome.TIMEOUT:
                impact_speed = math.hypot(cur.vx, cur.vy)

        return (self._observation(), float(reward), terminated, truncated,
                self._info(impact_speed=impact_speed))

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from rocket_env.render import Renderer
            self._renderer = Renderer(self.cfg, self.render_mode)
        return self._renderer.draw(self.state, self._target, self._outcome)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # --- 내부 헬퍼 ---

    def _fuel_frac(self) -> float:
        capacity = self.cfg["fuel"]["capacity"]
        if capacity is None:
            return 1.0
        return max(0.0, min(1.0, self.state.fuel / capacity))

    def _observation(self) -> np.ndarray:
        s = self.state
        tx, ty = self._target
        return np.array([
            (s.x - tx) / POS_OBS_SCALE,
            (s.y - ty) / POS_OBS_SCALE,
            s.vx / VEL_OBS_SCALE,
            s.vy / VEL_OBS_SCALE,
            math.sin(s.theta),
            math.cos(s.theta),
            s.omega / (math.pi / 2.0),
            s.phi / PHI_MAX,
            self._fuel_frac(),
            s.wind_x / WIND_OBS_SCALE,
            s.step / self.cfg["max_steps"],
        ], dtype=np.float32)

    def _info(self, impact_speed: float | None) -> dict:
        return {
            "is_success": self._outcome == Outcome.SUCCESS,
            "outcome": self._outcome,
            "fuel_left": self.state.fuel,
            "fuel_frac": self._fuel_frac(),
            "impact_speed": impact_speed,
            "wind_x": self.state.wind_x,
            "step": self.state.step,
        }
```

- [ ] **Step 4: `rocket_env/__init__.py` 구현**

```python
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
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
uv run pytest tests/test_env.py -v
```

기대: 16 passed.

- [ ] **Step 6: 전체 테스트 실행**

```bash
uv run pytest -v
```

기대: 전부 passed.

- [ ] **Step 7: 커밋**

```bash
git add rocket_env/env.py rocket_env/__init__.py tests/test_env.py
git commit -m "$(cat <<'EOF'
feat: RocketEnv Gymnasium 파사드와 환경 등록

rocket-v0 / rocket-landing-v0 / rocket-catch-v0 등록. cfg['seed']는
호출자 메타데이터로 남기고 환경은 읽지 않아, 학습 시 모든 에피소드가
동일해지는 버그를 피한다. 관찰 정규화 상수는 전부 환경 고정값이다.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Exploit 회귀 테스트

**Files:**
- Test: `tests/test_exploit_regression.py`

**Interfaces:**
- Consumes: `RocketEnv`, `PRESETS`, `Outcome`
- Produces: 없음 (테스트 전용). 이 테스트가 보상 명세의 실행 가능한 계약이다.

이 태스크는 코드를 추가하지 않는다. **보상 함수가 의도한 정책 순서를 만드는지**를 검증한다.
단위 테스트로는 잡히지 않는 명세 자체의 결함을 잡는 층이다.

- [ ] **Step 1: 회귀 테스트 작성**

`tests/test_exploit_regression.py`:

```python
"""보상 설계가 만들어내는 '정책 순서'의 회귀 테스트.

여기서 검증하는 것은 코드가 명세대로 도는가가 아니라, 명세가 의도한
인센티브를 만드는가다. 원본 rocket-recycling은 실패 보상에 남은 스텝 수를
곱해서 '빨리 자폭하기'가 고득점 전략이 되었다. 200명이 경사하강법으로
허점을 찾을 것이므로, 어떤 꼼수가 고득점을 받으면 안 되는지를 실행 가능한
형태로 박아둔다.
"""

import pytest

from rocket_env.config import PRESETS, build_config
from rocket_env.env import RocketEnv
from rocket_env.types import Outcome

NOOP = 1        # 추력 0 — 가장 빠른 추락
HOVER = 7       # 추력 1.0g, 노즐 정지 — 오래 버티기
FULL_UP = 10    # 추력 2.0g, 노즐 정지 — 위로 이탈

SEEDS = range(12)

# shaping 총합의 이론적 상한. Φ ∈ [-3.8, 0]이므로 |ΣF| ≤ 3.8.
SHAPING_BOUND = 4.0


def rollout(config, action, seed):
    env = RocketEnv(config=config)
    env.reset(seed=seed)
    total = 0.0
    while True:
        _, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            env.close()
            return total, info


@pytest.mark.parametrize("action", [NOOP, HOVER, FULL_UP])
@pytest.mark.parametrize("preset", list(PRESETS))
def test_no_fixed_action_policy_ever_succeeds(preset, action):
    """단일 행동 반복만으로는 어떤 라운드도 통과할 수 없어야 한다."""
    for seed in SEEDS:
        _, info = rollout(PRESETS[preset], action, seed)
        assert info["outcome"] != Outcome.SUCCESS


@pytest.mark.parametrize("action", [NOOP, HOVER, FULL_UP])
@pytest.mark.parametrize("preset", list(PRESETS))
def test_failure_returns_stay_under_the_failure_ceiling(preset, action):
    """실패 에피소드 점수는 failure_max + shaping 상한을 넘을 수 없다."""
    cfg = build_config(PRESETS[preset])
    ceiling = cfg["reward"]["failure_max"] + SHAPING_BOUND
    for seed in SEEDS:
        total, info = rollout(PRESETS[preset], action, seed)
        assert info["outcome"] != Outcome.SUCCESS
        assert total <= ceiling


def test_loitering_does_not_out_score_descending():
    """오래 버티는 정책이 곧장 목표로 향하는 정책보다 높은 점수를 받으면 안 된다.

    PBRS 총합이 에피소드 길이와 무관하기 때문에 성립하는 성질이다.

    max_steps=250인 이유: NOOP는 약 190스텝에 접지하고, HOVER는 250스텝
    동안 수평 이탈(|x|>300) 없이 버틴다. 두 정책 모두 경계 조건에 걸리지
    않는 구간이라 판정이 안정적이다.
    """
    cfg = {**PRESETS["landing-easy"], "max_steps": 250}
    for seed in SEEDS:
        loiter, loiter_info = rollout(cfg, HOVER, seed)
        descend, descend_info = rollout(cfg, NOOP, seed)
        assert loiter_info["step"] > descend_info["step"]
        assert loiter <= descend + SHAPING_BOUND


def test_shorter_episodes_do_not_earn_more_than_longer_ones():
    """원본 버그의 직접적 회귀 테스트.

    같은 정책·같은 시드에서 max_steps만 줄여 일찍 끊은 에피소드가, 끝까지
    간 에피소드보다 높은 점수를 받으면 안 된다. max_steps는 물리에 영향을
    주지 않으므로 두 궤적은 같은 접두사를 공유하고, 차이는 오직 '언제
    끝났는가'뿐이다. 원본은 실패 보상에 남은 스텝 수를 곱해서 이 부등식이
    뒤집혀 있었다.
    """
    for seed in SEEDS:
        short, _ = rollout({**PRESETS["landing-easy"], "max_steps": 60},
                           HOVER, seed)
        long, _ = rollout({**PRESETS["landing-easy"], "max_steps": 240},
                          HOVER, seed)
        assert short <= long + SHAPING_BOUND


def test_higher_fuel_penalty_lowers_the_return_of_a_burning_policy():
    """연료 패널티는 물리를 바꾸지 않으므로 궤적이 동일하고 점수만 낮아진다."""
    base = {**PRESETS["landing-easy"], "max_steps": 120}
    for seed in SEEDS:
        cheap, _ = rollout({**base, "reward": {"fuel_penalty": 0.0}},
                           HOVER, seed)
        pricey, _ = rollout({**base, "reward": {"fuel_penalty": 0.5}},
                            HOVER, seed)
        assert pricey < cheap
```

- [ ] **Step 2: 테스트 실행**

```bash
uv run pytest tests/test_exploit_regression.py -v
```

기대: 전부 passed. 실패하면 **테스트가 아니라 보상 설계를 고친다.**
특히 `test_crashing_early_is_not_rewarded_more_than_crashing_late`가 실패하면
`terminal_reward`의 실패 분기에 시간 의존성이 들어간 것이다.

- [ ] **Step 3: 커밋**

```bash
git add tests/test_exploit_regression.py
git commit -m "$(cat <<'EOF'
test: 보상 설계의 정책 순서 회귀 테스트

'빨리 자폭'과 '오래 버티기'가 고득점 전략이 되지 않음을 실행 가능한
형태로 고정. 원본 rocket-recycling이 실패한 지점이며 단위 테스트로는
잡히지 않는 층이다.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: 렌더링

**Files:**
- Create: `rocket_env/render.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `State`, `Outcome`, `rocket_env.physics` 상수
- Produces:
  - `rocket_env.render.Renderer(cfg: dict, render_mode: str)`
  - `.reset() -> None`
  - `.draw(state: State, target: tuple[float, float], outcome: str) -> np.ndarray | None`
  - `.close() -> None`
  - 상수 `WIDTH = 640`, `HEIGHT = 960`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_render.py`:

```python
"""렌더링 smoke 테스트.

픽셀 값을 검증하지는 않는다. 크래시 없이 올바른 형태의 배열이 나오는지,
두 태스크와 모든 종료 상태에서 그려지는지만 본다.
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
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_render.py -v
```

기대: `ModuleNotFoundError: No module named 'rocket_env.render'`

- [ ] **Step 3: `rocket_env/render.py` 구현**

```python
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
        pygame.quit()

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

        x_tower = self.cfg["catch"]["x_tower"]
        y_arm = self.cfg["catch"]["y_arm"]
        zone_r = self.cfg["success"]["zone_r"]
        base = self._to_px(x_tower, 0.0)
        top = self._to_px(x_tower, y_arm * 1.25)
        pygame.draw.line(self.surface, TOWER_COLOR, base, top, 8)
        left = self._to_px(x_tower - zone_r * 3.0, y_arm)
        right = self._to_px(x_tower + zone_r * 3.0, y_arm)
        pygame.draw.line(self.surface, ARM_COLOR, left, right, 7)

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
        tip = (length * math.sin(state.phi), -ROCKET_HEIGHT / 2.0 - length)
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
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
SDL_VIDEODRIVER=dummy uv run pytest tests/test_render.py -v
```

기대: 12 passed (parametrize 포함).

- [ ] **Step 5: 전체 테스트 실행**

```bash
SDL_VIDEODRIVER=dummy uv run pytest -v
```

기대: 전부 passed.

- [ ] **Step 6: 커밋**

```bash
git add rocket_env/render.py tests/test_render.py
git commit -m "$(cat <<'EOF'
feat: pygame 벡터 렌더링

착륙 패드와 젓가락 타워, 짐벌 각도가 반영된 화염, 궤적, HUD, 결과 배너를
외부 에셋 없이 전부 벡터로 그린다.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: SB3 통합 확인 + 문서

**Files:**
- Create: `LICENSE`, `README.md`
- Test: `tests/test_sb3_smoke.py`

**Interfaces:**
- Consumes: 등록된 환경 id
- Produces: 없음 (배포 준비 완료 상태)

- [ ] **Step 1: SB3 스모크 테스트 작성**

`tests/test_sb3_smoke.py`:

```python
"""stable-baselines3 DQN 통합 확인.

서버 워커와 학생 노트북이 정확히 이 경로를 쓴다. SB3가 설치되어 있지
않으면 건너뛴다 — SB3는 런타임 의존성이 아니다.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

pytest.importorskip("stable_baselines3")

import gymnasium as gym  # noqa: E402
from stable_baselines3 import DQN  # noqa: E402

import rocket_env  # noqa: F401,E402
from rocket_env.config import PRESETS  # noqa: E402


@pytest.mark.slow
def test_dqn_trains_and_predicts_without_error(tmp_path):
    env = gym.make("rocket-v0", render_mode="rgb_array",
                   config=PRESETS["landing-easy"])
    model = DQN("MlpPolicy", env, verbose=0, device="cpu",
                learning_starts=200, buffer_size=5_000,
                policy_kwargs={"net_arch": [64, 64]})
    model.learn(total_timesteps=5_000)

    path = tmp_path / "model.zip"
    model.save(path)
    loaded = DQN.load(path, env=env, device="cpu")

    obs, _ = env.reset(seed=0)
    action, _ = loaded.predict(obs, deterministic=True)
    assert env.action_space.contains(int(action))
    env.close()


@pytest.mark.slow
def test_server_evaluation_loop_shape_works():
    """서버 워커의 평가 루프와 동일한 형태로 돌려본다."""
    env = gym.make("rocket-v0", render_mode="rgb_array",
                   config=PRESETS["landing-normal"])
    scores, outcomes = [], []
    rng = np.random.default_rng(0)

    for i in range(3):
        obs, _ = env.reset(seed=1000 + i)
        done = truncated = False
        score = 0.0
        info = {}
        while not (done or truncated):
            action = int(rng.integers(env.action_space.n))
            obs, reward, done, truncated, info = env.step(action)
            score += float(reward)
        scores.append(score)
        outcomes.append(bool(info["is_success"]))

    assert len(scores) == 3
    assert all(isinstance(o, bool) for o in outcomes)
    env.close()
```

- [ ] **Step 2: SB3 설치 후 테스트 실행**

```bash
uv pip install "stable-baselines3>=2.3" torch
SDL_VIDEODRIVER=dummy uv run pytest tests/test_sb3_smoke.py -v -m slow
```

기대: 2 passed. (수 분 소요될 수 있다.)

- [ ] **Step 3: `LICENSE` 작성**

```
MIT License

Copyright (c) 2026 Eunsuk Sohn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 4: `README.md` 작성**

````markdown
# rocket-env

강화학습 수업용 로켓 착륙 / 젓가락 포획 환경. Gymnasium 호환.

두 가지 태스크를 제공한다.

- **`landing`** — 착륙 패드에 느리고 수직으로 접지한다.
- **`catch`** — 하강 중 발사탑의 젓가락 팔 높이를 충분히 느리게 통과해 포획된다.
  단 한 번의 기회뿐이다.

바람(정상풍 / 돌풍)과 유한 연료를 config로 조절해 난이도를 만든다.

## 설치

```bash
pip install git+https://github.com/essohn/rocket-env.git
```

의존성은 `gymnasium`, `numpy`, `pygame` 셋뿐이다. 학습 라이브러리는 포함하지 않는다.

## 사용

```python
import gymnasium as gym
import rocket_env  # 환경 등록

env = gym.make("rocket-v0", render_mode="rgb_array",
               config={"task": "catch", "fuel": {"capacity": 140.0}})

obs, info = env.reset(seed=42)
done = truncated = False
while not (done or truncated):
    obs, reward, done, truncated, info = env.step(env.action_space.sample())

print(info["outcome"], info["is_success"], info["impact_speed"])
```

라운드 프리셋:

```python
from rocket_env import PRESETS

env = gym.make("rocket-v0", config=PRESETS["catch-hard"])
```

`landing-easy`, `landing-normal`, `landing-hard`, `catch-normal`, `catch-hard`.

## 관찰과 행동

관찰은 11차원 `Box(float32)`이며, 목표 상대 좌표계를 쓴다.

| # | 성분 | # | 성분 |
|---|------|---|------|
| 0 | `dx / 300` | 6 | `omega / (pi/2)` |
| 1 | `dy / 300` | 7 | `phi / 20deg` |
| 2 | `vx / 50` | 8 | `fuel_frac` |
| 3 | `vy / 50` | 9 | `wind_x / 20` |
| 4 | `sin(theta)` | 10 | `step / max_steps` |
| 5 | `cos(theta)` | | |

행동은 `Discrete(12)` — 추력 `{0, 0.2g, 1.0g, 2.0g}` × 노즐 각속도 `{-30, 0, +30} deg/s`.
인덱스는 `thrust_idx * 3 + nozzle_idx`.

## 보상

- 스텝: 잠재함수 기반 shaping(PBRS) + 연료 패널티. `shaping_gamma=1.0`이라 총합이
  정확히 `Φ(s_T) - Φ(s_0)`로 접혀 에피소드 길이가 점수에 영향을 주지 않는다.
- 성공: 기본점 100 + 접촉 속도 / 위치 정밀도 / 자세 / 잔여 연료 / 시간 효율 보너스 (최대 250).
- 실패: 목표를 향한 진행도에 비례 (0–40). 시간 항이 없으므로 조기 종료가 이득이 되지 않는다.

`config["reward"]`의 가중치는 전부 조정 가능하다. 학습용 보상을 자유롭게 설계하되,
채점은 서버가 정한 평가 설정으로 이뤄진다.

## 개발

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
SDL_VIDEODRIVER=dummy uv run pytest
```

## 출처

물리 모델은 [jiupinjia/rocket-recycling](https://github.com/jiupinjia/rocket-recycling)
(Zhengxia Zou)에서 영감을 받았다. 이 저장소는 코드나 에셋을 복사하지 않은 독립 구현이며
MIT 라이선스로 배포한다.

## 라이선스

MIT — [LICENSE](LICENSE) 참조.
````

- [ ] **Step 5: 전체 테스트 실행 (slow 포함)**

```bash
SDL_VIDEODRIVER=dummy uv run pytest -v
```

기대: 전부 passed.

- [ ] **Step 6: 패키지가 깨끗하게 빌드·설치되는지 확인**

```bash
uv run python -c "
import gymnasium as gym, rocket_env
env = gym.make('rocket-catch-v0', config={'fuel': {'capacity': 100.0}})
obs, info = env.reset(seed=1)
print(obs.shape, obs.dtype, info['outcome'])
"
```

기대: `(11,) float32 in_progress`

- [ ] **Step 7: 커밋**

```bash
git add LICENSE README.md tests/test_sb3_smoke.py
git commit -m "$(cat <<'EOF'
docs: MIT 라이선스, README, SB3 통합 스모크 테스트

서버 워커의 평가 루프와 동일한 형태를 테스트로 고정. SB3는 optional
extra이며 미설치 시 건너뛴다.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## 완료 기준

- [ ] `SDL_VIDEODRIVER=dummy uv run pytest -v` 전부 통과
- [ ] `gym.make("rocket-v0", config=...)` 로 두 태스크 모두 동작
- [ ] `info` 계약 7개 키가 모든 스텝에서 존재
- [ ] 같은 시드로 두 번 돌린 에피소드가 bit-exact 동일
- [ ] `tests/test_exploit_regression.py` 전부 통과 — 고정 행동 정책이 실패 상한을 넘지 못함
- [ ] 런타임 의존성이 `gymnasium`, `numpy`, `pygame` 셋뿐
- [ ] 원본 저장소의 코드·이미지가 하나도 포함되지 않음

## 다음 단계 (이 계획 범위 밖)

1. 학생용 SDK `rocket_helper.py` — `submit-server/modules/highway_helper.py`를 대체
2. 서버 평가 워커 교체 — `backend/courses/ycs1003/evaluation.py`의 문자열 두 개
3. 라운드 커리큘럼과 등급 컷 설계
4. **`render_miss_fall` (캐치 실패 후 낙하 장면 렌더링)** — 종료 후에도 물리를
   계속 굴려야 해서 env 루프에 별도 상태가 필요하다. v1에서는 빼고, 학생 피드백을
   보고 판단한다.
5. **`info["fuel_left"]`가 무한 연료일 때 `inf`를 반환한다.** `json.dumps`는 이를
   비표준 `Infinity`로 직렬화하므로, 서버가 info를 저장·전송한다면 SDK/워커 쪽에서
   `fuel_frac`(항상 유한)을 쓰거나 `inf`를 `None`으로 변환해야 한다.
