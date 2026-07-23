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
from rocket_env.reward import potential, shaping, terminal_reward
from rocket_env.tasks import make_task
from rocket_env.types import Outcome
from rocket_env.wind import WindProcess

OBS_DIM = 11

# 관찰 정규화 상수. 전부 환경 고정값이며 config에서 파생하지 않는다.
# config에서 파생하면 같은 관찰값이 라운드마다 다른 물리량을 뜻하게 되어
# 정책이 라운드 간에 전이되지 않는다.
POS_OBS_SCALE = 900.0
VEL_OBS_SCALE = 200.0
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
        self._potential = 0.0
        self._initial_potential = 0.0
        self._outcome = Outcome.IN_PROGRESS
        self._last_action = None
        self._score = None            # 종료 시 지급되는 착륙 점수(연속 종단보상)

    # --- Gymnasium API ---

    def reset(self, *, seed=None, options=None):
        # cfg["seed"]는 의도적으로 읽지 않는다. 라운드 시스템이 에피소드마다
        # reset(seed=base_seed + i)로 넘겨주는 호출자 측 메타데이터다.
        super().reset(seed=seed)

        state = self.task.initial_state(self.np_random, self.cfg)
        wind_x = self.wind.reset(self.np_random)
        self.state = replace(state, wind_x=wind_x)

        self._target = self.task.target(self.cfg)
        self._potential = potential(self.state, self._target, self.cfg)
        # 종료 시 되돌려주어 shaping 총합을 정확히 0으로 만든다.
        self._initial_potential = self._potential
        self._outcome = Outcome.IN_PROGRESS
        self._last_action = None
        self._score = None

        if self._renderer is not None:
            self._renderer.reset()

        return self._observation(), self._info(impact_speed=None)

    def step(self, action):
        self._last_action = int(action)   # 렌더러 행동 시각화용
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

        # 종료 상태의 Φ 는 0으로 둔다. 그러면 망원경 합이 -Φ(s₀) 가 되고,
        # 아래에서 Φ(s₀) 를 되돌려주면 총합이 정확히 0이 된다. shaping 은
        # 학습 신호로만 작동하고 점수에는 기여하지 않는다.
        cur_potential = (0.0 if outcome is not None
                         else potential(cur, self._target, self.cfg))
        reward = (shaping(self._potential, cur_potential, self.cfg)
                  - self.cfg["reward"]["fuel_penalty"] * used)
        self._potential = cur_potential

        impact_speed = None
        if outcome is not None:
            self._outcome = outcome
            reward += self._initial_potential
            term_r = terminal_reward(outcome, cur, self._target, self.cfg,
                                     self._fuel_frac())
            reward += term_r
            self._score = term_r      # 착륙 점수 = 종단보상(수동 모드 표시·채점용)
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
        # grip=None 이면 렌더러가 근접도로 점진적 오므림을 계산한다 — 로켓이
        # 다가올수록 팔이 서서히 닫혀 순간이동처럼 보이지 않는다. 연출용일
        # 뿐 물리에는 영향이 없다.
        return self._renderer.draw(self.state, self._target, self._outcome,
                                   grip=None, action_info=self._last_action)

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
            "score": self._score,
            "fuel_left": self.state.fuel,
            "fuel_frac": self._fuel_frac(),
            "impact_speed": impact_speed,
            "wind_x": self.state.wind_x,
            "step": self.state.step,
        }
