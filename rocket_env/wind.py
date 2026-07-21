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
