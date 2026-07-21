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
