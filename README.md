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

env = gym.make("rocket-v0", config=PRESETS["catch"])
```

`landing-basic`, `landing-attitude`, `landing-descent`, `landing-wind`,
`landing-gust`, `catch`. landing 다섯 라운드는 난이도 축을 하나씩만
추가한다. `catch`는 이 규칙 밖이다 — gust를 빼고 바람을 줄이는 대신
성공 임계값 넷(속도·위치·자세·각속도)을 모두 조이는 다른 과제다.

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

행동은 `Discrete(12)` — 추력 `{0, 0.2g, 1.0g, 2.0g}` × 노즐 각속도 `{-120, 0, +120} deg/s`.
인덱스는 `thrust_idx * 3 + nozzle_idx`.

## 보상

- 스텝: 잠재함수 기반 shaping(PBRS) + 연료 패널티. `shaping_gamma=1.0`이라 총합이
  정확히 `Φ(s_T) - Φ(s_0)`로 접혀 에피소드 길이가 점수에 영향을 주지 않는다.
- 성공: 기본점 100 + 접촉 속도 / 위치 정밀도 / 자세 / 잔여 연료 / 시간 효율 보너스 (최대 250).
- 실패(CRASH/MISSED/OUT_OF_FUEL): 성공 판정에 쓰는 다섯 조건(위치·속도·자세·
  각속도) 각각에 대한 근접도 중 가장 나쁜 것으로 채점한다 (0–40). 시간 항이
  없으므로 조기 종료가 이득이 되지 않는다.
- 실패(TIMEOUT): 항상 0점이다. 판정 지점에 가지 않고 시간이 다 되도록
  맴돌기만 하면 시도 자체를 하지 않은 것으로 본다.

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
