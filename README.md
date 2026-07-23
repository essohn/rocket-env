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
`landing-gust`, `catch`. 난이도 축을 하나씩만 더해 올라간다. 초반은 낮은
고도라 기본 DQN 으로 학습되고, 후반은 높은 고도라 더 강한 셋업(PPO+정규화)이
필요하다 — 아래 "학습 — 알고리즘 선택" 참고. `catch`는 이 규칙 밖이다:
지면 대신 발사탑 젓가락 높이를 통과해야 하고 성공 임계값 넷(속도·위치·
자세·각속도)을 모두 조인다.

## 관찰과 행동

관찰은 11차원 `Box(float32)`이며, 목표 상대 좌표계를 쓴다.

| # | 성분 | # | 성분 |
|---|------|---|------|
| 0 | `dx / 900` | 6 | `omega / (pi/2)` |
| 1 | `dy / 900` | 7 | `phi / 8deg` |
| 2 | `vx / 200` | 8 | `fuel_frac` |
| 3 | `vy / 200` | 9 | `wind_x / 20` |
| 4 | `sin(theta)` | 10 | `step / max_steps` |
| 5 | `cos(theta)` | | |

행동은 `Discrete(15)` — 추력 `{0, 0.6g, 1.0g, 1.6g, 2.5g}` × 노즐 각속도
`{-120, 0, +120} deg/s`. 인덱스는 `thrust_idx * 3 + nozzle_idx`. 최대 추력이
중량의 2.5배(순감속 1.5g)라, 실제 착륙 로켓처럼 무겁게 내려와 낙하 구간
전체에 걸쳐 아슬아슬하게 감속한다.

## 보상

- 스텝: 잠재함수 기반 shaping(PBRS) + 연료 패널티. `shaping_gamma=1.0`이라 총합이
  정확히 `Φ(s_T) - Φ(s_0)`로 접혀 에피소드 길이가 점수에 영향을 주지 않는다.
- 성공/실패는 경계에서 **연속**이다. 실패(CRASH/MISSED/OUT_OF_FUEL)는 성공
  판정 조건(위치·속도·자세·각속도)에 대한 도달도 중 가장 나쁜 것으로 0–40점.
  각 축이 임계값에 닿으면 40점에 이른다. 성공은 그 40점을 바닥으로, **접지
  속도가 낮을수록**(가장 중요한 축) 그리고 위치·연료가 좋을수록 최대 220점까지
  이어진다. 경계에 점프가 없어 학습이 성공을 향해 매끄럽게 나아간다.
- 실패(TIMEOUT): 항상 0점이다. 판정 지점에 가지 않고 시간이 다 되도록
  맴돌기만 하면 시도 자체를 하지 않은 것으로 본다.

`config["reward"]`의 가중치는 전부 조정 가능하다. 학습용 보상을 자유롭게 설계하되,
채점은 서버가 정한 평가 설정으로 이뤄진다.

## 학습 — 알고리즘 선택

라운드마다 필요한 알고리즘의 급이 다르다. `scripts/train.py`가 두 셋업을 모두
제공한다.

```bash
uv run python scripts/train.py --preset landing-basic --algo dqn --steps 400000
uv run python scripts/train.py --preset landing-gust  --algo ppo --steps 1500000
```

- **`--algo dqn`** — 기본 DQN. 초반 라운드(`landing-basic`, `landing-attitude`)를
  학습한다. 낮은 고도의 짧은 낙하에 충분하다.
- **`--algo ppo`** — PPO + 관측/보상 정규화(VecNormalize). 후반 고고도 라운드는
  낙하 구간이 길어 바닐라 DQN 의 신용 할당이 무너지지만(1M 스텝에도 발산),
  이 셋업은 학습해낸다. 실측: 고고도 라운드에서 DQN 0% vs PPO+정규화 ~40%.

즉 알고리즘을 넘어설수록 더 높은 라운드를 통과할 수 있다. 라운드별 측정치는
[docs/baselines.md](docs/baselines.md) 참고.

## Colab에서 학습하기

화면·GPU 없이 브라우저만으로 학습·평가·영상까지 되는 자립형 노트북:
[`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb). Colab에 업로드하거나
GitHub 경로로 열면 된다(`https://colab.research.google.com/github/essohn/rocket-env/blob/main/notebooks/colab_train.ipynb`).

설치는 한 줄 — 환경·SB3·영상 인코더까지:

```bash
pip install "rocket-env[sb3] @ git+https://github.com/essohn/rocket-env.git" imageio imageio-ffmpeg
```

Colab은 화면이 없으므로 첫 `pygame` import 전에 `os.environ["SDL_VIDEODRIVER"]="dummy"`를
설정한다(노트북에 포함). 학습·평가·녹화 코드는 노트북에 자체 포함돼 있어 `pip install`
만으로 충분하다(`scripts/`는 설치에 포함되지 않는다).

## 직접 조종해 보기 (키보드)

정책 없이 사람이 직접 착륙/포획을 시도해 볼 수 있다 — 환경의 조작감과 난이도를
몸으로 느끼기에 좋다. 두 가지 방법:

**웹(설치 불필요)** — [`web/play.html`](web/play.html). 파이썬 물리를 그대로
포팅한 단일 HTML 파일이라 브라우저로 바로 열린다. Colab처럼 화면 없는 환경에서
쓰던 학생도 링크만으로 조종해 볼 수 있다. `↑/↓` 추력, `←/→` 노즐, `R` 재시작,
`1`–`4` 라운드.

**파이썬(로컬, pygame 창)** —

```bash
uv run python scripts/play.py --preset landing-basic --livery 내별명
```

`↑/↓`(추력 단계), `←/→`(노즐), `R`(재시작), `Esc`(종료).

## 로켓 도색 (별명)

`config["livery"]`에 문자열을 넣으면 로켓 몸통에 세로로 새겨진다. 학생이 자기
별명을 넣어 훈련 영상에서 구분할 수 있다. 로켓 길이를 넘는 글자는 잘린다.

```python
env = gym.make("rocket-v0", config={"task": "landing", "livery": "Alice"})
```

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
