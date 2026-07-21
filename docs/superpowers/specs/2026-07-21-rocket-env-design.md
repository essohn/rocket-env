# rocket-env 설계 명세

**작성일**: 2026-07-21
**대상**: YCS1003 강화학습 프로젝트에서 highway-env를 대체할 로켓 착륙/포획 환경
**상태**: 설계 확정, 구현 계획 작성 대기

---

## 1. 배경과 목표

YCS1003은 학생이 SB3 DQN 에이전트를 학습시켜 제출하면 서버 워커가 성능을 평가해 리더보드와
성적에 반영하는 구조로 운영된다. 현재 환경은 `highway-env`이며, 이를 로켓 착륙·포획 환경으로
**완전 대체**한다.

### 대체 동기

교육적 적합성과 흥미 유발. 부정행위 방지가 아니므로 원본 아이디어를 비틀 이유가 없고,
오히려 물리적으로 정직하게 만드는 것이 목표다.

highway-env 대비 이점:

- **성공/실패가 이진적으로 정의된다.** highway는 보상 가중합이라 "무엇을 잘했는지"가 흐릿하다.
  로켓은 `success_rate`가 그대로 성능 지표가 되며, 서버가 이미 `episode_outcomes`를 받도록
  설계되어 있다.
- **크레딧 할당 문제가 시각적으로 드러난다.** 200스텝 전의 자세 제어 실수가 마지막 착륙 실패로
  이어지는 것을 렌더링으로 관찰할 수 있다.
- **물리가 결정론적이고 설명 가능하다.** 8줄짜리 강체 동역학이라 학생이 실패 원인을 직접
  계산해볼 수 있다. highway의 IDM/MOBIL 차량 모델은 블랙박스에 가깝다.

### 이번 범위

| # | 작업 | 위치 | 이번 스펙 |
|---|------|------|-----------|
| 1 | Gymnasium 호환 rocket 환경 패키지 | 이 저장소 | O |
| 2 | 학생용 SDK (`rocket_helper.py`) | submit-server/modules | 다음 |
| 3 | 서버 평가 워커 교체 | submit-server/backend | 다음 |
| 4 | 라운드 커리큘럼·등급 컷 설계 | submit-server/semesters | 다음 |

#2~#4는 별도 스펙으로 다룬다. 단, 그들이 의존할 **인터페이스 계약**(3절)은 이번 스펙에서 확정한다.

---

## 2. 출처와 라이선스

원본 [jiupinjia/rocket-recycling](https://github.com/jiupinjia/rocket-recycling) (Zhengxia Zou)은
**CC BY-NC-SA 4.0**이다. SA 조항 때문에 파생 저작물은 동일 라이선스에 묶이고, NC 조항은 향후
상업적 활용을 막는다.

따라서 **clean-room 재구현**으로 간다:

- 강체 동역학 방정식과 알고리즘은 저작권 보호 대상이 아니다(저작권은 표현을 보호하지 아이디어를
  보호하지 않는다). 표준 교과서 수준의 물리를 직접 작성한다.
- 렌더링 코드와 배경 이미지 에셋(`landing.jpg`, `hover.jpg`)은 원저자의 창작 표현이므로
  **일절 사용하지 않는다.** pygame으로 새로 그린다.
- **GitHub Fork 버튼을 쓰지 않는다.** Fork는 파생 관계를 메타데이터에 기록하고 원본 커밋
  히스토리를 상속하므로, CC BY-NC-SA 파생물이라는 주장에 근거를 제공한다.
- 신규 저장소 `essohn/rocket-env`로 만들고 **MIT**로 배포한다. README에
  *"Inspired by jiupinjia/rocket-recycling (Zhengxia Zou)"* 로 출처를 밝힌다.

---

## 3. 인터페이스 계약

서버 평가 워커(`backend/courses/ycs1003/evaluation.py`)와 학생 SDK가 의존하는 계약이다.
**이 계약은 기존 highway 소비자 코드에 맞춰 역설계했으므로, 워커 변경은 문자열 두 개로 끝난다**
(`"highway-v0"` → `"rocket-v0"`, `import highway_env` → `import rocket_env`).

| 항목 | 값 |
|------|-----|
| 생성 | `gym.make("rocket-v0", render_mode="rgb_array", config=env_config)` |
| 별칭 | `rocket-landing-v0`, `rocket-catch-v0` (task를 미리 지정한 편의 id) |
| 태스크 선택 | `env_config["task"] = "landing" \| "catch"` |
| 성공 신호 | `info["is_success"]: bool` — 매 스텝 존재 |
| 종료 구분 | `terminated` = 착륙/포획/충돌/포획실패, `truncated` = `max_steps` 도달 |
| 재현성 | `env.reset(seed=...)` — 시드만으로 초기조건과 바람 시퀀스가 완전히 결정된다 |

`info` 딕셔너리 키:

| 키 | 타입 | 설명 |
|----|------|------|
| `is_success` | bool | 성공 여부 |
| `outcome` | str | `"in_progress"` / `"success"` / `"crash"` / `"missed"` / `"timeout"` / `"out_of_fuel"` |
| `fuel_left` | float | 잔여 연료 (무한이면 `inf`) |
| `fuel_frac` | float | 잔여 비율 0–1 (무한이면 1.0) |
| `impact_speed` | float \| None | 접지/포획 판정 시점의 `\|v\|`. 그 외에는 `None` |
| `wind_x` | float | 현재 수평 바람 (m/s) |
| `step` | int | 현재 스텝 |

`impact_speed`는 리더보드에 "평균 접촉 속도" 컬럼으로 노출하면 점수와 별개의 동기부여 장치가 된다.

---

## 4. 패키지 구조

```
rocket_env/
├── __init__.py       # gymnasium.register("rocket-v0" 외 2종)
├── types.py          # State(frozen dataclass), Outcome 문자열 상수
├── env.py            # RocketEnv(gym.Env) — Gymnasium 파사드, config 병합, 관찰 생성
├── physics.py        # 강체 동역학 적분 (태스크·보상 무관, 순수 함수)
├── reward.py         # 보상 설계 — PBRS shaping + 종료 보상
├── config.py         # DEFAULT_CONFIG, 잠금 그룹, 검증, 라운드 프리셋
├── wind.py           # OU 바람 프로세스
├── tasks/
│   ├── __init__.py   # make_task() 팩토리
│   ├── base.py       # Task 인터페이스, 공유 로직(out_of_bounds 등)
│   ├── landing.py    # 지면 착륙
│   └── catch.py      # 젓가락 포획
└── render.py         # pygame 드로잉 (rgb_array / human)
```

`physics.py`를 순수 함수로 분리하는 것은 테스트 편의만이 아니다. 보상과 물리가 한 파일에 섞이면
학생이 "보상을 바꾸면 물리가 바뀌나?"를 구분하지 못한다. 분리해두면 *물리는 고정, 보상은 설계 선택*
이라는 강화학습의 핵심 구분이 파일 구조로 드러난다.

`Task` 인터페이스:

```python
class Task(Protocol):
    def initial_state(self, rng, cfg) -> State: ...
    def target(self, cfg) -> tuple[float, float]:      # (target_x, target_y)
    def evaluate(self, prev: State, cur: State, cfg) -> Outcome | None:
        """종료 조건 판정. 진행 중이면 None."""
```

의존성: `gymnasium>=0.29`, `numpy`, `pygame>=2.5`. Python 3.10+.
`torch`/`stable-baselines3`는 **의존성에 넣지 않는다** — 사용하는 쪽(학생 노트북, 서버 워커) 책임이다.

설치: `pip install git+https://github.com/essohn/rocket-env.git`

---

## 5. 물리

세계 좌표: `x ∈ [-300, 300] m`, `y ∈ [0, 570] m`.
상수: `g = 9.8 m/s²`, 로켓 높이 `H = 50 m`, 관성모멘트 `I = H²/12`, `dt = 0.05 s`.

상태 변수: `x, y, vx, vy, θ, ω, φ, fuel, wind_x, t`
(`θ` = 기체 자세각, `ω` = 각속도, `φ` = 노즐 짐벌각)

### 적분 (semi-implicit Euler)

```
thrust_tangential = -f·sin(φ)     # 노즐이 기울어져 생기는 옆 방향 성분 → 토크를 만든다
thrust_axial      =  f·cos(φ)     # 기체 축 방향 성분 → 기체를 밀어올린다

fx = thrust_tangential·cos(θ) - thrust_axial·sin(θ)
fy = thrust_tangential·sin(θ) + thrust_axial·cos(θ)

# 항력 계수는 종단속도를 설계값으로 두고 역산한다.
# 무동력 낙하가 평형에 이르면 ρ·v_term = g 이므로:
v_term = 49.5 m/s                 # 이 로켓이 무동력 낙하 시 도달하는 속도
ρ      = g / v_term               # ≈ 0.198

ax = fx - ρ·(vx - wind_x)         # 항력은 공기 기준 상대속도에 비례
ay = fy - g - ρ·vy
aω = thrust_tangential·(H/2) / I  # = 6·thrust_tangential / H  (얇은 막대, I = H²/12)

x += vx·dt + 0.5·ax·dt²   ;  vx += ax·dt
y += vy·dt + 0.5·ay·dt²   ;  vy += ay·dt
θ += ω·dt  + 0.5·aω·dt²   ;  ω  += aω·dt
φ  = clip(φ + vφ·dt, -20°, +20°)
```

**항력 계수를 종단속도에서 역산하는 이유.** 계수 자체는 물리 법칙이 정해주지 않는 **설계 선택**이므로,
어떤 수로 표현하든 결국 누군가가 고른 값이다. 그렇다면 의미가 바로 읽히는 양으로 고르는 편이 낫다.
"종단속도 49.5 m/s인 로켓"은 학생이 즉시 이해하고 검증할 수 있는 반면, 같은 값을 다른 경로로 유도하면
숫자만 남고 의도가 사라진다. 테스트도 이 형태에서 더 강해진다 — 시뮬레이션이 **설계된** 종단속도에
도달하는지 확인하는 독립적 검증이 되기 때문이다.

**바람은 새 항이 아니라 기존 항력 항의 수정으로 들어간다.** 항력은 물리적으로 공기 기준 상대속도에
비례해야 하는데 원래 모델은 지면 기준 속도를 썼다. `wind_x = 0`이면 원본과 정확히 일치하고,
별도의 "바람 힘" 파라미터를 튜닝할 필요가 없으며, 학생에게 *바람이란 공기의 기준계가 움직이는 것*
이라는 개념을 코드 한 줄로 보여준다. 바람은 수평 성분만 모델링한다.

### 연료

```
fuel_used = (f / g) · dt          # 1단위 = 1G 추력으로 1초 분사
fuel     -= fuel_used
if fuel <= 0:  f = 0              # 엔진 정지 (이후 무동력 낙하)
```

추력을 중력 단위로 정규화했으므로 "1G로 1초 = 연료 1단위"라는 읽기 쉬운 단위가 된다.
최대추력 2G로 40초를 버티면 80단위가 소모되므로, 용량 80~140 범위가 의미 있는 조절 구간이다.
`capacity: null`이면 무한 연료.

### 바람 (Ornstein–Uhlenbeck)

```
reset:  w = U(-max_speed, max_speed)  if mode != "none" else 0.0
step:   w += ou_theta·(0 - w)·dt + ou_sigma·√dt·N(0,1)
        w  = clip(w, -max_speed, +max_speed)
```

`mode="constant"`는 `ou_theta = ou_sigma = 0`인 특수 케이스라 코드 경로가 하나로 통일된다.
`mode="none"`은 `max_speed = 0`.

---

## 6. 관찰과 행동 공간

### 관찰: `Box(-inf, inf, (11,), float32)`

목표 상대 좌표계를 쓰므로 두 태스크가 동일한 관찰 형식을 공유한다.

| # | 성분 | 정규화 | 비고 |
|---|------|--------|------|
| 0 | `dx = x - target_x` | `/ 300` | |
| 1 | `dy = y - target_y` | `/ 300` | |
| 2 | `vx` | `/ 50` | |
| 3 | `vy` | `/ 50` | |
| 4 | `sin θ` | — | 각도 불연속 제거 |
| 5 | `cos θ` | — | |
| 6 | `ω` | `/ (π/2)` | |
| 7 | `φ` | `/ 20°` | |
| 8 | `fuel_frac` | 0–1 | 무한 연료면 항상 1.0 |
| 9 | `wind_x` | `/ 20.0` | **고정 상수** (아래 참조) |
| 10 | `t_frac` | `step / max_steps` | |

**정규화 상수는 모두 하드코딩된 고정값이며 config에서 파생하지 않는다.**
예컨대 `wind_x`를 라운드별 `wind.max_speed`로 나누면, 같은 관찰값 0.5가 라운드마다 다른
물리적 바람을 뜻하게 되어 정책이 라운드 간에 전이되지 않는다. 관찰 스케일은 환경 상수다.

원본 대비 세 가지를 고쳤다:

1. **성분별 정규화.** 원본은 8차원 전체를 `/100`으로 나눠 각도가 0.03 수준으로 뭉개졌다.
   성분마다 물리적 스케일이 다르므로 따로 나눠야 신경망 입력이 균등해진다.
2. **`sin θ, cos θ` 인코딩.** `θ`를 그대로 넣으면 ±π 근처에서 점프가 생겨 신경망이 물리적으로
   같은 자세를 전혀 다른 입력으로 본다. 단위원 위 두 좌표로 바꾸면 불연속이 사라진다.
3. **`t_frac` 추가.** 에피소드가 `max_steps`에서 잘리는 것은 진짜 종료가 아니라 관측 절단이다.
   시간을 관측하지 않으면 환경이 비정상(non-stationary)으로 보여 가치함수 학습이 흔들린다.

### 행동: `Discrete(12)`

추력 `{0, 0.2g, 1.0g, 2.0g}` × 노즐 각속도 `{-120°/s, 0, +120°/s}`.

인덱스 순서: `action = thrust_idx * 3 + vphi_idx`.

**추력 0을 추가한 것이 핵심이다.** 원본 최소 추력은 0.2g라 엔진이 항상 켜져 있었고, 그 상태에서는
연료 개념을 넣어도 절약 전략이 성립하지 않는다. 엔진을 끌 수 있어야 연료가 의미 있는 자원이 된다.
행동이 9→12로 늘지만 DQN에 부담 없는 크기다.

---

## 7. 태스크 정의

### `landing` — 지면 착륙

| 항목 | 값 |
|------|-----|
| 목표점 | `(0, H/2)` = 패드 중앙 접지 지점 |
| 판정 시점 | `y ≤ H/2` (접지) |
| 성공 조건 | `\|x\| < zone_r` **AND** `\|v\| < v_max` **AND** `\|θ\| < θ_max` **AND** `\|ω\| < ω_max` |
| 기본 임계 | `zone_r=50m`, `v_max=15 m/s`, `θ_max=10°`, `ω_max=10°/s` |
| 실패 | 접지했으나 조건 미달 → `crash` |
| 경계 이탈 | `y ≥ 570 - H/2` 또는 `\|x\| ≥ 300` → `crash` |

### `catch` — 젓가락 포획

| 항목 | 값 |
|------|-----|
| 목표점 | `(x_tower, y_arm)` = `(0, 80)` |
| 판정 시점 | **하강 중 팔 높이를 아래로 가로지르는 스텝**: `y_prev > y_arm ≥ y_cur` **AND** `vy < 0` |
| 성공 조건 | `\|x - x_tower\| < 6m` **AND** `\|v\| < 5 m/s` **AND** `\|θ\| < 5°` **AND** `\|ω\| < 5°/s` |
| 실패 | 판정 미달 → **즉시 `terminated`**, `outcome="missed"` |
| 그 외 | 팔 높이에 도달하지 못하고 접지 → `crash`. 상단 이탈 → `crash` |

`outcome` 판별 순서: 태스크가 `"crash"`를 반환했고 종료 시점에 `fuel_frac == 0`이면
`"out_of_fuel"`로 승격된다. **`"crash"`만 승격 대상이다** — `"missed"`는 그 자체로 이미
명확한 실패 사유라 연료 여부로 덮을 이유가 없고, `"timeout"`을 승격하면 `truncated`(시간
초과)와 `terminated`(연료 소진으로 인한 충돌)의 의미가 뒤섞인다. 리더보드가 이 구분으로
재시도 가능 여부를 판단하므로 여기서는 코드가 옳다.

상승 중 팔 높이를 통과하는 것은 판정하지 않는다(아래에서 위로 지나가는 경우 무시).

**포획 실패 시 즉시 종료**하는 이유: ① 놓친 뒤 80m 낙하가 약 80스텝이라 평가 시간이 10% 늘고,
② 그 구간에서 쌓이는 보상이 점수를 오염시키며, ③ "단 한 번의 기회"라는 메시지가 더 선명해진다.
낙하 장면 관람(`render_miss_fall`)은 종료 후에도 물리를 계속 굴려야 해서 env 루프에 별도 상태가
필요하다. v1 범위에서 제외하고 학생 피드백을 보고 판단한다(12절).

---

## 8. 보상과 점수

### 스텝 보상 — 잠재함수 기반 shaping (PBRS)

```
Φ(s)    = -( w_dist·(|dx|/300 + |dy|/300) + w_att·|wrap(θ)|/(π/2) + w_speed·|v|/50 )
r_shape = γ·Φ(s') - Φ(s)
r_fuel  = -fuel_penalty · fuel_used
r_step  = r_shape + r_fuel
```

`wrap(θ)`는 각도를 `(-π, π]`로 접은 값이다. 물리는 θ를 감지 않아 여러 바퀴 돈 상태에서
`|θ|`가 계속 커지는데, 관찰은 `sin θ, cos θ`라 감긴 횟수를 볼 수 없다. 보상만 감기지 않은
θ에 의존하면 관찰로는 구분 안 되는 두 상태가 다른 보상을 받아 비마르코프가 된다.

PBRS를 쓰는 통상적 이유는 최적 정책 불변성 보장이지만, 여기서 더 중요한 것은 **부수 효과**다.
`shaping_gamma = 1.0`일 때 `r_shape`를 전부 더하면 정확히 망원경처럼 접힌다:

```
Σ_{t=0}^{T-1} (Φ(s_{t+1}) - Φ(s_t)) = Φ(s_T) - Φ(s_0)
```

즉 **shaping 총합이 에피소드 길이와 완전히 무관하다.** 따라서 "목표 근처에서 오래 버티며 점수
쌓기" 꼼수가 구조적으로 차단된다.

총합의 상한은 도달 가능한 상태 전체에서의 Φ 범위가 아니라 **`-Φ(s_0)`, 즉 초기 조건만으로
정해진다** (`Φ(s_T) ≤ 0`이므로). 실제 구현은 이를 프리셋별 초기 분포에서 직접 측정해 상한을
잡는다 — `tests/test_exploit_regression.py`의 `SHAPING_BOUND = 4.0`이 그 값이고,
`test_shaping_bound_covers_every_preset_initial_state`가 매 실행마다 실제로 덮이는지
재확인한다. 200 표본(시드 0) 기준 프리셋별 `-Φ(s_0)` 최악값:

| 프리셋 | `-Φ(s_0)` 최악값 |
|--------|------------------|
| `landing-descent` / `landing-wind` / `landing-gust` | ≈ 2.187 (동률 최댓값) |
| `catch` | ≈ 2.003 |
| `landing-attitude` | ≈ 1.102 |
| `landing-basic` | ≈ 0.970 |

`SHAPING_BOUND = 4.0`은 이 최댓값(≈2.187)에 여유를 둔 상수다. 라운드 설계가 바뀌어 초기
분포가 넓어지면 위 테스트가 실패해 갱신 시점을 알려준다 — 값 자체를 스펙에 하드코딩해
동기화가 깨지게 두지 않는다.

**`shaping_gamma`를 1.0으로 두는 것이 중요하다.** γ < 1이면 접힘이 정확하지 않다:

```
Σ (γΦ(s_{t+1}) - Φ(s_t)) = γΦ(s_T) - Φ(s_0) - (1-γ)·Σ_{t=1}^{T-1} Φ(s_t)
```

`Φ ≤ 0`이므로 마지막 항은 양수이고 **에피소드 길이에 비례해 커진다.** γ=0.99, 800스텝, 평균
Φ=-0.5면 약 +4점, 최악의 경우 +20점이 된다. 실패 상한이 40점인 것을 감안하면 무시할 수 없는
누수이므로, **평가 설정은 항상 `shaping_gamma = 1.0`을 쓴다.**

PBRS의 최적 정책 불변성 보장은 엄밀히는 shaping의 γ와 학습자의 γ가 같을 때 성립한다. 학생은 학습
시 자신의 γ에 맞춰 `shaping_gamma`를 조정할 수 있으며(자유 그룹), 평가는 1.0으로 고정된다.
이 어긋남 자체를 *"왜 감가율이 보상 설계와 얽히는가"* 라는 수업 소재로 쓸 수 있다.

### 종료 보상

**성공**:

```
R = success_base
  + w_speed    · exp(-|v| / v_ref)
  + w_position · (1 - |dx| / zone_r)
  + w_attitude · (1 - |θ|  / θ_max)
  + w_fuel     · fuel_frac
  + w_time     · (1 - t / max_steps)
```

**`timeout`**: 시간이 다 되도록 판정 지점(접지·팔 통과)에 이르지 못했다는 뜻이므로 **항상
0점**이다. 목표 근처에서 맴돌며 점수를 쌓는 경로를 원천 차단한다 — 맴돌기가 착륙을
시도하다 실패하는 쪽보다 높은 점수를 받으면, 최적 전략이 "절대 착륙하지 않기"가 되기
때문이다.

**`crash` / `missed` / `out_of_fuel`** (판정 지점에 실제로 도달한 실패):

```
R = failure_max · min(
        closeness(|x - target_x|, zone_r),
        closeness(|y - target_y|, zone_r),
        closeness(|v|,            v_max),
        closeness(|wrap(θ)|,      θ_max),
        closeness(|ω|,            ω_max),
    )

closeness(value, threshold) = clip(1 - value / (2·threshold), 0, 1)
```

`closeness`는 임계값의 2배 지점에서 0이 되는 선형 근접도이며, 임계값 자체에서 0.5다.
실패 점수를 **성공 판정에 쓰는 다섯 조건(수평 오차·고도 오차·속도·자세·각속도) 각각의
근접도 중 가장 나쁜 것**으로 매기는 것이 원본 버그의 해법이다. 원본은 실패에도
`(max_steps - step_id)`를 곱해 조기 자폭이 고득점이었다. 목표까지의 직선 거리 진행도로
매기던 예전 설계도 문제가 있었다 — 목표가 지면에 있어 중력이 거리를 공짜로 좁혀주는 탓에
자유낙하만으로 실패 점수의 80~89%를 받았다. 평균이 아니라 **최솟값**을 쓰는 이유도
같은 맥락이다 — 평균이면 "속도만 빼고 완벽"이 높은 점수를 받아 같은 허점이 되살아난다.

### 태스크별 기본 프로파일

| 파라미터 | `landing` | `catch` | 근거 |
|----------|-----------|---------|------|
| `success_base` | 100 | 100 | |
| `w_speed` | 40 | **60** | 캐치는 접촉 속도가 핵심 지표 |
| `v_ref` | 5.0 | **2.0** | 참조 속도를 임계값보다 작게 잡아 원점 근처 감도를 높임 |
| `w_position` | 30 | 30 | |
| `w_attitude` | 20 | 20 | |
| `w_fuel` | 30 | 20 | 캐치는 연료보다 정밀도 우선 |
| `w_time` | 30 | 20 | |
| `failure_max` | 40 | 40 | |
| `shaping_gamma` | 1.0 | 1.0 | |
| `shaping_w_dist` | 1.0 | 1.0 | |
| `shaping_w_attitude` | 0.5 | 0.5 | |
| `shaping_w_speed` | 0.5 | 0.5 | Φ에 속도 항을 더해 느린 접근에도 shaping 신호를 준다 |
| `fuel_penalty` | 0.05 | 0.05 | |
| **성공 점수 범위** | 100–250 | 100–250 | |
| **실패 점수 범위** | 0–40 | 0–40 | |

`v_ref`를 성공 임계값(5 m/s)보다 훨씬 작게 잡는 것이 캐치의 변별력을 만든다.
`40·exp(-v/5)`는 v=0에서 40점, v=5에서 14.7점으로 변별폭이 25점뿐이고 곡선이 완만하다.
`60·exp(-v/2)`는 60점 → 4.9점으로 변별폭 55점이며, **마지막 1 m/s를 짜내는 것이 가장 크게
보상된다.** 지수형이 선형보다 나은 이유가 이 원점 근처 고감도다 — 선형이면 5→4와 1→0이 같은
점수라 이미 잘하는 학생에게 더 잘할 동기가 없다.

| `\|v\|` | `40·exp(-v/5)` | `60·exp(-v/2)` |
|------|------|------|
| 0.0 | 40.0 | 60.0 |
| 0.5 | 36.2 | 46.7 |
| 1.0 | 32.7 | 36.4 |
| 2.0 | 26.8 | 22.1 |
| 3.0 | 22.0 | 13.4 |
| 5.0 | 14.7 | 4.9 |

에피소드 점수는 보상 총합이며, 서버 `evaluate_model`이 이미 이 방식으로 합산한다.

---

## 9. Config 스키마와 잠금 정책

### 전체 스키마 (기본값 = 프리셋과 무관한 원본 기본값)

```python
{
  "task": "landing",                      # "landing" | "catch"
  "max_steps": 800,
  "seed": None,

  "wind": {
    "mode": "constant",                   # "none" | "constant" | "gust"
    "max_speed": 8.0,                     # m/s
    "ou_theta": 0.0,                      # 평균회귀 강도 (gust에서만 >0)
    "ou_sigma": 0.0,                      # 변동성 (gust에서만 >0)
  },

  "fuel": {"capacity": 120.0},            # None = 무한

  "init": {
    "y": 450.0,
    "vy_range": [-60.0, -50.0],
    "x_range": [-150.0, 150.0],
    "theta_range_deg": [-45.0, 45.0],
  },

  "success": {
    "v_max": 15.0, "theta_max_deg": 10.0,
    "omega_max_deg": 10.0, "zone_r": 50.0,
  },

  "catch": {"x_tower": 0.0, "y_arm": 80.0},

  "reward": {
    "success_base": 100.0,
    "w_speed": 40.0, "v_ref": 5.0,
    "w_position": 30.0, "w_attitude": 20.0,
    "w_fuel": 30.0, "w_time": 30.0,
    "failure_max": 40.0,
    "shaping_gamma": 1.0,
    "shaping_w_dist": 1.0, "shaping_w_attitude": 0.5, "shaping_w_speed": 0.5,
    "fuel_penalty": 0.05,
  },

}
```

`task="catch"`로 설정하면 `success`와 `reward` 기본값이 캐치 프로파일로 교체된다
(사용자가 명시한 키는 덮어쓰지 않는다). 캐치 프로파일의 `success` 기본값:

```python
"success": {"v_max": 5.0, "theta_max_deg": 5.0, "omega_max_deg": 5.0, "zone_r": 6.0}
```

### 잠금 정책

시스템은 이미 **학습용 환경 설정(학생 자유)** 과 **평가용 환경 설정(서버 고정)** 을 분리하고 있다
(`highway_helper.py:1393-1394`의 `train_env_config` / `eval_env_config`, 평가는 `:1995`의
라운드 설정 사용). 강제되는 것은 `observation` 하나뿐이다.

이 구조를 그대로 유지하되, 무의미한 변경은 차단한다:

| 그룹 | 키 | 정책 |
|------|-----|------|
| 잠금 | `dt`, `g`, `H`, 관찰 정규화 상수, 행동 테이블 | config로 노출하지 않음. 변경 시도는 `ConfigError` |
| 경고 후 허용 | `task`, `success.*`, `init.*`, `catch.*` | 평가 설정과 다르면 경고 출력, 학습은 진행 |
| 자유 | `reward.*` 전부, `wind.*`, `fuel.*`, `max_steps` | 학생 설계 영역 |

검증 함수 `rocket_env.config.validate_train_config(train_cfg, eval_cfg) -> (ok, warnings, errors)`를
제공하여 학생 SDK가 `_validate_observation_config`와 동일한 패턴으로 호출할 수 있게 한다.

**이 분리가 이 과목의 가장 값진 교육 요소일 수 있다.** 학생은 "내가 최적화하는 것(training reward)"과
"내가 평가받는 것(true objective)"이 다르다는 것을 몸으로 겪는다. 보상을 후하게 조작해 학습 곡선을
예쁘게 만들어도 리더보드 점수는 오르지 않는다. 실무 ML에서 대리 지표와 진짜 목표가 어긋나는 문제와
정확히 같은 구조다.

### 라운드 프리셋

`rocket_env.config.PRESETS`에 6종을 제공한다. 라운드마다 난이도 축을 하나씩만
추가한다 — 두 라운드 사이에서 여러 축이 동시에 바뀌면 학생은 무엇이 새로
어려워졌는지 모르고, 조교는 어디서 막히는지 진단할 수 없다. 착륙 라운드는
모두 `success` 임계값(zone_r 포함)을 공유한다 — 난이도는 초기 조건과 외란에서만
온다.

| 프리셋 | task | wind | fuel | y₀ | vy₀ | init θ | zone_r |
|--------|------|------|------|----|----|--------|--------|
| `landing-basic` | landing | none | ∞ | 200 | -20 | ±5° | 50 |
| `landing-attitude` | landing | none | ∞ | 200 | -20 | ±30° | 50 |
| `landing-descent` | landing | none | ∞ | 450 | -50~-40 | ±30° | 50 |
| `landing-wind` | landing | constant 8 m/s | ∞ | 450 | -50~-40 | ±30° | 50 |
| `landing-gust` | landing | gust (σ=3, ±12) | 120 | 450 | -50~-40 | ±30° | 50 |
| `catch` | catch | constant 5 m/s | 140 | 450 | -50~-40 | ±30° | 6 |

축 진행: `landing-basic`이 자세 보정 없이(±5°는 성공 임계 ±10° 안쪽) 감속만
가르치고, `landing-attitude`가 자세 보정을 추가하고(±30°), `landing-descent`가
고도·하강속도를 올리고, `landing-wind`가 일정한 옆바람을, `landing-gust`가
돌풍과 유한 연료를(이 라운드만 축이 둘이며, 둘 다 "예측 불가능한 조건에서
버티기"라는 한 주제다) 추가한다. `catch`는 지면 대신 발사탑 팔 높이를
통과해야 하는 별도 태스크다.

`gust` 프리셋은 `ou_theta = 0.15`를 쓴다. `ou_theta`는 평균회귀 속도로, 0.15에서
바람의 상관 시간이 대략 `1/0.15 ≈ 6.7초`가 된다 — 40초 에피소드 안에 약 6회 방향이 바뀌는 셈이라
"돌풍"으로 체감되면서도 제어 불가능할 만큼 빠르지는 않다.

---

## 10. 렌더링

pygame, 640×960 세로 화면, `render_mode`는 `"rgb_array"`와 `"human"` 모두 지원.
Colab 헤드리스 환경에서는 `SDL_VIDEODRIVER=dummy`로 동작한다.

구성 요소:

- 하늘 그라디언트 배경, 지면
- 태스크별 구조물: **착륙 패드**(landing, 반경 `zone_r` 표시) 또는 **타워 + 젓가락 팔**(catch, `y_arm` 높이)
- 로켓: 본체, 노즈콘, 그리드핀, `φ`만큼 회전한 노즐, 추력에 비례한 길이·색의 화염
- 궤적: 페이드하지 않는다. 최근 400개 점만 유지하고 그 이전은 그대로 잘려나간다
- HUD: 모노스페이스 텍스트 6줄(고도·속도·자세각·연료·바람·스텝). 게이지나 화살표 같은
  그래픽 위젯은 없다 — 숫자를 그대로 읽는 편이 채점 디버깅에 더 유용하다는 판단이다
- 결과 배너: `LANDED / CAUGHT`(성공은 태스크 불문 한 문구) / `CRASHED` / `MISSED` /
  `OUT OF TIME` / `OUT OF FUEL`

배경 이미지 에셋을 쓰지 않고 전부 벡터로 그린다(2절 라이선스 결정).

---

## 11. 테스트 전략

| # | 테스트 | 목적 |
|---|--------|------|
| 1 | `gymnasium.utils.env_checker.check_env` 통과 | API 준수 |
| 2 | 시드 결정론 — 같은 시드 + 같은 행동열 → bit-exact 동일 궤적 | 채점 재현성 |
| 3 | 무풍·무추력 조건에서 해석적 자유낙하 해와 일치 | 물리 정확성 |
| 4 | 성공/실패 임계값 경계 (±ε) 파라미터 테스트 | 판정 정확성 |
| 5 | **exploit 회귀** — `즉시 자폭 ≈ 0점 < 무행동 < 근접 호버 < 성공` | 원본 버그 재발 방지 |
| 6 | PBRS 텔레스코핑 — `sum(r_shape) == Φ(s_T) - Φ(s_0)` (오차 1e-6) | 에피소드 길이 편향 0 검증 |
| 7 | 잠금 config 키 변경 시 `ConfigError` | 학생 설정 검증 |
| 8 | 무풍(`wind_x=0`)에서 항력이 그대로 감속으로 작용하고, 바람과 같은 속도로 움직이면 항력이 0이 됨을 확인 — clean-room 재구현이라 원본 저장소 수치와 비교하는 오라클은 없다 | 바람 도입이 무풍 거동을 물리적으로 일관되게 바꾸는지 확인 |
| 9 | SB3 DQN 5k스텝 smoke — 오류 없이 학습되고 무작위 정책보다 나음 | 통합 확인 |
| 10 | 렌더 smoke — `rgb_array` shape/dtype, 두 태스크 모두 | 렌더 크래시 방지 |

**5번이 가장 중요하다.** 보통 테스트는 "코드가 명세대로 동작하는가"를 검증하지만, 이 테스트는
**"보상 함수가 의도한 정책 순서를 만드는가"** 를 검증한다 — 명세 자체의 결함을 잡는 테스트다.
원본이 정확히 이 지점에서 실패했고, 단위 테스트로는 잡히지 않는다. 200명이 경사하강법으로 허점을
찾아낼 것이므로, **어떤 꼼수가 고득점을 받으면 안 되는지를 실행 가능한 테스트로 박아두는 것**이
보상 명세를 지키는 유일한 방법이다.

9번은 CI에서 느리므로 `@pytest.mark.slow`로 분리한다.

---

## 12. 범위 밖

- 학생용 SDK(`rocket_helper.py`) 작성 및 Colab 노트북
- 서버 평가 워커·리더보드·성적 계산기 변경
- 라운드별 등급 컷과 커리큘럼 설계
- 과거 highway 학기 데이터 마이그레이션
- 연속 행동 공간, 이미지 관찰, 멀티 로켓 등 확장
- `render_miss_fall` — 캐치 실패 후 지면까지 낙하하는 장면의 렌더링 전용 재생
- `info["fuel_left"]`가 무한 연료(`capacity: null`)일 때 `math.inf`를 그대로 반환한다.
  `json.dumps`는 이를 비표준 `Infinity`로 직렬화하므로, 서버가 `info`를 저장·전송한다면
  SDK/평가 워커 스펙에서 이 값 대신 항상 유한한 `fuel_frac`을 쓰거나 `inf → None` 변환을
  명시적으로 정의해야 한다.
