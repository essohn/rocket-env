"""환경 설정: 기본값, 태스크 프로파일, 라운드 프리셋, 검증.

설계 원칙 — 학생은 **학습용** 설정을 자유롭게 바꿀 수 있고, 채점은 **평가용**
설정으로 이뤄진다. 이 어긋남 자체가 수업의 핵심 교보재다: 내가 최적화하는 것과
내가 평가받는 것은 다르다.

다만 물리 상수처럼 바꿔봐야 다른 문제를 푸는 셈인 키는 잠근다.
"""

import copy
from typing import Any

from rocket_env.physics import ROCKET_HEIGHT, WORLD_X_MAX, WORLD_X_MIN, WORLD_Y_MAX

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

    # 로켓 몸통에 세로로 새길 문구(순전히 연출용, 물리·보상 무관). 학생이
    # 자기 별명을 넣을 수 있다. 로켓 길이를 넘는 글자는 렌더러가 잘라낸다.
    "livery": "YONSEI인이활",

    "wind": {
        "mode": "constant",   # "none" | "constant" | "gust"
        "max_speed": 8.0,
        "ou_theta": 0.0,
        "ou_sigma": 0.0,
    },

    "fuel": {"capacity": 140.0},   # None이면 무한

    "init": {
        "y": 450.0,
        "vy_range": [-60.0, -50.0],
        "x_range": [-150.0, 150.0],
        "theta_range_deg": [-45.0, 45.0],
        # 초기 각속도의 크기 범위(deg/s). 부호는 매 에피소드 무작위.
        # 순수 난이도 축이다 — 스핀이 클수록 짐벌로 자세를 잡기 어렵다.
        # (개루프 방지는 이제 물리가 맡는다: 종단속도 250 m/s 라 관찰을 안
        # 읽는 duty-cycle 정책의 종단속도가 v_max 를 넘어 구조적으로 실패한다.)
        "omega_abs_range_deg": [6.0, 10.0],
    },

    "success": {
        "v_max": 15.0,
        "theta_max_deg": 10.0,
        "omega_max_deg": 10.0,
        "zone_r": 50.0,
    },

    "catch": {"x_tower": 0.0, "y_arm": 120.0},

    "reward": {
        # 종료 보상. 실패↔성공이 경계에서 연속이 되도록 설계했다(reward.py
        # 참조). failure_max 는 실패 상한이자 성공의 바닥이다 — 학생이 요청한
        # "실패 구간 0~40"을 지키면서 절벽을 없앤다.
        "failure_max": 40.0,
        # 성공 보너스: 접지 속도가 낮을수록 크다. soft_v_ref 가 작을수록
        # 아주 느린 착지에만 큰 점수를 몰아준다.
        "success_soft": 140.0,
        "soft_v_ref": 3.0,
        "success_position": 20.0,
        "success_fuel": 20.0,
        # 1.0이어야 shaping 총합이 정확히 Φ(s_T) - Φ(s_0)로 접힌다.
        # γ<1이면 에피소드가 길수록 shaping 총합이 커지는 편향이 생긴다.
        "shaping_gamma": 1.0,
        # 총합이 정확히 0이므로 점수에는 영향이 없다. 값이 클수록 스텝당
        # 학습 신호가 강해진다. 예전 값(1.0/0.5/0.5)에서는 shaping 이
        # 리턴의 1% 미만이라 탐색을 전혀 안내하지 못했다.
        "shaping_w_dist": 20.0,
        "shaping_w_attitude": 10.0,
        "shaping_w_speed": 10.0,
        "fuel_penalty": 0.05,
    },
}

# task="catch"일 때 갈아끼우는 프로파일. 사용자가 명시한 값은 덮지 않는다.
#
# success 네 값은 원래 landing 대비 8.3배/3배/2배/2배로 동시에 조여져
# 있었다 — 난이도 축을 한 번에 넷 올리는 셈이라 절벽이었다. 1M 스텝, 3개
# 독립 시드로 측정한 성공률이 전부 0.0%(리턴 -2.01/-1.98/-1.89, 연료
# 페널티만 남은 값)였고, 시드 간 분산이 거의 없다는 점이 핵심이다 —
# 운 나쁜 시드가 아니라 어떤 시드도 성공 궤적 근처에도 못 갔다는 뜻이다.
# 손으로 짠 PD 컨트롤러는 이 라운드에서 약 50%를 내므로 과제 자체는
# 풀리는 문제이지만, 이 학습 예산으로는 배울 수 없는 문제였다. 그래서
# landing과 기존 catch 값의 중간쯤으로 완화한다.
CATCH_OVERRIDES: dict[str, Any] = {
    "success": {
        "v_max": 8.0,
        "theta_max_deg": 8.0,
        "omega_max_deg": 8.0,
        "zone_r": 15.0,
    },
    "reward": {
        # 캐치는 접촉 속도가 핵심 지표다. 속도 보너스를 키우고(140→200)
        # 참조 속도를 더 조여(3→1.5) 아주 느린 포획에만 큰 점수를 몰아준다.
        "success_soft": 200.0,
        "soft_v_ref": 1.5,
    },
}

PRESETS: dict[str, dict[str, Any]] = {
    # 여섯 라운드는 난이도 축을 하나씩만 더한다. 여러 축을 동시에 올리면
    # 학생은 무엇이 새로 어려워졌는지 모르고, 조교는 어디서 막히는지 진단할
    # 수 없다. catch는 이 규칙 밖이다 — 아래 주석 참고.
    #
    # 핵심 설계: 하이브리드 사다리. 초반 라운드는 낮은 고도라 기본 DQN 으로
    # 학습되고, 후반 라운드는 높은 고도라 더 강한 셋업(PPO + 관측/보상
    # 정규화)이 필요하다. 실측으로 확인한 사실 — 높은 고도(≥500m)는 바닐라
    # DQN 이 1M 스텝에도 발산해 0% 지만, PPO+VecNormalize 는 y=800 에서 ~40%
    # 를 낸다. 이 어긋남이 과제의 변별력을 만든다: 알고리즘을 넘어서는 학생이
    # 더 많은 라운드를 통과한다. (docs/training.md 의 알고리즘 선택지 참고.)
    #
    # x_range(수평 퍼짐)는 그 자체로 난이도 축이다: 옆이동에는 기울임이
    # 필요해 자세 축과 곱셈적으로 상호작용한다. 라운드가 올라갈수록 단조
    # 비감소로 둔다.
    #
    # omega_abs_range_deg(초기 스핀)는 이제 순수 난이도 축이다. 예전에는
    # 개루프(관찰 미사용) 정책을 막는 방어였지만, 종단속도를 250 m/s 로
    # 올린 뒤로는 항력 끌개의 종단속도가 v_max 를 크게 넘어 개루프가
    # 구조적으로 실패한다(test_exploit_regression 로 고정). 그래서 스핀을
    # 학습 가능한 수준으로 낮춰 난이도 조절에만 쓴다.
    "landing-basic": {
        "task": "landing",
        "wind": {"mode": "none", "max_speed": 0.0},
        "fuel": {"capacity": None},
        "init": {"y": 250.0, "vy_range": [-20.0, -20.0],
                 "x_range": [-30.0, 30.0], "theta_range_deg": [-5.0, 5.0],
                 "omega_abs_range_deg": [6.0, 10.0]},
    },
    # + 자세 보정 & 착륙 데미지 채점. 기울기가 성공 임계(±10°)를 벗어나고,
    # 접근 속도가 빨라지며(vy -28), 착륙 데미지가 엄격해진다: 접지 속도 임계
    # v_max 12, 데미지 민감도 soft_v_ref 2.5(작을수록 같은 충격이 더 큰 감점).
    # 여기부터는 바닐라 DQN 이 학습하지 못해(0%) PPO+정규화가 필요하므로,
    # 난이도를 올려도 DQN 입문 라운드(basic)를 깨지 않는다.
    "landing-attitude": {
        "task": "landing",
        "wind": {"mode": "none", "max_speed": 0.0},
        "fuel": {"capacity": None},
        "success": {"v_max": 12.0},
        "reward": {"soft_v_ref": 2.5, "success_soft": 150.0},
        "init": {"y": 300.0, "vy_range": [-28.0, -28.0],
                 "x_range": [-30.0, 30.0], "theta_range_deg": [-20.0, 20.0],
                 "omega_abs_range_deg": [6.0, 10.0]},
    },
    # + 고도 상승 & 데미지 강화: 낙하 구간이 길고 접근 속도(vy -40)가 빨라
    # 제동을 더 일찍 정확히 시작해야 하며, 착륙 데미지 채점이 엄격하다
    # (v_max 10, soft_v_ref 2.0).
    "landing-descent": {
        "task": "landing",
        "wind": {"mode": "none", "max_speed": 0.0},
        "fuel": {"capacity": None},
        "success": {"v_max": 10.0},
        "reward": {"soft_v_ref": 2.0, "success_soft": 160.0},
        "init": {"y": 550.0, "vy_range": [-40.0, -40.0],
                 "x_range": [-60.0, 60.0], "theta_range_deg": [-20.0, 20.0],
                 "omega_abs_range_deg": [8.0, 12.0]},
    },
    # + 외란: 에피소드 내내 일정한 옆바람. 데미지 채점 더 엄격(v_max 9).
    "landing-wind": {
        "task": "landing",
        "wind": {"mode": "constant", "max_speed": 8.0},
        "fuel": {"capacity": None},
        "success": {"v_max": 9.0},
        "reward": {"soft_v_ref": 2.0, "success_soft": 160.0},
        "init": {"y": 550.0, "vy_range": [-40.0, -40.0],
                 "x_range": [-60.0, 60.0], "theta_range_deg": [-25.0, 25.0],
                 "omega_abs_range_deg": [8.0, 12.0]},
    },
    # + 고고도·불확실성·자원: 사실적 고고도 낙하에 돌풍과 유한 연료가 겹친다.
    # 이 라운드가 시네마틱 데모(y≈1400)에 가장 가까운 '진짜 로켓 착륙'이며,
    # 착륙 데미지 채점이 가장 엄격하다(v_max 8, soft_v_ref 1.5). PPO+정규화로도
    # 성공률이 낮은 상위권 도전 과제다.
    "landing-gust": {
        "task": "landing",
        "wind": {"mode": "gust", "max_speed": 12.0,
                 "ou_theta": 0.15, "ou_sigma": 3.0},
        # 최대 소모는 fuel_cost(2.5g) * max_steps = 0.125 * 800 = 100 이다.
        # 70 은 그 상한보다 낮아 소모 관리가 실제로 필요하다.
        "fuel": {"capacity": 70.0},
        "success": {"v_max": 8.0},
        "reward": {"soft_v_ref": 1.5, "success_soft": 180.0},
        "init": {"y": 800.0, "vy_range": [-45.0, -45.0],
                 "x_range": [-90.0, 90.0], "theta_range_deg": [-30.0, 30.0],
                 "omega_abs_range_deg": [10.0, 14.0]},
    },
    # + 정밀 포획: 지면 대신 발사탑 팔 높이를 통과해야 한다. 다른 다섯
    # 라운드와 달리 축을 하나 얹는 게 아니라 아예 다른 과제다 — 성공 임계값
    # 넷(속도·위치·자세·각속도)을 모두 조인다.
    "catch": {
        "task": "catch",
        "wind": {"mode": "constant", "max_speed": 5.0},
        # 80 은 fuel_cost(2.5g) * max_steps = 100 보다 낮아 소모 관리가 필요하다.
        "fuel": {"capacity": 80.0},
        "init": {"y": 700.0, "vy_range": [-35.0, -35.0],
                 "x_range": [-60.0, 60.0], "theta_range_deg": [-25.0, 25.0],
                 "omega_abs_range_deg": [8.0, 12.0]},
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
        expected = schema[key]
        if _kind(expected) != "none" and _kind(value) not in (_kind(expected), "none"):
            raise ConfigError(
                f"{full!r} 의 형태가 스키마와 다릅니다: "
                f"{_kind(value)} (스키마는 {_kind(expected)})")
        if isinstance(value, dict) and isinstance(schema[key], dict):
            _reject_unknown_keys(value, schema[key], path=f"{full}.")


def _kind(value) -> str:
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, (list, tuple)):
        return "list"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "str"
    return "none"


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

    _normalize_wind(cfg, user.get("wind", {}))
    _validate_ranges(cfg)
    return cfg


def _normalize_wind(cfg: dict, user_wind: dict) -> None:
    """mode 가 실제 동작을 결정하도록 동반 값을 맞춘다.

    WindProcess 는 mode 를 읽지 않고 max_speed/ou_theta/ou_sigma 만 본다.
    사용자가 명시하지 않은 값(기본값 상속)은 조용히 채워 넣지만, 사용자가
    명시적으로 지정한 값이 mode 와 모순되면 조용히 0으로 덮지 않고
    ConfigError 를 낸다 — 정당한 키를 말없이 버리는 것은 `_reject_unknown_keys`
    가 막으려던 바로 그 실패 모드다.
    """
    wind = cfg["wind"]
    if wind["mode"] == "none":
        for key in ("max_speed", "ou_theta", "ou_sigma"):
            if key in user_wind and user_wind[key] != 0.0:
                raise ConfigError(
                    f"wind.mode='none' 인데 wind.{key}={user_wind[key]!r} 로 "
                    "모순됩니다. mode를 바꾸거나 값을 0으로 하세요.")
        wind["max_speed"] = 0.0
        wind["ou_theta"] = 0.0
        wind["ou_sigma"] = 0.0
    elif wind["mode"] == "constant":
        for key in ("ou_theta", "ou_sigma"):
            if key in user_wind and user_wind[key] != 0.0:
                raise ConfigError(
                    f"wind.mode='constant' 인데 wind.{key}={user_wind[key]!r} "
                    "로 모순됩니다. mode='gust'를 쓰거나 값을 0으로 하세요.")
        wind["ou_theta"] = 0.0
        wind["ou_sigma"] = 0.0
    elif wind["mode"] == "gust" and wind["ou_sigma"] <= 0.0:
        raise ConfigError(
            "wind.mode='gust' 인데 ou_sigma 가 0 이하입니다 — 돌풍이 아니라 "
            f"상수 바람이 됩니다: ou_sigma={wind['ou_sigma']}"
        )


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

    # soft_v_ref 는 exp(-speed / soft_v_ref) 의 분모다. 0이면
    # ZeroDivisionError, 음수면 속도가 빠를수록 점수가 오르도록 부호가 뒤집힌다.
    if cfg["reward"]["soft_v_ref"] <= 0:
        raise ConfigError(
            f"reward.soft_v_ref는 양수여야 합니다: {cfg['reward']['soft_v_ref']}"
        )

    ground = ROCKET_HEIGHT / 2.0
    ceiling = WORLD_Y_MAX - ROCKET_HEIGHT / 2.0

    init = cfg["init"]
    if not ground < init["y"] < ceiling:
        raise ConfigError(
            f"init.y는 {ground}와 {ceiling} 사이여야 합니다: {init['y']}")
    for key in ("x_range", "vy_range", "theta_range_deg", "omega_abs_range_deg"):
        pair = init[key]
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2
                and pair[0] <= pair[1]):
            raise ConfigError(
                f"init.{key}는 [최솟값, 최댓값] 2원소여야 합니다: {pair!r}")
    if not (WORLD_X_MIN < init["x_range"][0]
            and init["x_range"][1] < WORLD_X_MAX):
        raise ConfigError(f"init.x_range가 세계 경계를 벗어납니다: {init['x_range']}")
    # 하한이 양수가 아니면 개루프(짐벌 미사용) 정책이 |omega|=0 을 뽑을 수
    # 있어, 이 범위를 두는 이유(성공 임계 밖으로 못 벗어나게 하기)가 깨진다.
    if init["omega_abs_range_deg"][0] <= 0:
        raise ConfigError(
            f"init.omega_abs_range_deg의 최솟값은 양수여야 합니다: "
            f"{init['omega_abs_range_deg']!r}")
    if not ground < cfg["catch"]["y_arm"] < ceiling:
        raise ConfigError(
            f"catch.y_arm은 {ground}와 {ceiling} 사이여야 합니다: "
            f"{cfg['catch']['y_arm']}")
    if not WORLD_X_MIN < cfg["catch"]["x_tower"] < WORLD_X_MAX:
        raise ConfigError(
            f"catch.x_tower가 세계 경계를 벗어납니다: {cfg['catch']['x_tower']}")


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
