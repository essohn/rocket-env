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
