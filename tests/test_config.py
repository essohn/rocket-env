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
    """잠금 키는 '알 수 없는 키'가 아니라 잠금 전용 메시지로 거부되어야 한다.

    잠금 키는 스키마에도 없으므로 두 검사의 순서가 뒤바뀌면 unknown-key
    메시지가 나온다. `match=key` 로는 두 메시지 모두에 키 이름이 들어가
    구별하지 못하므로, 잠금 메시지에만 있는 문구를 단언해 순서를 고정한다.
    """
    for key in ("dt", "g", "H", "observation", "action"):
        with pytest.raises(ConfigError, match="환경 상수"):
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


def test_non_positive_v_ref_raises_config_error():
    with pytest.raises(ConfigError, match="v_ref"):
        build_config({"reward": {"v_ref": 0.0}})


def test_none_fuel_capacity_is_allowed():
    assert build_config({"fuel": {"capacity": None}})["fuel"]["capacity"] is None


@pytest.mark.parametrize("name", [
    "landing-basic", "landing-attitude", "landing-descent",
    "landing-wind", "landing-gust", "catch",
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
    ("landing-basic", ("wind", "max_speed"), 0.0),
    ("landing-basic", ("fuel", "capacity"), None),
    ("landing-basic", ("init", "y"), 200.0),
    ("landing-attitude", ("init", "theta_range_deg"), [-30.0, 30.0]),
    ("landing-attitude", ("init", "y"), 200.0),
    ("landing-descent", ("init", "y"), 450.0),
    ("landing-descent", ("wind", "max_speed"), 0.0),
    ("landing-wind", ("wind", "mode"), "constant"),
    ("landing-wind", ("wind", "max_speed"), 8.0),
    ("landing-gust", ("wind", "ou_sigma"), 3.0),
    ("landing-gust", ("fuel", "capacity"), 120.0),
    ("catch", ("success", "zone_r"), 6.0),
    ("catch", ("reward", "w_speed"), 60.0),
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
    eval_cfg = build_config(PRESETS["landing-descent"])
    train_cfg = build_config({**PRESETS["landing-descent"],
                              "reward": {"success_base": 999.0}})
    ok, warnings, errors = validate_train_config(train_cfg, eval_cfg)
    assert ok
    assert errors == []
    assert warnings == []


def test_success_threshold_change_warns_but_passes():
    eval_cfg = build_config(PRESETS["landing-descent"])
    train_cfg = build_config({**PRESETS["landing-descent"],
                              "success": {"v_max": 99.0}})
    ok, warnings, errors = validate_train_config(train_cfg, eval_cfg)
    assert ok
    assert errors == []
    assert any("success.v_max" in w for w in warnings)


def test_task_mismatch_is_an_error():
    eval_cfg = build_config(PRESETS["landing-descent"])
    train_cfg = build_config(PRESETS["catch"])
    ok, warnings, errors = validate_train_config(train_cfg, eval_cfg)
    assert not ok
    assert any("task" in e for e in errors)


# --- Step 1: wind.mode 가 실제 동작을 결정한다 ---

def test_wind_mode_none_forces_zero_wind():
    """max_speed 를 함께 지정하지 않아도 mode 만으로 무풍이 되어야 한다."""
    cfg = build_config({"wind": {"mode": "none"}})
    assert cfg["wind"]["max_speed"] == 0.0
    assert cfg["wind"]["ou_theta"] == 0.0
    assert cfg["wind"]["ou_sigma"] == 0.0


def test_wind_mode_constant_clears_the_ou_terms():
    cfg = build_config({"wind": {"mode": "constant", "max_speed": 5.0,
                                 "ou_sigma": 3.0}})
    assert cfg["wind"]["ou_sigma"] == 0.0


def test_gust_without_sigma_raises_config_error():
    with pytest.raises(ConfigError, match="ou_sigma"):
        build_config({"wind": {"mode": "gust", "max_speed": 10.0}})


# --- Step 2: init.* 와 catch.* 검증 ---

def test_init_y_out_of_bounds_raises_config_error():
    with pytest.raises(ConfigError, match="init.y"):
        build_config({"init": {"y": 900.0}})


def test_init_vy_range_scalar_raises_config_error():
    with pytest.raises(ConfigError, match="vy_range"):
        build_config({"init": {"vy_range": -50.0}})


def test_init_x_range_outside_world_bounds_raises_config_error():
    with pytest.raises(ConfigError, match="x_range"):
        build_config({"init": {"x_range": [-400.0, 400.0]}})


def test_catch_y_arm_out_of_bounds_raises_config_error():
    with pytest.raises(ConfigError, match="catch.y_arm"):
        build_config({"catch": {"y_arm": 900.0}})


@pytest.mark.parametrize("name", [
    "landing-basic", "landing-attitude", "landing-descent",
    "landing-wind", "landing-gust", "catch",
])
def test_every_preset_still_builds_after_range_validation(name):
    """Step 1~3 의 새 검증이 과하지 않은지 확인하는 회귀 체크."""
    assert build_config(PRESETS[name])


# --- Step 3: _reject_unknown_keys 형태 검사 ---

def test_wrong_scalar_type_raises_config_error():
    with pytest.raises(ConfigError, match="max_steps"):
        build_config({"max_steps": "800"})


def test_seed_schema_none_key_is_exempt_from_shape_check():
    """seed 는 스키마 기본값이 None 이라 형태 검사에서 예외로 둔다.

    이 경우 잘못된 값은 여기서 잡히지 않고, 나중에 reset(seed=...) 호출부에서
    터진다. 형태를 알 수 없는 키까지 억지로 검사하면 오탐이 더 위험하다는
    판단이다.
    """
    cfg = build_config({"seed": {"typo": 1}})
    assert cfg["seed"] == {"typo": 1}


def test_fuel_capacity_none_override_still_passes():
    assert build_config({"fuel": {"capacity": None}})["fuel"]["capacity"] is None
