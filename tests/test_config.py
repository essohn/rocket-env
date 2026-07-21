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
