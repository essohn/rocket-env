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
