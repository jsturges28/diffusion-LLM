"""Tests for durable, origin-independent UI state.

Strategy: ``ui_state`` mirrors the frontend's localStorage values into a
single JSON file under a results directory so they survive desktop-app
restarts (which otherwise change the window origin and orphan
localStorage). These tests use a tmp results dir to prove: a missing
file reads as empty, a set/get round-trips, unknown keys and oversized
or non-string values are rejected, and a corrupt file degrades to
defaults instead of raising. Passing proves the /api/ui-state endpoints
receive safe, bounded, correctly shaped data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.web.ui_state import (
    UI_STATE_KEYS,
    load_ui_state,
    set_ui_state_key,
)


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_ui_state(tmp_path) == {}


def test_set_then_load_round_trips(tmp_path: Path) -> None:
    value = json.dumps(["2026-01-01_00-00-00_llada"])
    state = set_ui_state_key(tmp_path, "diffusion_new_runs", value)

    assert state["diffusion_new_runs"] == value
    assert load_ui_state(tmp_path)["diffusion_new_runs"] == value


def test_set_multiple_keys_are_independent(tmp_path: Path) -> None:
    set_ui_state_key(tmp_path, "diffusion_settings", "{}")
    set_ui_state_key(tmp_path, "diffusion_generate_teased", "1")

    loaded = load_ui_state(tmp_path)
    assert loaded["diffusion_settings"] == "{}"
    assert loaded["diffusion_generate_teased"] == "1"


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        set_ui_state_key(tmp_path, "not_a_real_key", "x")


def test_non_string_value_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        set_ui_state_key(
            tmp_path, "diffusion_settings", 123  # type: ignore[arg-type]
        )


def test_oversized_value_is_rejected(tmp_path: Path) -> None:
    limit = UI_STATE_KEYS["diffusion_generate_teased"]
    with pytest.raises(ValueError):
        set_ui_state_key(
            tmp_path, "diffusion_generate_teased", "x" * (limit + 1)
        )


def test_corrupt_file_degrades_to_empty(tmp_path: Path) -> None:
    (tmp_path / "ui_state.json").write_text(
        "{not valid json", encoding="utf-8"
    )
    assert load_ui_state(tmp_path) == {}


def test_unknown_keys_in_file_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "ui_state.json").write_text(
        json.dumps(
            {"diffusion_settings": "{}", "stale_key": "drop me"}
        ),
        encoding="utf-8",
    )
    loaded = load_ui_state(tmp_path)
    assert loaded == {"diffusion_settings": "{}"}
