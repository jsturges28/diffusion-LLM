"""Tests for durable, origin-independent UI state.

Strategy: ``ui_state`` mirrors the frontend's localStorage values into a
single JSON file under a results directory so they survive desktop-app
restarts (which otherwise change the window origin and orphan
localStorage). These tests use a tmp results dir to prove: a missing
file reads as empty, a set/get round-trips, unknown keys and oversized
or non-string values are rejected, and a corrupt file degrades to
defaults instead of raising. Passing proves the /api/ui-state endpoints
receive safe, bounded, correctly shaped data.

The second half is about not losing writes, which is a different
property from the atomicity the first half assumes. Both halves of
that come from ``DATA-02``: a read-modify-write must hold the lock
across the read, or a value derived from an older snapshot lands on
top of a newer one, and the lock must reach across processes, because
the browser supervisor and the desktop supervisor are two of them
writing one file. The concurrency tests here race real threads and
real processes rather than asserting a lock object exists, since a
lock taken over the wrong span looks identical from the outside.
"""

from __future__ import annotations

import json
import multiprocessing
import threading
import time
from pathlib import Path
from typing import List, Optional

import pytest

from src.web.ui_state import (
    UI_STATE_KEYS,
    load_ui_state,
    mutate_ui_state_key,
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


# -- deriving a new value from the stored one --

KEY = "diffusion_generate_teased"


def test_mutate_sees_the_stored_value(tmp_path: Path) -> None:
    """The point of the helper. A caller that reads first and writes
    later is reading through a gap; this one reads inside the hold."""
    set_ui_state_key(tmp_path, KEY, "one")
    seen: List[Optional[str]] = []

    mutate_ui_state_key(
        tmp_path, KEY, lambda raw: seen.append(raw) or "two"
    )

    assert seen == ["one"]
    assert load_ui_state(tmp_path)[KEY] == "two"


def test_mutate_sees_absence_as_none(tmp_path: Path) -> None:
    seen: List[Optional[str]] = []

    mutate_ui_state_key(
        tmp_path, KEY, lambda raw: seen.append(raw) or "x"
    )

    assert seen == [None]


def test_returning_none_writes_nothing(tmp_path: Path) -> None:
    """The reconcilers take this branch on every page load where no
    run has been deleted, which is nearly all of them."""
    set_ui_state_key(tmp_path, KEY, "keep")
    before = (tmp_path / "ui_state.json").stat().st_mtime_ns

    state = mutate_ui_state_key(tmp_path, KEY, lambda _raw: None)

    assert state[KEY] == "keep"
    assert (tmp_path / "ui_state.json").stat().st_mtime_ns == before


def test_mutate_rejects_an_unknown_key(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        mutate_ui_state_key(
            tmp_path, "not_a_real_key", lambda _raw: "x"
        )


def test_mutate_rejects_an_oversized_value(tmp_path: Path) -> None:
    """The callback is arbitrary code, so its output is checked the
    same way a client's is."""
    limit = UI_STATE_KEYS[KEY]

    with pytest.raises(ValueError):
        mutate_ui_state_key(
            tmp_path, KEY, lambda _raw: "x" * (limit + 1)
        )


# -- not losing writes --


def _append_one(results_dir: Path) -> None:
    """Read-modify-write that loses updates if the span is wrong."""

    def grow(raw: Optional[str]) -> str:
        # Long enough to lose the race on any plausible scheduler if
        # the read and the write are not under one hold.
        time.sleep(0.005)
        return (raw or "") + "x"

    mutate_ui_state_key(results_dir, KEY, grow)


def test_threads_racing_one_key_lose_nothing(tmp_path: Path) -> None:
    """Sixteen appends must produce sixteen characters. A read outside
    the lock produces fewer, and which ones is scheduler-dependent,
    which is exactly why this is raced rather than reasoned about."""
    workers = 16
    threads = [
        threading.Thread(target=_append_one, args=(tmp_path,))
        for _ in range(workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert load_ui_state(tmp_path)[KEY] == "x" * workers


def test_processes_racing_a_key_lose_nothing(tmp_path: Path) -> None:
    """The half a threading lock cannot give. Two supervisors is not
    hypothetical here: the browser entry point and the desktop app are
    separate processes pointed at one results directory."""
    workers = 8
    context = multiprocessing.get_context("fork")
    procs = [
        context.Process(target=_append_one, args=(tmp_path,))
        for _ in range(workers)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=30)

    assert all(proc.exitcode == 0 for proc in procs)
    assert load_ui_state(tmp_path)[KEY] == "x" * workers


def test_a_plain_set_waits_for_a_mutate_in_flight(
    tmp_path: Path,
) -> None:
    """The two entry points share one lock, which is what stops a PUT
    landing inside a reconcile and being overwritten by it."""
    set_ui_state_key(tmp_path, KEY, "start")
    inside = threading.Event()
    release = threading.Event()

    def slow(raw: Optional[str]) -> str:
        inside.set()
        assert release.wait(timeout=10), "main thread never released"
        return (raw or "") + "-mutated"

    worker = threading.Thread(
        target=mutate_ui_state_key, args=(tmp_path, KEY, slow)
    )
    worker.start()
    assert inside.wait(timeout=10), "callback never ran"

    setter = threading.Thread(
        target=set_ui_state_key, args=(tmp_path, KEY, "set")
    )
    setter.start()
    # Still blocked: the mutate holds the lock across its callback.
    setter.join(timeout=0.2)
    assert setter.is_alive()

    release.set()
    worker.join(timeout=10)
    setter.join(timeout=10)
    assert load_ui_state(tmp_path)[KEY] == "set"
