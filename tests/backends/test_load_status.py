"""Tests for the worker's /health load-status decision.

What is tested: ``resolve_load_status``, which turns three signals
(did the load fail, is the model ready, and what progress dict is the
backend holding) into the status string the supervisor polls.

Strategy: call it directly with each combination, including the two
progress shapes that now flow through the same attribute. The function
is pure, so no worker or model is needed.

A pass proves that a download and a memory load are told apart by the
``phase`` key, that a payload without one is still read as a download
(which is what ``hf_download`` sends), and that failure and readiness
outrank any progress left lying around.
"""

from __future__ import annotations

from typing import Any

from src.backends.worker_base import resolve_load_status


def test_no_progress_is_a_plain_load() -> None:
    """The gap before the sampler attaches: still loading, no bar."""
    status = resolve_load_status(
        failed=False, ready=False, progress=None
    )
    assert status == "loading"


def test_a_phaseless_payload_is_a_download() -> None:
    """hf_download predates the phase key and must keep its status."""
    status = resolve_load_status(
        failed=False,
        ready=False,
        progress={"fraction": 0.4, "total_bytes": 100},
    )
    assert status == "downloading"


def test_an_explicit_download_phase_is_a_download() -> None:
    status = resolve_load_status(
        failed=False,
        ready=False,
        progress={"phase": "download", "fraction": 0.4},
    )
    assert status == "downloading"


def test_a_load_phase_is_a_load() -> None:
    """The point of the key: one attribute, two statuses."""
    status = resolve_load_status(
        failed=False,
        ready=False,
        progress={"phase": "load", "fraction": 0.4},
    )
    assert status == "loading"


def test_an_unknown_phase_falls_back_to_download() -> None:
    status = resolve_load_status(
        failed=False,
        ready=False,
        progress={"phase": "something-else"},
    )
    assert status == "downloading"


def test_a_non_dict_progress_is_ignored() -> None:
    """Defensive: only a dict can carry a phase, so anything else is
    treated as no progress rather than probed."""
    not_a_dict: Any = "0.5"
    status = resolve_load_status(
        failed=False, ready=False, progress=not_a_dict
    )
    assert status == "loading"


def test_ready_outranks_leftover_progress() -> None:
    """The sampler's final emit lands at 100% just as the load
    finishes, so a ready worker routinely still holds a progress
    dict."""
    status = resolve_load_status(
        failed=False,
        ready=True,
        progress={"phase": "load", "fraction": 1.0},
    )
    assert status == "ready"


def test_failure_outranks_everything() -> None:
    status = resolve_load_status(
        failed=True,
        ready=True,
        progress={"phase": "load", "fraction": 1.0},
    )
    assert status == "error"
