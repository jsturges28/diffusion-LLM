"""The browser tells the worker when an edit session opens.

Strategy: source inspection of `app.js`, the approach this repo uses
for its classic-script pages until `ORG-02` gives them a testable
seam. The worker's half is exercised properly in
`tests/backends/test_llada_resume_state.py`.

A resume replaces the worker's retained history with the branch it
produced, and every way of abandoning a session rolled back only the
browser. The two then disagreed about which frames exist, and a
second edit at the same or a later frame remasked a canvas from the
branch the user had just discarded while they clicked tokens on the
original. It was silent, because a wrong canvas still denoises into
fluent text.

Passing proves the rewind is attached to the one place a session can
open rather than to each of the ways one can close, since two of
those ways (a reload, a closed tab) cannot send anything at all.
"""

from __future__ import annotations

import re
from pathlib import Path

APP_JS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "web"
    / "static"
    / "app.js"
)


def _source() -> str:
    return APP_JS.read_text(encoding="utf-8")


def _region(anchor: str, chars: int) -> str:
    source = _source()
    start = source.find(anchor)
    assert start != -1, (
        f"anchor {anchor!r} is gone from app.js; update this test"
        " rather than deleting it"
    )
    return source[start : start + chars]


# -- where it is sent from --


def test_opening_a_session_rewinds_the_worker() -> None:
    region = _region("function captureEditSnapshot()", 600)

    assert "rewindWorkerRun()" in region


def test_both_session_kinds_go_through_that_one_call() -> None:
    """Diffusion editing and What If both open by snapshotting, so
    neither needs its own send. The substitution path is a no-op at
    the worker, which is correct: it never adopts its branch."""
    for anchor in (
        "function beginEditSession()",
        "function beginSubstitutionSession()",
    ):
        region = _region(anchor, 400)
        assert "captureEditSnapshot()" in region, anchor


def test_it_is_sent_once_and_only_from_there() -> None:
    """A second caller would mean a session that rewinds midway,
    which would discard the chain of partial resumes a guided edit
    is in the middle of building."""
    calls = re.findall(r"\brewindWorkerRun\(\)", _source())

    assert len(calls) == 2, "one definition, one caller"


def test_the_chained_path_does_not_reopen_a_session() -> None:
    """`Run to Here` returns to edit mode through the phase machine
    rather than by reopening a session, which is what makes a single
    send at session start safe for a multi-frame edit."""
    region = _region("function handleGuidedDone()", 1200)

    assert "captureEditSnapshot" not in region
    assert "RUN_PHASE_EDIT" in region


# -- what is sent --


def test_the_message_names_the_run() -> None:
    region = _region("function rewindWorkerRun()", 700)

    assert 'type: "rewind"' in region
    assert "run_token: activeRunToken" in region


def test_it_is_skipped_without_a_run_to_name() -> None:
    """Before the first generation there is no token, and no
    retained run to rewind either."""
    region = _region("function rewindWorkerRun()", 700)

    assert "if (!activeRunToken)" in region


def test_it_is_skipped_on_a_closed_socket() -> None:
    region = _region("function rewindWorkerRun()", 700)

    assert "ws.readyState !== WebSocket.OPEN" in region


# -- what is deliberately absent --


def test_the_exit_paths_do_not_send_their_own() -> None:
    """The design under test. Attaching a rewind to Retry and Exit
    would look more direct and would miss the run-scoped error
    unwind, a reload, and a closed tab."""
    for anchor in (
        "function retryGuidedEdit()",
        "function exitRemaskMode()",
    ):
        region = _region(anchor, 400)
        assert "rewindWorkerRun" not in region, anchor
