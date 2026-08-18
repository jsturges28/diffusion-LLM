"""A run in flight can be stopped, and says so afterwards.

Strategy: source inspection of `app.js` and `index.html`, the
approach this repo uses for its classic-script pages. What the
worker does with a cancel is covered in
`tests/backends/test_worker_dispatch.py`, and what a stopped run
records is covered in `tests/web/test_partial_runs.py`. What
neither can check is the browser half: that a Stop control exists
at all, that it emits the message the worker has always understood
but nothing ever sent, and that a stopped run is not then presented
as a finished one.

The last part is the one worth guarding. The report explicitly
rejected simply clearing the generating flag on disconnect, because
that leaves partial output looking saveable and complete. So the
tests below check not only that the page recovers, but that it
reaches a labelled stopped state and carries that label into the
save.

Passing proves the wiring exists and is reachable. Whether a GPU
actually stops is a manual item.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)
APP_JS = STATIC / "app.js"
INDEX_HTML = STATIC / "index.html"


def _app() -> str:
    return APP_JS.read_text(encoding="utf-8")


def _region(anchor: str, chars: int) -> str:
    source = _app()
    start = source.find(anchor)
    assert start != -1, (
        f"anchor {anchor!r} is gone from app.js; update this test"
        " rather than deleting it"
    )
    return source[start : start + chars]


# -- the client finally sends the message --


def test_the_page_sends_a_cancel() -> None:
    """`MSG_CANCEL` existed for months with no sender at all."""
    source = _app()

    assert 'type: "cancel"' in source


def test_only_the_cancel_helper_sends_it() -> None:
    # One sender, so the guards it applies cannot be bypassed by a
    # second path that grew later.
    source = _app()
    senders = re.findall(r'type:\s*"cancel"', source)

    assert len(senders) == 1
    assert 'type: "cancel"' in _region(
        "function requestCancel()", 700
    )


def test_a_cancel_is_not_sent_without_a_run() -> None:
    """Guarded twice: a live socket and a run to stop."""
    body = _region("function requestCancel()", 700)

    assert "ws.readyState !== WebSocket.OPEN" in body
    assert "if (!isGenerating)" in body


# -- the primary button becomes Stop --


def test_the_button_says_stop_while_a_run_is_in_flight() -> None:
    body = _region("function currentGenerateLabel()", 260)

    assert "isGenerating" in body
    assert '"Stop"' in body


def test_the_button_is_live_while_a_run_is_in_flight() -> None:
    """It used to grey out for the whole run, doing nothing."""
    body = _region("function updateGenerateButton()", 700)

    assert "btnGenerate.disabled = false;" in body


def test_the_click_handler_follows_the_same_order() -> None:
    # Label and action are decided separately, so they can drift:
    # a button reading Stop that starts a run would be worse than
    # no button. Both branch on isGenerating first.
    body = _region("btnGenerate.addEventListener(", 420)

    assert "requestCancel()" in body
    assert body.index("isGenerating") < body.index(
        "editedRunSaved"
    )


# -- a stopped run is not a finished one --


def test_the_terminal_frame_is_read_for_the_stopped_flag() -> None:
    body = _region("function handleDone(data)", 900)

    assert "data.cancelled === true" in body
    assert '"Stopped."' in body


def test_losing_the_socket_reaches_the_stopped_state() -> None:
    """Not merely clearing the flag, which the report rejected."""
    body = _region("ws.onclose = function ()", 900)

    assert "enterInterruptedState()" in body

    interrupted = _region(
        "function enterInterruptedState()", 900
    )
    assert "runInterrupted = true;" in interrupted
    assert "setGenerating(false);" in interrupted


def test_a_stopped_run_keeps_its_frames() -> None:
    """Stopped is not discarded: the output stays usable."""
    body = _region("function enterInterruptedState()", 900)

    assert "setSaveAvailable(true)" in body
    assert "activateScrubber()" in body


def test_the_save_carries_the_stopped_flag() -> None:
    source = _app()

    assert "payload.partial = true;" in source
    assert "if (runInterrupted) {" in source


def test_the_flag_survives_a_trip_to_analytics() -> None:
    """Otherwise a stopped run returns looking complete."""
    source = _app()

    assert "runInterrupted: runInterrupted," in source
    assert "runInterrupted = !!s.runInterrupted;" in source


def test_a_fresh_run_clears_the_flag() -> None:
    body = _region("function resetRunState()", 1400)

    assert "runInterrupted = false;" in body


# -- the in-app docs keep up --


def test_help_explains_stopping() -> None:
    help_text = INDEX_HTML.read_text(encoding="utf-8")

    assert "Stopping a run" in help_text
    assert "Leaving the page also stops the run" in help_text


def test_help_does_not_still_say_generate_greys_out() -> None:
    # The old sentence described the behaviour this change removes.
    help_text = INDEX_HTML.read_text(encoding="utf-8")

    assert (
        "<strong>Generate</strong> is disabled and the status bar"
        not in help_text
    )
