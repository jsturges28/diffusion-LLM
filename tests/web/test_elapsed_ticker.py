"""The elapsed line advances between frames, not only on them.

Strategy: source inspection of `app.js`, the approach this repo uses
for its classic-script pages.

The reading used to move only when a frame landed, because
`updateRunRateFooter` has one caller and that caller is
`handleFrame`. Smoothness was never the point: a wedged run and a
merely slow one looked identical, since both simply stopped
counting.

The number is the worker's own `time.monotonic` measurement, and it
is the same figure that reaches the saved run and the Analytics
duration. So this interpolates from a stamp rather than running a
browser clock, which would count the socket hop and the render too
and drift above the record. That is the same class of disagreement
`ANALYTICS-02` fixed between the live throughput and the chart, and
the reason these tests pin the direction of every read.

Passing proves the tick exists, that it extrapolates from the
worker's value rather than replacing it, that a finished run comes
to rest on the worker's exact figure, and that no timer outlives the
readout it writes to.
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


# -- it moves on its own --


def test_a_ticker_drives_the_reading() -> None:
    body = _region("function elapsedTick()", 500)

    assert "setInterval" in body
    assert "ELAPSED_TICK_MS" in body


def test_the_ticker_is_started_by_a_frame() -> None:
    body = _region("function updateRunRateFooter()", 500)

    assert "elapsedTick()" in body


def test_starting_twice_does_not_stack_timers() -> None:
    """A frame calls this every time, so the guard is what keeps one
    run from ending with a hundred intervals writing the same line."""
    body = _region("function elapsedTick()", 500)

    assert "if (elapsedTimer !== null)" in body
    assert "return" in body


# -- it extrapolates rather than replaces --


def test_a_frame_stamps_the_worker_value() -> None:
    body = _region("function updateRunRateFooter()", 500)

    stamped = "elapsedStampSeconds = runFrames.elapsed[frames - 1]"
    assert stamped in body
    assert "elapsedStampAt = Date.now()" in body


def test_the_tick_adds_local_time_to_that_stamp() -> None:
    """Not `Date.now() - runStart`: the displayed number has to stay
    the worker's, or the page and the saved run disagree."""
    body = _region("function elapsedTick()", 600)

    assert "Date.now() - elapsedStampAt" in body
    assert "elapsedStampSeconds + since" in body


def test_the_reading_has_one_formatter() -> None:
    """Three writers of the same string would be three chances for
    the tick and the frame to disagree about precision."""
    source = _source()
    built = re.findall(
        r'"Elapsed: " \+ seconds\.toFixed', source
    )

    assert len(built) == 1


# -- it comes to rest on the truth --


def test_a_finished_run_settles_on_the_worker_figure() -> None:
    """The last tick will have extrapolated past the final frame by
    up to one interval. Landing there would leave the page showing a
    duration the saved run does not carry."""
    body = _region("function elapsedSettle()", 400)

    assert "renderElapsed(elapsedStampSeconds)" in body
    assert "elapsedStop()" in body


def test_every_terminal_path_settles_it() -> None:
    """A done frame, a cancelled one, a dropped connection and a
    run-scoped error all end here, which is why the call is in one
    place rather than four."""
    body = _region("function endRunStatus()", 400)

    assert "elapsedSettle()" in body


# -- and nothing outlives it --


def test_clearing_the_footer_stops_the_ticker_first() -> None:
    """Order matters: a tick still in flight would paint a stale
    number back over the dash a fraction of a second later."""
    body = _region("function resetStatus()", 500)
    stopped = body.find("elapsedStop()")
    cleared = body.find('"Elapsed: -"')

    assert stopped != -1
    assert cleared != -1
    assert stopped < cleared


def test_stopping_forgets_the_stamp() -> None:
    """Otherwise a later settle would repaint the previous run's
    duration over a footer that had moved on."""
    body = _region("function elapsedStop()", 400)

    assert "clearInterval(elapsedTimer)" in body
    assert "elapsedTimer = null" in body
    assert "elapsedStampSeconds = null" in body


def test_the_timer_is_only_created_in_one_place() -> None:
    source = _source()
    created = re.findall(r"elapsedTimer = setInterval", source)

    assert len(created) == 1


# -- throughput is deliberately left alone --


def test_the_rate_still_moves_only_on_a_frame() -> None:
    """Decided rather than overlooked. A rate that decayed between
    frames would be informative while a run is stuck and restless
    every other second, so only the elapsed line ticks."""
    body = _region("function elapsedTick()", 600)

    assert "renderTpsFooter" not in body
    assert "currentTokensPerSecond" not in body


def test_the_rate_reads_the_worker_series_not_a_clock() -> None:
    """What keeps the footer and the Analytics chart agreeing."""
    body = _region("function currentTokensPerSecond()", 900)

    assert "runFrames.elapsed" in body
    assert "Date.now()" not in body
