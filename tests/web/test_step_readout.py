"""The step reading follows the scrubber instead of freezing.

Strategy: source inspection of `app.js`, the approach this repo uses
for its classic-script pages. The formatting itself is trivial; what
needs guarding is that two callers share it and that the value it
needs survives a trip to Analytics.

The bug: `statusStep` was written in exactly two places, the live
frame handler and the session restore. `navigateToFrame` updated only
the "Frame N / M" label beside it, so scrubbing a finished
DiffusionGemma run left "Step 87, Canvas 4" on screen at every frame,
describing the run's end rather than the frame being looked at.

Two callers means two chances to drift, which is why they share one
formatter rather than each building the string. They do supply
different step numbers on purpose: during a resume the live view
counts the branch's own steps, while the scrubber counts the whole
run, which is the number the label beside it already shows.

Passing proves the formatter exists and both paths use it, that the
scrubber calls it, that the step total is captured rather than
discarded, and that it survives the session snapshot so scrubbing
still works after coming back from Analytics.

It also pins which of the two numbers each readout gets. They were
one variable at first, and a resume overwrote it with the steps its
branch had left. A run edited at frame 64 of 128 then scrubbed to
"Step 128/64", and abandoning a branch with Retry left the finished
run measured against the segment that no longer existed.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)
APP_JS = STATIC / "app.js"


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


# -- one formatter, two callers --


def test_the_formatter_exists() -> None:
    source = _app()

    assert (
        "function stepReadout(step, canvasIndex, totalSteps, prefix)"
        in source
    )


def test_both_shapes_live_in_the_formatter() -> None:
    """A fixed schedule reports out of a total; an adaptive-stopping
    model has no total and names its canvas instead."""
    body = _region("function stepReadout(", 700)

    assert '"/"' in body
    assert '", Canvas "' in body
    assert "canvas + 1" in body


def test_the_live_path_uses_it() -> None:
    body = _region("function updateLiveFrameStatus(data)", 900)

    assert "statusStep.textContent = stepReadout(" in body
    assert '"Resuming "' in body


def test_both_frame_shapes_reach_the_live_path() -> None:
    """A snapshot frame and an append frame put the same reading on
    screen, because they call the same writer. Two copies of it
    would be two chances for one shape's live line to drift from the
    other's."""
    for handler in (
        "function handleFrame(data)",
        "function handleAppendFrame(data)",
    ):
        assert "updateLiveFrameStatus(data);" in _region(
            handler, 2000
        ), handler


def test_the_scrubber_uses_it() -> None:
    body = _region("function renderScrubStepReadout(index)", 700)

    assert "statusStep.textContent = stepReadout(" in body
    assert "runFrames.canvasIndex[index]" in body


def test_navigating_repaints_the_readout() -> None:
    """The whole bug: this call did not exist."""
    body = _region("function navigateToFrame(index)", 900)

    assert "renderScrubStepReadout(index)" in body


def test_nothing_else_formats_the_reading() -> None:
    # Two writers of the finished string, the formatter's callers:
    # the live line and the scrubbed one. A third would be the drift
    # this consolidation prevents. The session restore and the idle
    # reset assign literals, which is not the same thing.
    #
    # Both frame shapes go through the live writer rather than each
    # bringing its own, which is the reason this still reads two
    # after a second delivery shape arrived.
    source = _app()
    built = re.findall(
        r"statusStep\.textContent = stepReadout\(", source
    )

    assert len(built) == 2


# -- the value the scrubber needs survives --


def test_the_step_total_is_captured() -> None:
    """It was read off every frame and thrown away, which is why the
    scrubber could not rebuild the reading."""
    source = _app()

    assert "var lastRunTotalSteps = null;" in source
    assert "lastRunTotalSteps =" in _region(
        "function updateLiveFrameStatus(data)", 900
    )


def test_a_resume_does_not_claim_the_run_total() -> None:
    """The two numbers, kept apart.

    A resume's frames report the steps that branch has left, which
    is the right figure for the live line and the wrong one for a
    scrubber counting the whole run.
    """
    body = _region("function updateLiveFrameStatus(data)", 900)
    guarded = body.find("if (!isResuming) {")
    written = body.find("lastRunTotalSteps = frameSteps;")

    assert guarded != -1, "the run total is written unguarded"
    assert written != -1
    assert 0 < written - guarded < 80


def test_the_live_line_reads_the_frame_not_the_run() -> None:
    """Otherwise the guard above would freeze the live readout at
    the generation's total and "Resuming 12/64" would read
    "Resuming 12/128"."""
    body = _region("function updateLiveFrameStatus(data)", 900)
    start = body.find("statusStep.textContent = stepReadout(")
    call = body[start : start + 220]

    assert "frameSteps" in call
    assert "lastRunTotalSteps" not in call


def test_the_scrubber_reads_the_run_not_the_frame() -> None:
    body = _region("function renderScrubStepReadout(index)", 700)

    assert "lastRunTotalSteps" in body
    assert "frameSteps" not in body


def test_substitution_counts_as_a_resume() -> None:
    """What If reuses the resume splice path, so its branch must
    not claim the run total either. For an autoregressive run the
    two numbers coincide, which would make this right by accident
    rather than on purpose."""
    body = _region("function doSubstitute(", 1400)

    assert "isResuming = true;" in body


def test_a_fresh_run_clears_it() -> None:
    body = _region("function resetRunState()", 1500)

    assert "lastRunTotalSteps = null;" in body


def test_it_survives_a_trip_to_analytics() -> None:
    source = _app()

    assert "lastRunTotalSteps: lastRunTotalSteps," in source
    assert "typeof s.lastRunTotalSteps === \"number\"" in source


def test_an_adaptive_model_restores_as_having_no_total() -> None:
    # null, not zero: a zero total would render "Step 5/0" rather
    # than falling through to the canvas form.
    body = _region('typeof s.lastRunTotalSteps === "number"', 200)

    assert ": null" in body
