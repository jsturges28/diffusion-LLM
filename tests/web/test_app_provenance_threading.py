"""The run's provenance survives the trip through the browser.

Strategy: source inspection of `app.js`, for the same reason
`test_analytics_escaping.py` inspects `analytics.js`: it is a classic
script that reaches for the DOM at load and cannot be imported into a
test. Giving it a testable seam is `ORG-02` in stage 5. The two ends
of this path are tested by execution, in
`tests/backends/test_worker_provenance.py` (the worker attests) and
`tests/web/test_run_provenance.py` (the save prefers it).

What passing proves is that the middle of the path exists at every
point it has to. The envelope arrives on the done frame, is held
across the run, is sent with the save, survives a trip to Analytics
and back, and is dropped when a new run starts. Miss any one of
those and the fix silently reverts to the old behavior: the save
just goes back to describing whichever model is resident, which
looks completely normal until the moment it does not.

The last one is the easiest to forget and the worst to get wrong. A
provenance left over from a previous run would attach one run's facts
to another run's text, which is a wrong record rather than a missing
one.
"""

from __future__ import annotations

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
        f"anchor {anchor!r} is gone; update this test rather than"
        " deleting it"
    )
    return source[start : start + chars]


def test_the_envelope_is_captured_from_the_done_frame() -> None:
    region = _region("function handleDone(data)", 2000)

    assert "data.provenance" in region
    assert "lastRunProvenance = data.provenance" in region


def test_the_envelope_is_submitted_with_the_save() -> None:
    source = _source()

    assert "payload.provenance = lastRunProvenance" in source


def test_the_save_omits_it_rather_than_sending_null() -> None:
    """Absent means "this run predates provenance", which is what
    makes the server fall back. Sending null would have to be
    special-cased on the far side to mean the same thing."""
    source = _source()

    guard = source.find("if (lastRunProvenance !== null)")
    assignment = source.find(
        "payload.provenance = lastRunProvenance"
    )

    assert guard != -1
    assert guard < assignment


def test_the_envelope_survives_a_trip_to_analytics() -> None:
    """The session snapshot is the gap in which another window can
    switch the model, so it is exactly where this must not be
    dropped."""
    source = _source()

    assert "provenance: lastRunProvenance" in source
    assert "lastRunProvenance =\n    s.provenance" in source


def test_a_new_run_clears_the_previous_envelope() -> None:
    """Otherwise one run's facts would be attached to another run's
    text, which is worse than having none."""
    source = _source()

    assert "lastRunProvenance = null;" in source


def test_the_envelope_is_cleared_beside_the_prompt_length() -> None:
    """These are per-run facts read off the same frames, so they are
    reset together; separating them is how one gets forgotten.

    The set grows. ``lastRunTotalSteps`` joined it when the scrubber
    needed the step total to rebuild its readout, and it has exactly
    the same failure mode: a value from the previous run surviving
    into the next one, describing a run that is no longer on screen.
    """
    source = _source()
    together = (
        "lastRunPromptLen = null;\n"
        "  lastRunTotalSteps = null;\n"
        "  lastRunProvenance = null;"
    )

    assert together in source
