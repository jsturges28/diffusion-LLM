"""The run's provenance survives the trip through the browser.

Strategy: inspect the run controller and its app.js transport wiring.
The controller has a DOM-free factory and is also driven by
`generator_run.test.js`. The two ends of this path are tested in
`tests/backends/test_worker_provenance.py` (the worker attests) and
`tests/web/test_run_provenance.py` (the save prefers it).

What passing proves is that the middle of the path exists at every
point it has to. The envelope arrives on a run's opening frame and
again on its done frame, is held across the run, is sent with the
save, survives a trip to Analytics and back, and is dropped when a
new run starts. Miss any one of those and the fix silently reverts
to the old behavior: the save just goes back to describing whichever
model is resident, which looks completely normal until the moment it
does not. The frame and done halves are also driven for real, in
`tests/web/static/generator_run_provenance.test.js`.

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


SNAPSHOT_JS = APP_JS.with_name("run_snapshot.js")
RUN_JS = APP_JS.with_name("generator_run.js")


def _source(path: Path = APP_JS) -> str:
    return path.read_text(encoding="utf-8")


def _region(path: Path, anchor: str, chars: int) -> str:
    source = _source(path)
    start = source.find(anchor)
    assert start != -1, (
        f"anchor {anchor!r} is gone from {path.name}; update this"
        " test rather than deleting it"
    )
    return source[start : start + chars]


def test_the_envelope_is_captured_from_the_done_frame() -> None:
    region = _region(APP_JS, "function handleDone(data)", 1200)

    assert "generatorRun.finish(data)" in region


def test_the_envelope_is_captured_from_the_opening_frame() -> None:
    """A run whose connection drops never sends a done frame, so its
    first frame is the only place its envelope can come from."""
    region = _region(APP_JS, "function handleFrame(data)", 500)

    assert "generatorRun.appendFrame(data)" in region


def test_adopting_an_envelope_records_it() -> None:
    region = _region(
        RUN_JS, "function adoptProvenance(data)", 300
    )

    assert "data.provenance" in region
    assert "provenance = data.provenance" in region


def test_the_envelope_is_submitted_with_the_save() -> None:
    source = _source(RUN_JS)

    assert "payload.provenance = copyJson(provenance)" in source


def test_the_save_omits_it_rather_than_sending_null() -> None:
    """Absent means "this run predates provenance", which is what
    makes the server fall back. Sending null would have to be
    special-cased on the far side to mean the same thing."""
    source = _source(RUN_JS)

    guard = source.find("if (provenance !== null)")
    assignment = source.find(
        "payload.provenance = copyJson(provenance)"
    )

    assert guard != -1
    assert guard < assignment


def test_the_envelope_survives_a_trip_to_analytics() -> None:
    """The session snapshot is the gap in which another window can
    switch the model, so it is exactly where this must not be
    dropped. The snapshot codec reads it back; the page applies it."""
    source = _source(RUN_JS)
    codec = SNAPSHOT_JS.read_text(encoding="utf-8")
    guard = (
        'source.provenance && typeof source.provenance === "object"'
    )

    assert "provenance: provenance" in source
    assert guard in codec
    assert "provenance = restored.provenance;" in source


def test_a_new_run_clears_the_previous_envelope() -> None:
    """Otherwise one run's facts would be attached to another run's
    text, which is worse than having none."""
    source = _source(RUN_JS)

    assert "provenance = null;" in source


def test_the_envelope_is_cleared_beside_the_prompt_length() -> None:
    """These are per-run facts read off the same frames, so they are
    reset together; separating them is how one gets forgotten.

    The set grows. ``lastRunTotalSteps`` joined it when the scrubber
    needed the step total to rebuild its readout, and it has exactly
    the same failure mode: a value from the previous run surviving
    into the next one, describing a run that is no longer on screen.
    """
    source = _source(RUN_JS)
    together = (
        "promptLength = null;\n"
        "    provenance = null;\n"
        "    totalSteps = null;"
    )

    assert together in source
