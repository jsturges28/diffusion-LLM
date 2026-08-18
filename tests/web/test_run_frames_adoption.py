"""The generator reaches its frame arrays only through their owner.

Strategy: read the shipped `app.js` and `index.html`. The operations
themselves are unit-tested in
`tests/web/static/run_frames.test.js`, which drives the module in a
`vm`; what cannot be checked there is whether anything still goes
around it. A family that one call site can still take apart is not a
family, and the invariant the module asserts is only worth as much as
the number of paths that cannot avoid it.

What passing proves is that the count went from nine to zero. Six
arrays indexed by frame were declared separately and enumerated by
hand at nine places: appended, frozen into the original-run copy,
snapshotted, restored, truncated, cleared, projected into the save
payload, serialised, and read back. `ORG-02` exists because adding a
seventh meant getting all nine right, and the comment on the old
`truncateRunArraysAt` records the one that was missed.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)
APP_JS = STATIC / "app.js"
INDEX_HTML = STATIC / "index.html"
MODULE_JS = STATIC / "run_frames.js"

# What the two families used to be called as free variables. The
# second is the baseline: the run as it was before the first edit,
# frozen once and read by everything that compares an edited run
# against what it branched from.
FORMER_NAMES = (
    "frameHistory",
    "frameTokens",
    "frameCanvasIndex",
    "frameMeanConf",
    "perFrameElapsed",
    "frameRevealed",
    "originalFrameHistory",
    "originalFrameTokens",
    "originalPerFrameElapsed",
    "originalMeanConf",
    "originalPositionAlts",
    "originalTotalFrames",
)


def _app() -> str:
    return APP_JS.read_text(encoding="utf-8")


# -- nothing reaches around the module --


def test_no_array_is_a_variable_of_its_own_any_more() -> None:
    """A bare mention would be a seventh path to the arrays, and the
    one that forgets a sibling. Matched as a whole word not preceded
    by a dot, so the wire key names in a snapshot are untouched."""
    source = _app()

    for name in FORMER_NAMES:
        pattern = r"(?<![\w.$])" + name + r"(?![\w$])"
        assert re.search(pattern, source) is None, name


def test_the_family_is_declared_once() -> None:
    source = _app()

    assert source.count("var runFrames = runFramesCreate()") == 1


def test_the_family_is_never_reassigned() -> None:
    """Mutated in place, so a reference taken anywhere stays valid.
    Reassignment is how six separate variables became awkward to hold
    together in the first place."""
    source = _app()
    writes = re.findall(r"(?<![\w.$])runFrames\s*=(?!=)", source)

    assert len(writes) == 1


# -- and the module is actually there --


def test_the_module_loads_before_the_page_that_uses_it() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "/run_frames.js" in html
    assert html.index("/run_frames.js") < html.index("/app.js")


def test_the_module_touches_no_dom() -> None:
    """What keeps it drivable in a vm, and what the other extracted
    modules already hold to."""
    source = MODULE_JS.read_text(encoding="utf-8")

    assert "document" not in source
    assert "window" not in source


# -- every former enumeration site now delegates --


def _region(anchor: str, chars: int) -> str:
    source = _app()
    start = source.find(anchor)
    assert start != -1, (
        f"anchor {anchor!r} is gone from app.js; update this test"
        " rather than deleting it"
    )
    return source[start : start + chars]


def test_a_frame_arrives_through_append() -> None:
    assert "runFramesAppend(runFrames, {" in _region(
        "function handleFrame(data)", 1400
    )


def test_an_edit_snapshot_is_taken_by_the_module() -> None:
    assert "runFramesSnapshot(runFrames)" in _region(
        "function captureEditSnapshot()", 500
    )


def test_an_edit_rollback_goes_back_through_it() -> None:
    assert "runFramesRestore(runFrames" in _region(
        "function restoreEditSnapshot()", 700
    )


def test_a_resume_truncates_through_it() -> None:
    """The site whose hand-written list dropped `perFrameElapsed` and
    knocked the Timing chart's x axis out of step."""
    assert "runFramesTruncate(runFrames, offset)" in _region(
        "function truncateRunArraysAt(offset)", 700
    )


def test_a_fresh_run_clears_through_it() -> None:
    assert "runFramesClear(runFrames)" in _region(
        "function resetRunState()", 900
    )


def test_the_snapshot_is_serialised_through_it() -> None:
    """Both payloads: the light one that survives a storage-quota
    refusal, and the full one."""
    region = _region("function saveSessionState()", 2600)
    light = "runFramesToJson(runFrames, RUN_FRAME_LIGHT_FIELDS)"

    assert light in region
    assert "runFramesToJson(runFrames)" in region


def test_the_snapshot_is_read_back_through_it() -> None:
    region = _region("function restoreSessionState()", 1200)

    assert "runFramesFromJson(s)" in region
    assert "runFramesRestore(runFrames, restored)" in region


# -- and so does the baseline --


def test_the_baseline_is_declared_once_and_held() -> None:
    source = _app()
    writes = re.findall(r"(?<![\w.$])originalRun\s*=(?!=)", source)

    assert source.count("var originalRun = originalRunCreate()") == 1
    assert len(writes) == 1


def test_the_baseline_is_frozen_through_the_module() -> None:
    """The module refuses a second capture, so the guard that used to
    sit around this at the call site is gone."""
    region = _region("function handleDone(data)", 2000)

    assert "originalRunCapture(originalRun, runFrames" in region


def test_the_baseline_is_cleared_and_stored_through_it() -> None:
    source = _app()

    assert "originalRunClear(originalRun)" in source
    assert "originalRunToJson(originalRun)" in source
    assert "originalRunRestore(originalRun, s," in source
