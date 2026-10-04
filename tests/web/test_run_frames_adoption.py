"""The generator reaches its frame arrays only through their owner.

Strategy: read `generator_run.js`, `generator_edit.js`, the shipped
`app.js` and `index.html`. The operations themselves are unit-tested
in `tests/web/static/run_frames.test.js`, which drives the module in
a `vm`; `generator_run.test.js` drives their active-run owner. What
neither can check is whether the page still goes around that owner.
A family one call site can take apart is not a family.

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
RUN_JS = STATIC / "generator_run.js"
EDIT_JS = STATIC / "generator_edit.js"
SNAPSHOT_JS = STATIC / "run_snapshot.js"

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


def _snapshot() -> str:
    """The snapshot codec, which serialises both families for the
    page and reads them back."""
    return SNAPSHOT_JS.read_text(encoding="utf-8")


def _run() -> str:
    """The active-run controller that holds both frame families."""
    return RUN_JS.read_text(encoding="utf-8")


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
    source = _run()

    assert source.count("var frames = runFramesCreate()") == 1
    assert "var runFrames =" not in _app()


def test_the_family_is_never_reassigned() -> None:
    """Mutated in place, so a reference taken anywhere stays valid.
    Reassignment is how six separate variables became awkward to hold
    together in the first place."""
    source = _run()
    writes = re.findall(r"(?<![\w.$])frames\s*=(?!=)", source)

    assert len(writes) == 1


# -- and the module is actually there --


def test_the_module_loads_before_the_page_that_uses_it() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "/run_frames.js" in html
    assert html.index("/run_frames.js") < html.index("/app.js")
    assert html.index("/run_snapshot.js") < html.index(
        "/generator_run.js"
    )
    assert html.index("/generator_run.js") < html.index("/app.js")


def test_the_module_touches_no_dom() -> None:
    """What keeps it drivable in a vm, and what the other extracted
    modules already hold to."""
    source = MODULE_JS.read_text(encoding="utf-8")

    assert "document" not in source
    assert "window" not in source


# -- every former enumeration site now delegates --


def _region(path: Path, anchor: str, chars: int) -> str:
    source = path.read_text(encoding="utf-8")
    start = source.find(anchor)
    assert start != -1, (
        f"anchor {anchor!r} is gone from {path.name};"
        " update this test"
        " rather than deleting it"
    )
    return source[start : start + chars]


def test_a_frame_arrives_through_append() -> None:
    region = _region(
        RUN_JS, "function appendSnapshotFrame(data)", 900
    )

    assert "runFramesAppend(frames, {" in region
    append = _region(
        RUN_JS, "function appendPositionFrame(data)", 900
    )
    assert "runFramesAppendPosition(frames, {" in append


def test_an_edit_snapshot_is_taken_by_the_module() -> None:
    region = _region(RUN_JS, "function captureCheckpoint()", 700)

    assert "runFramesSnapshot(frames)" in region
    assert "run.captureCheckpoint()" in _region(
        EDIT_JS, "function capturePreEditCheckpoint()", 500
    )


def test_an_edit_rollback_goes_back_through_it() -> None:
    region = _region(RUN_JS, "function restoreCheckpoint(", 700)

    assert "runFramesRestore(frames" in region
    assert "run.restoreCheckpoint(" in _region(
        EDIT_JS, "function restorePreEditCheckpoint()", 600
    )


def test_a_resume_truncates_through_it() -> None:
    """The site whose hand-written list dropped `perFrameElapsed` and
    knocked the Timing chart's x axis out of step."""
    assert "runFramesTruncate(frames, offset)" in _region(
        RUN_JS, "function truncate(offset)", 700
    )
    assert "run.truncate(frameIndex)" in _region(
        EDIT_JS, "function resumeGuided(action)", 1800
    )


def test_a_fresh_run_clears_through_it() -> None:
    assert "runFramesClear(frames)" in _region(
        RUN_JS, "function reset()", 900
    )
    assert "generatorRun.reset()" in _region(
        APP_JS, "function resetRunState()", 600
    )


def test_the_snapshot_is_serialised_through_it() -> None:
    """Both payloads: the light one that survives a storage-quota
    refusal, and the full one. The codec serialises the family the
    page hands it."""
    codec = _snapshot()
    light = "runFramesToJson(record.frames, RUN_FRAME_LIGHT_FIELDS)"
    handed = _region(RUN_JS, "function sessionRecord()", 2200)

    assert light in codec
    assert "runFramesToJson(record.frames)" in codec
    assert "frames: frames," in handed


def test_the_snapshot_is_read_back_through_it() -> None:
    region = _region(
        RUN_JS, "function applyRestored(restored)", 600
    )

    assert "runFramesFromJson(source)" in _snapshot()
    assert "runFramesRestore(frames, restored.frames)" in region


# -- and so does the baseline --


def test_the_baseline_is_declared_once_and_held() -> None:
    source = _run()
    writes = re.findall(r"(?<![\w.$])original\s*=(?!=)", source)

    assert source.count("var original = originalRunCreate()") == 1
    assert len(writes) == 1


def test_the_baseline_is_frozen_through_the_module() -> None:
    """The module refuses a second capture, so the guard that used to
    sit around this at the call site is gone."""
    region = _region(RUN_JS, "function finish(data)", 1400)

    assert "originalRunCapture(original, frames" in region


def test_the_baseline_is_cleared_and_stored_through_it() -> None:
    """Stored and read back by the codec, and put back in place by the
    page, into the one baseline it holds."""
    source = _run()
    codec = _snapshot()
    assigned = "originalRunAssign(original, restored.original)"

    assert "originalRunClear(original)" in source
    assert "originalRunToJson(record.original)" in codec
    assert "originalRunFromJson(" in codec
    assert assigned in source
