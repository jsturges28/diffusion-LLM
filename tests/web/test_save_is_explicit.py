"""Nothing is written to disk unless the user asked for it.

Strategy: source inspection of `app.js` and `generator_edit.js`.
What a save *does* is covered in
`test_save_idempotence.py` and `test_run_store.py`; what cannot be
checked there is how many places start one.

Opening an editor used to save. Choosing a frame and marking tokens
are reversible and entirely local, and the run is only destroyed by
the resume that follows, so the write bought nothing that Confirm does
not already do. It cost two things. On a long autoregressive run the
per-frame token records are quadratic in the output length, so merely
opening What If to look at candidates posted megabytes. And an
implicit save races the navigation that follows it, which is how one
generation ended up as two rows in Analytics.

What passing proves is the narrow, checkable half: exactly three
things start a save, and each of them is either the user pressing a
button or a run about to be lost. The behaviour itself needs a GPU and
is items 164 and 165.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)
APP_JS = STATIC / "app.js"
EDIT_JS = STATIC / "generator_edit.js"
RUN_JS = STATIC / "generator_run.js"
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


def _edit_region(anchor: str, chars: int) -> str:
    source = EDIT_JS.read_text(encoding="utf-8")
    start = source.find(anchor)
    assert start != -1, (
        f"anchor {anchor!r} is gone from generator_edit.js;"
        " update this test rather than deleting it"
    )
    return source[start : start + chars]


def _run_region(anchor: str, chars: int) -> str:
    source = RUN_JS.read_text(encoding="utf-8")
    start = source.find(anchor)
    assert start != -1, (
        f"anchor {anchor!r} is gone from generator_run.js;"
        " update this test rather than deleting it"
    )
    return source[start : start + chars]


# -- who may start a save --


def test_only_three_places_start_a_save() -> None:
    """The button, Confirm and rescue share one waiting adapter."""
    source = _app()
    adapter_calls = re.findall(
        r"(?<!function )saveRun\(\)", source
    )
    confirmation_calls = re.findall(
        r"saveRun\(\{ editConfirmation: true \}\)", source
    )
    run_calls = re.findall(r"generatorRun\.save\(\)", source)

    assert len(adapter_calls) == 1
    assert len(confirmation_calls) == 1
    assert len(run_calls) == 1


def test_the_save_button_is_one_of_them() -> None:
    assert 'btnSave.addEventListener("click", saveRun)' in _app()


def test_confirming_an_edit_is_one_of_them() -> None:
    """Confirm is itself a save, so it is not an implicit one."""
    body = _edit_region("function confirm()", 900)

    assert "requestCommit()" in body
    assert "requestSave()" in body


def test_the_rescue_is_the_third() -> None:
    """Not a convenience: another window has replaced the model, the
    run cannot survive it, and the alternative is losing it."""
    region = _region("function rescueRunThenReload()", 700)

    assert "saveRun()" in region


# -- and who may not --


def test_opening_the_frame_editor_writes_nothing() -> None:
    region = _edit_region("function enterFrames()", 700)

    assert "requestSave" not in region


def test_opening_what_if_writes_nothing() -> None:
    region = _edit_region("function enterWhatIf()", 700)

    assert "requestSave" not in region


def test_retrying_an_edit_writes_nothing() -> None:
    """Retry restores the pre-edit run and starts again, which is
    entering a session, not finishing one."""
    region = _edit_region("function retry()", 600)

    assert "requestSave" not in region


def test_no_save_is_gated_on_the_run_being_unsaved() -> None:
    """`if (!runSaved) saveRun()` was the shape of the implicit save,
    at both editor entry points."""
    source = _app()

    assert not re.search(r"!runSaved\s*\)\s*\{\s*saveRun", source)


# -- and the docs say so --


def test_the_help_no_longer_promises_an_automatic_save() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "auto-save" not in html
    assert "saves the original in the background" not in html


def test_the_help_says_what_happens_instead() -> None:
    """A user who relied on the old behaviour needs telling, and the
    replacement rule is short enough to state."""
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "Nothing is written unless you ask for it" in html
    assert "Saving twice cannot duplicate a run" in html


# -- and a run that cannot be saved in full is not saved at all --


def test_a_run_without_its_detail_is_refused() -> None:
    """A long run comes back from its snapshot with the frame text
    and the timings but no per-token detail. Saving it would write
    that hollowed-out version permanently, in place of the run that
    was on screen before the navigation."""
    region = _run_region("function save()", 1400)

    assert "frameLacksDetail()" in region


def test_the_refusal_comes_before_the_request() -> None:
    """Checked with the other reasons not to save, not after the
    payload has been built and posted."""
    region = _run_region("function save()", 2600)
    guard = region.find("frameLacksDetail()")
    post = region.find('requestSave("/api/save"')

    assert guard != -1
    assert post != -1
    assert guard < post


def test_the_refusal_says_why() -> None:
    """A save that silently does nothing is worse than one that
    writes the wrong thing, because nothing tells the user either."""
    region = _run_region("function save()", 1400)

    assert "cannot be saved in full" in region
