"""Nothing is written to disk unless the user asked for it.

Strategy: source inspection of `app.js`, the approach this repo uses
for its classic-script pages. What a save *does* is covered in
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


# -- who may start a save --


def test_only_two_places_call_a_save() -> None:
    """Confirm and the rescue. The Save button is the third trigger
    and passes the function by reference, so it is counted by its own
    test below rather than here."""
    source = _app()
    calls = re.findall(r"(?<!function )saveRun\(\)", source)

    assert len(calls) == 2


def test_the_save_button_is_one_of_them() -> None:
    assert 'btnSave.addEventListener("click", saveRun)' in _app()


def test_confirming_an_edit_is_one_of_them() -> None:
    """Confirm is itself a save, so it is not an implicit one."""
    assert "saveRun()" in _region("function confirmGuidedEdit()", 400)


def test_the_rescue_is_the_third() -> None:
    """Not a convenience: another window has replaced the model, the
    run cannot survive it, and the alternative is losing it."""
    region = _region("function rescueRunThenReload()", 700)

    assert "saveRun()" in region


# -- and who may not --


def test_opening_the_frame_editor_writes_nothing() -> None:
    region = _region("function enterRemaskMode()", 900)

    assert "saveRun" not in region


def test_opening_what_if_writes_nothing() -> None:
    region = _region("function enterSubstitutionMode()", 700)

    assert "saveRun" not in region


def test_retrying_an_edit_writes_nothing() -> None:
    """Retry restores the pre-edit run and starts again, which is
    entering a session, not finishing one."""
    region = _region("function retryGuidedEdit()", 600)

    assert "saveRun" not in region


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
